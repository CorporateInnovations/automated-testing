"""
Minimal agent runner with a two-step loop:
1) Extract page meaning (structured digest)
2) Choose and execute an action like a human (dismiss popups first)

Includes lightweight heuristics, validation, and retry/backoff.
"""
import argparse
import asyncio
import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeoutError
from tools.agent_runner.agents.action_agent import choose_action
from tools.agent_runner.agents.meaning_agent import extract_meaning
from tools.agent_runner.page_model import build_page_map, collect_clickable_elements


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env(env_path: str = ".env") -> None:
  if not os.path.exists(env_path):
    return
  with open(env_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      k, v = line.split("=", 1)
      if k and v and k not in os.environ:
        os.environ[k] = v


def trim(text: str, max_len: int) -> str:
  if not text:
    return ""
  return text[:max_len]


def ensure_tmp_dir() -> str:
  tmp_dir = os.path.join(os.getcwd(), "tmp")
  os.makedirs(tmp_dir, exist_ok=True)
  os.environ["TMPDIR"] = tmp_dir
  os.environ["TMP"] = tmp_dir
  os.environ["TEMP"] = tmp_dir
  return tmp_dir


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Python agent runner")
  parser.add_argument("plan", help="Path to plan JSON")
  parser.add_argument("--tests", help="Comma-separated test case IDs to run", default=None)
  parser.add_argument("--headed", action="store_true", help="Run headed (browser visible)")
  parser.add_argument("--model", help="OpenAI model", default=None)
  return parser.parse_args()


def load_plan_arg(plan_arg: str) -> Dict[str, Any]:
  """Load plan from a file path or a JSON string."""
  plan_arg = plan_arg or ""
  if plan_arg.strip().startswith("{"):
    return json.loads(plan_arg)
  plan_path = os.path.abspath(plan_arg)
  with open(plan_path, "r", encoding="utf-8") as f:
    return json.load(f)


def normalize_memory(memory: Optional[Dict[str, str]]) -> Dict[str, str]:
  memory = memory or {}
  return {
    "evaluation_previous_goal": memory.get("evaluation_previous_goal", ""),
    "memory": memory.get("memory", ""),
    "next_goal": memory.get("next_goal", ""),
  }


def openai_client() -> OpenAI:
  key = os.environ.get("OPENAI_API_KEY")
  if not key:
    raise RuntimeError("OPENAI_API_KEY not set")
  return OpenAI(api_key=key)


# ---------------------------------------------------------------------------
# Page capture and digestion
# ---------------------------------------------------------------------------

def digest_for_action(page_map: Dict[str, Any], limit: int = 40) -> List[Dict[str, Any]]:
  digest = page_map.get("digest", {}) or {}
  refs = page_map.get("refs", []) or []
  why_by_ref = {}
  for ref_list, why in (
    (digest.get("cookies") or [], "cookie"),
    (digest.get("nav") or [], "nav"),
    (digest.get("ctas") or [], "cta"),
    (digest.get("subnav") or [], "nav"),
  ):
    for el in ref_list:
      why_by_ref[el.get("ref")] = why

  scored = []
  for el in refs:
    ref = el.get("ref")
    text = (el.get("text") or "").lower()
    score = 0
    score += 50 if why_by_ref.get(ref) == "cookie" else 0
    score += 30 if "about" in text else 0
    score += 25 if "client" in text else 0
    score += 20 if why_by_ref.get(ref) == "nav" else 0
    score += 10 if why_by_ref.get(ref) == "cta" else 0
    score += max(0, 10 - (el.get("ref") or 0))
    scored.append((score, el))

  scored.sort(key=lambda t: t[0], reverse=True)
  selected = [el for _, el in scored[:limit]]
  digest_list = []
  for el in selected:
    digest_list.append(
      {
        "ref": el.get("ref"),
        "tag": el.get("tag"),
        "text": trim(el.get("text", ""), 80),
        "aria": trim(el.get("aria", ""), 80),
        "role": el.get("role", ""),
        "bbox": el.get("bbox", {}),
        "why": why_by_ref.get(el.get("ref"), "other"),
      }
    )
  return digest_list


async def capture_state(page: Page) -> Dict[str, Any]:
  try:
    await page.wait_for_load_state("networkidle")
  except Exception:
    pass
  screenshot = await page.screenshot(type="png", timeout=10000)
  accessibility = None
  try:
    accessibility = await page.accessibility.snapshot(interesting_only=True)
  except Exception:
    accessibility = None
  clickables = await collect_clickable_elements(page)
  try:
    raw_html = await page.content()
  except Exception:
    raw_html = ""
  visible_text = ""
  try:
    visible_text = await page.evaluate("() => document.body.innerText || ''")
  except Exception:
    visible_text = ""
  try:
    title = await page.title()
  except Exception:
    title = ""
  return {
    "screenshotBase64": base64.b64encode(screenshot).decode("utf-8"),
    "accessibilityText": trim(json.dumps(accessibility or {}, indent=2), 3000),
    "clickables": clickables,
    "rawHtml": raw_html,
    "visibleText": trim(visible_text, 2000),
    "pageUrl": page.url,
    "pageTitle": title,
  }


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

async def auto_dismiss_popups(page: Page) -> Optional[str]:
  close_labels = ["close", "dismiss", "no thanks", "x", "✕"]
  accept_labels = ["accept", "allow all", "agree", "got it", "yes", "consent"]
  for text in close_labels + accept_labels:
    try:
      btn = page.get_by_role("button", name=re.compile(text, re.IGNORECASE)).first
      if await btn.is_visible(timeout=300):
        await btn.click(timeout=1200, force=True)
        await page.wait_for_timeout(200)
        return text
    except Exception:
      pass
  return None


async def click_with_fallback(page: Page, el: Dict[str, Any], hover_first: bool = False) -> None:
  if not el:
    raise ValueError("No element to click")
  selector = el.get("selector")
  strategies = el.get("strategies") or []
  bbox = el.get("bbox") or el.get("boundingBox")

  async def perform(locator):
    if hover_first:
      try:
        await locator.hover(timeout=1500)
        await page.wait_for_timeout(150)
      except Exception:
        pass
    await locator.click(timeout=8000, force=True)

  for strat in strategies:
    try:
      if strat["type"] == "role":
        v = strat.get("value") or {}
        await perform(page.get_by_role(v.get("role"), name=re.compile(v.get("name", ""), re.IGNORECASE)))
        return
      if strat["type"] == "aria":
        await perform(page.get_by_label(re.compile(strat["value"], re.IGNORECASE)))
        return
      if strat["type"] == "text":
        await perform(page.get_by_text(re.compile(strat["value"], re.IGNORECASE), exact=False))
        return
      if strat["type"] == "css":
        await perform(page.locator(strat["value"]))
        return
    except Exception:
      continue
  if selector:
    try:
      await perform(page.locator(selector))
      return
    except Exception:
      pass
  if bbox:
    await page.mouse.click(bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)
    return
  raise RuntimeError("No strategy worked for click")


def ensure_url_from_step(step: str) -> Optional[str]:
  m = re.search(r"https?://[^\s\"'<>]+", step)
  return m.group(0) if m else None


def build_goal(test_case: Dict[str, Any], step: str, step_index: int, total_steps: int, memory: Dict[str, str]) -> str:
  return "\n\n".join(
    [
      f"Test Case: {test_case.get('id')} - {test_case.get('title')}",
      f"User Story: {test_case.get('user_story_id', '')}",
      f"Expected: {test_case.get('expected_result', '')}",
      "All steps:\n- " + "\n- ".join(test_case.get("steps", [])),
      f"Current focus step ({step_index + 1}/{total_steps}): {step}",
      f"Memory: prev_eval={memory.get('evaluation_previous_goal','')}, memo={memory.get('memory','')}, next={memory.get('next_goal','')}",
    ]
  )


async def apply_action(page: Page, action: Dict[str, Any], elements: List[Dict[str, Any]]) -> bool:
  name = action.get("action")
  primary = action.get("primary_ref")
  secondary = action.get("secondary_ref")
  value = action.get("value")
  if name == "done":
    return False
  if primary is None and secondary is None:
    raise RuntimeError("No target ref provided")
  if name in ("click", "hover_click"):
    target = elements[primary] if primary is not None and 0 <= primary < len(elements) else None
    if not target and secondary is not None and 0 <= secondary < len(elements):
      target = elements[secondary]
    if not target:
      raise RuntimeError("Target ref out of range")
    hover = name == "hover_click"
    await click_with_fallback(page, target, hover_first=hover)
    if secondary is not None and 0 <= secondary < len(elements) and hover:
      await asyncio.sleep(0.2)
      await click_with_fallback(page, elements[secondary], hover_first=False)
    return True
  if name == "assert_url":
    current_url = page.url if isinstance(page.url, str) else page.url()
    if value and value.lower() not in current_url.lower():
      raise AssertionError(f"URL missing substring {value}")
    return False
  if name == "assert_text":
    locator = page.get_by_text(re.compile(value or "", re.IGNORECASE), exact=False)
    await locator.first.wait_for(state="visible", timeout=4000)
    return False
  raise RuntimeError(f"Unknown action {name}")


async def validate_step(page: Page, step: str, test_case: Dict[str, Any]) -> None:
  """Lightweight validator: looks for expected_result text and simple goal hints."""
  visible_text = ""
  try:
    visible_text = await page.evaluate("() => document.body.innerText || ''")
  except Exception:
    visible_text = ""
  url_lower = (page.url if isinstance(page.url, str) else page.url()).lower()
  step_lower = (step or "").lower()
  expected = (test_case.get("expected_result") or "").strip()
  if expected and len(expected) < 400:
    found = expected.lower() in visible_text.lower()
    print(f"Validation (expected_result substring): {'PASS' if found else 'WARN'}")
  if "about" in step_lower and "about" not in url_lower:
    print("Validation (about in URL): WARN - URL missing 'about'")
  if "client" in step_lower and "client" not in visible_text.lower():
    print("Validation (clients text visible): WARN - no 'client' substring in visible text")


async def handle_action_error(page: Page, error: Exception, attempt: int) -> None:
  await auto_dismiss_popups(page)
  if isinstance(error, PlaywrightTimeoutError):
    try:
      await page.reload(timeout=12000)
    except Exception:
      pass
  await page.wait_for_timeout(min(800 + attempt * 100, 2000))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_test_case(client: OpenAI, browser, test_case: Dict[str, Any], model: str, headed: bool) -> None:
  page = await browser.new_page(viewport={"width": 1440, "height": 900})
  memory = {"evaluation_previous_goal": "", "memory": "", "next_goal": ""}
  steps = test_case.get("steps", [])
  max_actions_per_step = 10
  print(f"\n=== {test_case.get('id')}: {test_case.get('title')} ===")
  try:
    for idx, step in enumerate(steps):
      url = ensure_url_from_step(step)
      if url:
        await page.goto(url, wait_until="networkidle", timeout=30000)
      last_intent_key = ""
      repeat_count = 0
      action_count = 0
      for attempt in range(12):
        await auto_dismiss_popups(page)
        state = await capture_state(page)
        page_map = build_page_map(state.get("rawHtml", ""), state.get("clickables", []), state.get("pageUrl", ""), state.get("pageTitle", ""))
        digest = digest_for_action(page_map)
        goal = build_goal(test_case, step, idx, len(steps), memory)
        meaning = await extract_meaning(client, model, goal, state)
        meaning["screenshotBase64"] = state.get("screenshotBase64","")
        intent = await choose_action(client, model, goal, digest, meaning)
        if intent.get("primary_ref") is None and meaning.get("recommended_refs"):
          intent["primary_ref"] = meaning["recommended_refs"][0]

        intent_key = json.dumps(intent, sort_keys=True)
        repeat_count = repeat_count + 1 if intent_key == last_intent_key else 0
        last_intent_key = intent_key
        if repeat_count >= 3:
          print("Repeating the same intent too often; breaking out to avoid a toggle loop.")
          break

        print("Intent:", intent)
        if intent.get("action") == "done":
          break
        try:
          should_continue = await apply_action(page, intent, state["clickables"])
          if should_continue:
            action_count += 1
          memory.update(
            {
              "evaluation_previous_goal": intent.get("notes", ""),
              "memory": f"last_action={intent.get('action')}",
              "next_goal": meaning.get("notes", ""),
            }
          )
          if action_count >= max_actions_per_step:
            print(f"Reached max actions ({max_actions_per_step}) for this step; moving on.")
            break
        except Exception as e:
          await handle_action_error(page, e, attempt)
          continue

        await asyncio.sleep(0.4)
      await validate_step(page, step, test_case)
  finally:
    await page.close()


async def main_async():
  load_env()
  ensure_tmp_dir()
  args = parse_args()
  model = args.model or os.environ.get("MODEL") or "gpt-4o"
  tests_filter = set(t.strip() for t in args.tests.split(",")) if args.tests else None
  plan = load_plan_arg(args.plan)
  test_cases = plan.get("test_cases", [])
  selected = [tc for tc in test_cases if not tests_filter or tc.get("id") in tests_filter]
  if not selected:
    raise RuntimeError("No test cases selected")
  client = openai_client()
  async with async_playwright() as p:
    browser = await p.chromium.launch(headless=not args.headed)
    for tc in selected:
      await run_test_case(client, browser, tc, model, args.headed)
    await browser.close()


def main():
  asyncio.run(main_async())


if __name__ == "__main__":
  main()
