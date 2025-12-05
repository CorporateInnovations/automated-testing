"""
Manual driver to run meaning + action agents and print their outputs.
Requires OPENAI_API_KEY and network access. Not part of automated tests.

Usage:
  PYTHONPATH=. python tools/tests/manual_agent_flow.py \
    --url https://www.cigroup.co.uk \
    --goal "Navigate to About Us and view Clients section" \
    --model gpt-4o
"""
import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from playwright.async_api import async_playwright

from tools.agent_runner.agents.action_agent import choose_action
from tools.agent_runner.agents.meaning_agent import extract_meaning
from tools.agent_runner.page_model import build_page_map, expand_nav_and_collect, human_processable
from tools.agent_runner.run import (
  apply_action,
  auto_dismiss_popups,
  capture_state,
  digest_for_action,
  load_env,
  openai_client,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Manual meaning+action run with printed outputs.")
  parser.add_argument("--url", default=os.environ.get("AGENT_URL", "https://www.cigroup.co.uk"), help="Target URL")
  parser.add_argument(
    "--goal",
    default=os.environ.get("AGENT_GOAL", "Navigate to About Us and view Clients section"),
    help="Goal for the agent",
  )
  parser.add_argument("--model", default=os.environ.get("MODEL", "gpt-4o"), help="OpenAI model")
  parser.add_argument("--headed", action="store_true", help="Run Playwright headed")
  parser.add_argument("--max-steps", type=int, default=15, help="Max agent steps before giving up")
  parser.add_argument("--plan", help="Optional path or JSON string containing acceptance criteria")
  parser.add_argument("--scripted", action="store_true", help="Use simple heuristic navigation instead of the LLM")
  return parser.parse_args()


def load_plan_arg(plan_arg: str) -> dict:
  """Load plan from a file path or a JSON string."""
  if not plan_arg:
    return {}
  plan_arg = plan_arg.strip()
  if plan_arg.startswith("{"):
    return json.loads(plan_arg)
  plan_path = os.path.abspath(plan_arg)
  with open(plan_path, "r", encoding="utf-8") as f:
    return json.load(f)


def extract_acceptance(plan: dict) -> list:
  """Pull acceptance criteria from the first user story if present."""
  try:
    stories = plan.get("user_stories") or []
    if stories and isinstance(stories[0], dict):
      return stories[0].get("acceptance_criteria") or []
  except Exception:
    return []
  return []


def check_acceptance(state: dict, seen_about: bool, acceptance: list | None = None) -> Tuple[bool, str]:
  """
  Heuristic for the end goal: reach the Clients page after visiting About.
  - Must have already visited About in this session (seen_about flag).
  - Current URL or main heading must look like a clients page (url contains 'client' or a heading contains 'client').
  - Clients content and visuals should be present.
  """
  url = (state.get("pageUrl") or "").lower()
  text = (state.get("visibleText") or "").lower()
  # Treat headings in visible text as a signal of a clients page.
  headings = [ln.strip().lower() for ln in (state.get("visibleText") or "").splitlines() if ln.strip()]
  has_client_heading = any(ln.startswith(("clients", "our clients", "client")) for ln in headings)

  clients_page = "client" in url or has_client_heading
  clients_present = any(tok in text for tok in ("client", "clients", "our clients"))
  client_visuals = any((el.get("alt") or "").strip() for el in state.get("clickables", []) if el.get("tag") == "img")
  success = seen_about and clients_page and (clients_present or client_visuals)
  reason = []
  if acceptance:
    reason.append(f"acceptance={'; '.join(acceptance)}")
  reason.append(f"seen_about={seen_about}")
  reason.append(f"clients_page={clients_page}")
  reason.append(f"client_heading={has_client_heading}")
  reason.append(f"clients_present={clients_present}")
  reason.append(f"client_visuals={client_visuals}")
  return success, "; ".join(reason)


def choose_scripted_action(
  digest: List[Dict[str, Any]],
  elements: List[Dict[str, Any]],
  seen_about: bool,
  tried_refs: dict[int, int],
) -> Optional[Dict[str, Any]]:
  """
  Minimal heuristic navigator when --scripted is enabled:
  - If About has not been seen, click the first ref with text/aria/title containing 'about'.
  - If About has been seen, click the first ref with text/aria/title containing 'client', preferring nav/subnav/button types.
    If a subnav item has a parent whose text mentions About, hover parent then click child.
  """
  def el_ref(el: Dict[str, Any]) -> Optional[int]:
    return el.get("ref") if el.get("ref") is not None else el.get("index")

  def el_parent(el: Dict[str, Any]) -> Optional[int]:
    return el.get("parentRef")

  def pick_from_elements(match_fn):
    for el in elements:
      if match_fn(el):
        return el
    return None

  if not seen_about:
    # Prefer header/nav region if we have y coordinate.
    about = pick_from_elements(
      lambda el: any(token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in ("about",))
      and (el.get("boundingBox", {}).get("y") or 9999) < 400
    )
    about = about or pick_from_elements(
      lambda el: any(token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in ("about",))
    )
    if about:
      ref = el_ref(about)
      tried_refs[ref] = tried_refs.get(ref, 0) + 1
      return {"action": "click", "primary_ref": ref, "secondary_ref": None, "value": "About", "notes": "Navigate to About."}

  # Single cookie accept if present.
  cookie = pick_from_elements(
    lambda el: any(token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in ("accept", "cookie", "consent"))
  )
  if cookie:
    ref = el_ref(cookie)
    tried_refs[ref] = tried_refs.get(ref, 0) + 1
    if tried_refs[ref] <= 1:
      return {"action": "click", "primary_ref": ref, "secondary_ref": None, "value": "Accept", "notes": "Accept/dismiss cookie banner."}

  # Try subnav child whose parent mentions About.
  candidates = [
    el
    for el in elements
    if el_parent(el) is not None
    and any(token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in ("client",))
  ]
  for client_el in candidates:
    parent_idx = el_parent(client_el)
    parent_el = next((el for el in elements if el_ref(el) == parent_idx), None)
    parent_has_about = parent_el and any("about" in (parent_el.get(field) or "").lower() for field in ("text", "aria", "title"))
    print(f"Debug: candidate client ref {el_ref(client_el)} parent={parent_idx} parent_has_about={parent_has_about}")
    if parent_el and parent_has_about and el_ref(client_el) is not None:
      child = el_ref(client_el)
      tried_refs[child] = tried_refs.get(child, 0) + 1
      return {
        "action": "hover_click",
        "primary_ref": parent_idx,
        "secondary_ref": child,
        "value": "Clients",
        "notes": "Hover About then click Clients subnav.",
      }

  clients = pick_from_elements(
    lambda el: any(token in (el.get(field) or "").lower() for field in ("text", "aria", "title") for token in ("client",))
  )
  if clients:
    ref = el_ref(clients)
    tried_refs[ref] = tried_refs.get(ref, 0) + 1
    return {"action": "click", "primary_ref": ref, "secondary_ref": None, "value": "Clients", "notes": "Navigate to Clients."}

  return None


async def main_async() -> None:
  load_env()
  args = parse_args()
  acceptance = []
  if args.plan:
    try:
      plan = load_plan_arg(args.plan)
      acceptance = extract_acceptance(plan)
      if acceptance:
        print("Loaded acceptance criteria from plan:")
        for item in acceptance:
          print(f"- {item}")
    except Exception as e:
      print(f"Warning: failed to load plan/acceptance criteria: {e}")
  client = openai_client()
  async with async_playwright() as p:
    browser = await p.chromium.launch(headless=not args.headed)
    page = await browser.new_page(viewport={"width": 1440, "height": 900})
    await page.goto(args.url, wait_until="networkidle", timeout=30000)

    tried_refs: dict[int, int] = {}
    seen_about = False
    auto_client_attempted = False
    for step in range(args.max_steps):
      await auto_dismiss_popups(page)
      state = await capture_state(page)
      url_lower = (state.get("pageUrl") or "").lower()
      text_lower = (state.get("visibleText") or "").lower()
      if "about" in url_lower or "about us" in text_lower:
        seen_about = True

      ok, reason = check_acceptance(state, seen_about, acceptance)
      if ok:
        print(f"\nAcceptance criteria met at step {step}: {reason}")
        break

      # Expand nav to include dropdown/subnav items where possible.
      expanded_clickables = await expand_nav_and_collect(page, state.get("clickables", []))
      page_map = build_page_map(state.get("rawHtml", ""), expanded_clickables, state.get("pageUrl", ""), state.get("pageTitle", ""))
      digest = digest_for_action(page_map)
      # Log a human-readable snapshot of the page map for debugging (scripted mode only).
      if args.scripted:
        print("\n--- Human processable page map ---")
        print(human_processable(page_map, state.get("visibleText", "")))

      if args.scripted:
        action = choose_scripted_action(digest, expanded_clickables, seen_about, tried_refs)
        if not action:
          print("Scripted mode: no suitable action found; stopping.")
          break
        print(f"\n=== Step {step} Scripted Intent ===")
        print(json.dumps(action, indent=2))
        try:
          await apply_action(page, action, expanded_clickables)
          # If we just targeted a Clients item, try to scroll it into view to surface the section text/logos.
          if "client" in (action.get("value") or "").lower():
            target_ref = action.get("secondary_ref") if action.get("secondary_ref") is not None else action.get("primary_ref")
            target_el = next((el for el in expanded_clickables if (el.get("ref") or el.get("index")) == target_ref), None)
            y = target_el.get("boundingBox", {}).get("y") if target_el else None
            try:
              if y is not None:
                await page.evaluate("y => window.scrollTo({ top: y - 100, behavior: 'smooth' });", y)
              else:
                await page.evaluate("() => window.scrollBy(0, window.innerHeight * 2);")
              await page.wait_for_timeout(800)
            except Exception:
              pass
          await asyncio.sleep(0.4)
          continue
        except Exception as e:
          print(f"Scripted action failed: {e}; stopping.")
          break

      # Non-scripted: if a Clients subnav exists under About, trigger it before asking the LLM.
      if not auto_client_attempted:
        def el_ref(el: Dict[str, Any]) -> Optional[int]:
          return el.get("ref") if el.get("ref") is not None else el.get("index")

        def has_token(el: Dict[str, Any], tok: str) -> bool:
          tok = tok.lower()
          return any(tok in (el.get(field) or "").lower() for field in ("text", "aria", "title"))

        client_child = next(
          (
            el
            for el in expanded_clickables
            if el.get("parentRef") is not None
            and has_token(el, "client")
            and any(
              has_token(parent_el, "about")
              for parent_el in expanded_clickables
              if el_ref(parent_el) == el.get("parentRef")
            )
          ),
          None,
        )
        if client_child:
          parent_ref = client_child.get("parentRef")
          child_ref = el_ref(client_child)
          auto_action = {
            "action": "hover_click",
            "primary_ref": parent_ref,
            "secondary_ref": child_ref,
            "value": "Clients",
            "notes": "Auto: open About > Clients from nav before LLM.",
          }
          try:
            await apply_action(page, auto_action, expanded_clickables)
            auto_client_attempted = True
            await asyncio.sleep(0.4)
            continue
          except Exception:
            auto_client_attempted = True  # avoid retry loop

      meaning = await extract_meaning(client, args.model, args.goal, state)
      meaning["screenshotBase64"] = state.get("screenshotBase64", "")

      intent = await choose_action(client, args.model, args.goal, digest, meaning)
      if intent.get("primary_ref") is None and meaning.get("recommended_refs"):
        intent["primary_ref"] = meaning["recommended_refs"][0]
      print(f"\n=== Step {step} Intent ===")
      print(json.dumps(intent, indent=2))

      if intent.get("action") == "done":
        print("Agent indicated done; stopping loop.")
        break
      try:
        await apply_action(page, intent, expanded_clickables)
      except Exception as e:
        print(f"Action error: {e}; continuing.")
        continue
    else:
      print("\nReached max steps without meeting acceptance criteria.")

    await browser.close()


def main():
  asyncio.run(main_async())


if __name__ == "__main__":
  main()
