import json
import re
import textwrap
from typing import Any, Dict

from tools.agent_runner.page_model import build_page_map, human_processable
from tools.agent_runner.agents.snapshot_agent import analyze_snapshot


def model_supports_vision_json(model: str) -> bool:
  return "4o" in (model or "")


def parse_llm_json(content: str) -> Dict[str, Any]:
  try:
    return json.loads(content)
  except Exception:
    match = re.search(r"\{[\s\S]*\}", content or "")
    if not match:
      raise
    return json.loads(match.group(0))


def clamp_text(text: str, limit: int = 2000) -> str:
  if not text:
    return ""
  return text[:limit]


async def extract_meaning(client, model: str, goal: str, state: Dict[str, Any]) -> Dict[str, Any]:
  system = textwrap.dedent(
    """
    You extract a human-like understanding of the current page from a structured page map.
    Return JSON only:
    {
      "sections": "<short summary of visible sections>",
      "likely_nav": { "about": <ref or null>, "clients": <ref or null>, "menu": <ref or null> },
      "blockers": ["list issues like popups or missing nav"],
      "recommended_refs": [<ref numbers to try first>],
      "notes": "<brief reasoning>",
      "screenshotBase64": "<echoed if provided>"
    }
    """
  ).strip()

  page_map = build_page_map(
    state.get("rawHtml", ""),
    state.get("clickables", []),
    state.get("pageUrl", ""),
    state.get("pageTitle", ""),
  )

  # Optional: prune a snapshot of visible/accessibility text using the snapshot analyzer to reduce noise.
  pruned_snapshot = None
  snapshot_text = state.get("visibleText", "")
  clickables = state.get("clickables", [])
  if snapshot_text:
    try:
      ranges = await analyze_snapshot(
        client,
        model,
        goal,
        [],
        snapshot_text,
        total_refs=len(clickables) or snapshot_text.count("\n") + 1,
      )
      lines = snapshot_text.splitlines()
      kept = []
      for line in lines:
        try:
          ref_id = int(line.split(":", 1)[0].strip())
        except Exception:
          continue
        for r in ranges or []:
          if r.get("start") is not None and r.get("end") is not None and r["start"] <= ref_id <= r["end"]:
            kept.append(line)
            break
      if kept:
        pruned_snapshot = "\n".join(kept)
    except Exception:
      pruned_snapshot = None

  digest_text = clamp_text(human_processable(page_map, snapshot_text), 2000)
  snapshot_block = ""
  if pruned_snapshot:
    snapshot_block = f"\nPruned snapshot (truncated):\n{clamp_text(pruned_snapshot, 2000)}\n"

  user = f"""Goal: {goal}
Human digest (truncated):
{digest_text}
{snapshot_block}"""

  include_image = model_supports_vision_json(model) and state.get("screenshotBase64")
  user_content = [{"type": "text", "text": user}]
  if include_image:
    user_content.append(
      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{state.get('screenshotBase64','')}" }}
    )

  messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user_content},
  ]
  kwargs = {"model": model, "messages": messages}
  if model_supports_vision_json(model):
    kwargs["response_format"] = {"type": "json_object"}

  resp = client.chat.completions.create(**kwargs)
  content = resp.choices[0].message.content
  result = parse_llm_json(content)
  if include_image:
    result["screenshotBase64"] = state.get("screenshotBase64", "")
  return result
