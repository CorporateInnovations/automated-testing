import json
import re
import textwrap
from typing import Any, Dict, List


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


async def choose_action(client, model: str, goal: str, digest: List[Dict[str, Any]], meaning: Dict[str, Any]) -> Dict[str, Any]:
  system = textwrap.dedent(
    """
    You are a human-like browsing agent. Plan one immediate action based on the meaning summary.
    Always dismiss cookie/consent popups first if present, preferring a Close/X control before Accept.
    Use hover before click for dropdown nav.
    Prefer nav/subnav refs that match the goal (e.g., About/Clients) before generic links.
    Return JSON only:
    {
      "action": "click" | "hover_click" | "assert_text" | "assert_url" | "done",
      "primary_ref": <number or null>,
      "secondary_ref": <number or null>,
      "value": "<text substring for asserts or target text>",
      "notes": "<reasoning>"
    }
    """
  ).strip()

  user = f"""Goal: {goal}
Meaning summary (truncated): {clamp_text(json.dumps(meaning, indent=2), 2000)}
Digest (refs only): {clamp_text(json.dumps(digest, indent=2), 2000)}
"""

  include_image = model_supports_vision_json(model) and meaning.get("screenshotBase64")
  user_content = [{"type": "text", "text": user}]
  if include_image:
    user_content.append(
      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{meaning.get('screenshotBase64','')}" }}
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
  return parse_llm_json(content)
