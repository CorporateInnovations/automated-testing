import json
import re
import textwrap
from typing import Any, Dict, List, Optional


def clamp_text(text: str, limit: int = 5000) -> str:
  if not text:
    return ""
  return text[:limit]


def parse_llm_json(content: str) -> List[Dict[str, int]]:
  try:
    return json.loads(content)
  except Exception:
    match = re.search(r"\[[\s\S]*\]", content or "")
    if not match:
      raise
    return json.loads(match.group(0))


async def analyze_snapshot(
  client,
  model: str,
  user_request: str,
  history_steps: List[str],
  snapshot: str,
  total_refs: int,
  max_snapshot_chars: int = 8000,
) -> List[Dict[str, int]]:
  """
  Ask an LLM to select relevant ref ranges from an accessibility snapshot.

  Returns a list of {start: int, end: int} ranges.
  """
  history_block = "\n".join(f"<step> {s} </step>" for s in history_steps)
  prompt = textwrap.dedent(
    f"""
    You are a snapshot analyzer for a browser agent.
    Your task: Identify which parts of the accessibility tree snapshot are relevant to the user's current request.
    Prioritize elements that let a user satisfy the request (navigation paths, forms, buttons, inputs, dialogs).
    Deprioritize unrelated CTAs, footer/legal links, and long lists unless they directly support the request.
    Aggressively trim repetitive content while preserving ALL interactive elements.
    <conversation_history>
    <user_request> {user_request} </user_request>
    {history_block}
    </conversation_history>
    RULES:
    - KEEP all navigation, forms, buttons, modals, dialogs
    - TRIM repetitive lists to first 5 items (unless the list itself is the target of the request)
    - Keep 30-40 refs context around important elements
    SNAPSHOT ({total_refs} refs):
    {clamp_text(snapshot, max_snapshot_chars)}
    Return: [{{ "start": X, "end": Y }}, ...]
    """
  ).strip()
  resp = client.chat.completions.create(
    model=model,
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": f"{prompt}\n\nReturn JSON only."}],
  )
  content = resp.choices[0].message.content
  ranges = parse_llm_json(content)

  return ranges
