from __future__ import annotations

import json
import asyncio
import argparse
from typing import Any, Dict, List, Tuple

from playwright.async_api import async_playwright


def build_page_map(html: str, elements: List[Dict[str, Any]], url: str, title: str) -> Dict[str, Any]:
  raw_html = html or ""
  nav = []
  ctas = []
  cookies = []
  inputs = []
  links = []
  images = []
  subnav = []
  refs = []
  menu_tree = []

  def bucket(el: Dict[str, Any]) -> str:
    text = (el.get("text") or "").lower()
    aria = (el.get("ariaLabel") or "").lower()
    title_val = (el.get("title") or "").lower()
    alt = (el.get("alt") or "").lower()
    combined = f"{text} {aria} {title_val} {alt}"
    if any(k in combined for k in ["about", "client", "work", "contact", "menu", "navigation", "services"]):
      return "nav"
    if any(k in combined for k in ["sign up", "sign in", "register", "join", "start", "continue", "next", "submit"]):
      return "cta"
    if any(k in combined for k in ["accept", "allow", "agree", "consent", "cookie", "got it"]):
      return "cookie"
    if el.get("tag") in ["input", "textarea", "select"]:
      return "input"
    if el.get("tag") == "a":
      return "link"
    if el.get("tag") == "img":
      return "image"
    return "other"

  def element_type(kind: str, el: Dict[str, Any]) -> str:
    if el.get("parentRef") is not None and kind in ("nav", "link"):
      return "nav-sub"
    if kind == "nav":
      return "nav-link"
    if kind == "cta":
      return "cta-button"
    if kind == "cookie":
      return "cookie-button"
    if kind == "input":
      return "input"
    if kind == "link":
      return "link"
    if kind == "image":
      return "image"
    if el.get("role") == "button":
      return "button"
    return "other"

  for el in elements:
    kind = bucket(el)
    bbox = el.get("boundingBox", {}) or {}
    record = {
      "ref": el.get("index"),
      "tag": el.get("tag"),
      "text": el.get("text", "")[:120],
      "aria": el.get("ariaLabel", "")[:120],
      "title": el.get("title", "")[:120],
      "alt": el.get("alt", "")[:120],
      "role": el.get("role", ""),
      "type": element_type(kind, el),
      "position": {"x": bbox.get("x"), "y": bbox.get("y")},
      "bbox": bbox,
      "selector": el.get("selector", ""),
      "parentRef": el.get("parentRef"),
    }
    refs.append(record)
    if kind == "nav":
      nav.append(record)
    elif kind == "cta":
      ctas.append(record)
    elif kind == "cookie":
      cookies.append(record)
    elif kind == "input":
      inputs.append(record)
    elif kind == "link":
      links.append(record)
    elif kind == "image":
      images.append(record)
    if record["type"] == "nav-sub":
      subnav.append(record)

  digest = {
    "nav": nav[:20],
    "ctas": ctas[:20],
    "cookies": cookies[:10],
    "inputs": inputs[:20],
    "links": links[:20],
    "images": images[:20],
    "subnav": subnav[:30],
  }
  # Build a simple menu tree (nav -> subnav children)
  nav_by_ref = {el["ref"]: el for el in nav}
  children_by_parent = {}
  for child in subnav:
    parent = child.get("parentRef")
    if parent is None:
      continue
    children_by_parent.setdefault(parent, []).append(child)
  for parent_ref, kids in children_by_parent.items():
    parent = nav_by_ref.get(parent_ref)
    if not parent:
      continue
    menu_tree.append(
      {
        "parent_ref": parent_ref,
        "parent_text": parent.get("text", ""),
        "children": [{"ref": k.get("ref"), "text": k.get("text", ""), "selector": k.get("selector", "")} for k in kids],
      }
    )

  return {
    "url": url,
    "title": title,
    "raw_html": raw_html[:8000],
    "digest": digest,
    "menu_tree": menu_tree,
    "refs": refs,
  }


def serialize_page_map(page_map: Dict[str, Any]) -> str:
  return json.dumps(page_map, indent=2)


def human_processable(page_map: Dict[str, Any], visible_text: str = "") -> str:
  digest = page_map.get("digest", {})
  def fmt_list(items, label):
    if not items:
      return ""
    lines = [label]
    for el in items[:15]:
      name = el.get("text") or el.get("aria") or el.get("title") or "untitled"
      parent = el.get("parentRef")
      parent_txt = f", parent {parent}" if parent is not None else ""
      lines.append(f"  - {name.strip()} (ref {el.get('ref')}, type {el.get('type')}{parent_txt})")
    return "\n".join(lines)

  nav_txt = fmt_list(digest.get("nav"), "Top Navigation:")
  subnav_txt = fmt_list(digest.get("subnav"), "Sub Navigation (dropdowns):")
  cta_txt = fmt_list(digest.get("ctas"), "Primary CTAs:")
  cookie_txt = fmt_list(digest.get("cookies"), "Cookie/Consent:")
  link_txt = fmt_list(digest.get("links"), "Links:")
  image_alts = [el for el in digest.get("images", []) if el.get("alt")]
  logos_txt = fmt_list(image_alts, "Visible Logos/Images:")
  # Menu tree grouping (nav -> children)
  menu_tree = page_map.get("menu_tree") or []
  tree_lines = []
  for node in menu_tree[:12]:
    parent_text = (node.get("parent_text") or "").strip() or f"ref {node.get('parent_ref')}"
    children = node.get("children") or []
    child_parts = [f"{c.get('text') or 'untitled'} (ref {c.get('ref')})" for c in children[:8]]
    tree_lines.append(f"- {parent_text}: " + ", ".join(child_parts))
  tree_txt = "Menu Tree:\n" + "\n".join(tree_lines) if tree_lines else ""

  visible_lines = [ln.strip() for ln in (visible_text or "").splitlines() if ln.strip()]
  main_text = "\n".join(visible_lines[:8])

  sections = [nav_txt, subnav_txt, tree_txt, cta_txt, cookie_txt, logos_txt, link_txt, "Main Visible Text:\n" + main_text]
  return "\n\n".join([s for s in sections if s])


async def collect_clickable_elements(page):
  script = """
    () => {
      function isVisible(el) {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return (
          el.offsetParent !== null &&
          rect.width > 0 &&
          rect.height > 0 &&
          style.visibility !== "hidden" &&
          style.display !== "none"
        );
      }
      function walk(root) {
        const results = [];
        const nodes = root.querySelectorAll("a, button, [role='button'], input, [tabindex], img");
        nodes.forEach((el) => {
          if (!isVisible(el)) return;
          const rect = el.getBoundingClientRect();
          const text = ((el.innerText || el.textContent || "").trim() || (el.value || "").toString()).slice(0, 200);
          const aria = (el.getAttribute("aria-label") || "").trim();
          const title = (el.getAttribute("title") || "").trim();
          const role = (el.getAttribute("role") || "").trim();
          const alt = (el.getAttribute("alt") || "").trim();
          const dataTestId = el.getAttribute("data-testid") || el.getAttribute("data-test") || "";
          const toPath = (node) => {
            const parts = [];
            let cur = node;
            while (cur && cur.tagName && cur.tagName.toLowerCase() !== "html") {
              const parent = cur.parentElement;
              if (!parent) break;
              const siblings = Array.from(parent.children).filter((c) => c.tagName === cur.tagName);
              const idx = siblings.indexOf(cur) + 1;
              parts.unshift(`${cur.tagName.toLowerCase()}:nth-of-type(${idx})`);
              cur = parent;
            }
            return parts.length ? parts.join(" > ") : null;
          };
          const cssPath = toPath(el);
          const strategies = [];
          if (dataTestId) strategies.push({ type: "css", value: `[data-testid="${dataTestId}"]` });
          if (aria) strategies.push({ type: "aria", value: aria });
          if (role && text) strategies.push({ type: "role", value: { role, name: text } });
          if (text) strategies.push({ type: "text", value: text.slice(0, 100) });
          if (cssPath) strategies.push({ type: "css", value: cssPath });
          results.push({
            index: results.length,
            tag: el.tagName.toLowerCase(),
            text,
            ariaLabel: aria,
            title,
            alt,
            role,
            dataTestId,
            strategies,
            selector: cssPath,
            boundingBox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
            visible: true,
            inShadowRoot: !!el.getRootNode().host,
            rootNodeName: el.getRootNode().host ? el.getRootNode().host.tagName.toLowerCase() : "document",
          });
        });
        return results;
      }
      const all = [];
      all.push(...walk(document));
      const shadowHosts = Array.from(document.querySelectorAll("*")).filter((el) => el.shadowRoot);
      for (const host of shadowHosts) {
        all.push(...walk(host.shadowRoot));
      }
      return all.slice(0, 150);
    }
  """
  return await page.evaluate(script)


async def expand_nav_and_collect(page, base_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  elements = list(base_elements)
  existing: set[Tuple[str, str]] = set()
  for el in elements:
    existing.add((el.get("selector") or "", el.get("text") or ""))
  # Prefer top nav anchors/buttons in the header region as hover candidates.
  nav_candidates: List[Dict[str, Any]] = []
  for el in elements:
    y = el.get("boundingBox", {}).get("y")
    if el.get("tag") in ("a", "button") and y is not None and y < 400:
      nav_candidates.append(el)
  # Limit to first 16 candidates to avoid over-hovering.
  nav_candidates = nav_candidates[:16]
  for cand in nav_candidates:
    sel = cand.get("selector")
    if not sel:
      continue
    loc = page.locator(sel)
    try:
      await loc.hover(timeout=2000)
      await page.wait_for_timeout(350)
    except Exception:
      pass
    try:
      await loc.click(timeout=1500)
      await page.wait_for_timeout(350)
    except Exception:
      pass
    try:
      extra = await collect_clickable_elements(page)
    except Exception:
      continue
    for e in extra:
      key = (e.get("selector") or "", e.get("text") or "")
      if key in existing:
        continue
      e["parentRef"] = cand.get("index")
      e["index"] = len(elements)
      elements.append(e)
      existing.add(key)
  return elements


async def fetch_and_build(url: str, headed: bool = False) -> Dict[str, Any]:
  async with async_playwright() as p:
    browser = await p.chromium.launch(headless=not headed)
    page = await browser.new_page(viewport={"width": 1440, "height": 900})
    await page.goto("about:blank")
    await page.wait_for_timeout(200)
    await page.goto(url, wait_until="networkidle", timeout=30000)
    html = await page.content()
    base_elements = await collect_clickable_elements(page)
    elements = await expand_nav_and_collect(page, base_elements)
    try:
      title = await page.title()
    except Exception:
      title = ""
    await browser.close()
    return build_page_map(html, elements, url=url, title=title)


if __name__ == "__main__":
  page_map = asyncio.run(fetch_and_build("https://www.cigroup.co.uk", headed=False))
  print("Human Processable: " + human_processable(page_map))
