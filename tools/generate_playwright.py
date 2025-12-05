import json
import re
from pathlib import Path
from typing import Any, List


TEMPLATE = """import {{ test, expect }} from '@playwright/test';

test.describe('{title}', () => {{
{tests}
}});
"""

TEST_SNIPPET = """  test(`{id} | {title}`, async ({{ page }}) => {{
{step_lines}
    // TODO: add actions/assertions to verify: {expected_result}
    // meta: priority={priority} type={type} tags={tags} feature={feature} trace={trace}
  }});"""


def _escape(text: str) -> str:
    return (text or "").replace("'", "\\'")


def slug(text: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", text or "").strip("-").lower()
    return sanitized or "test-plan"


def render_steps(steps: Any) -> str:
    if isinstance(steps, str):
        steps = [steps]
    if not isinstance(steps, list):
        steps = []
    lines: List[str] = []
    for idx, step in enumerate(steps or [], 1):
        lines.append(f"    // Step {idx}: {step}")
    if not lines:
        lines.append("    // TODO: add steps")
    return "\n".join(lines)


def build_tests(plan: dict) -> str:
    tests = []
    for tc in plan.get("test_cases", []):
        tests.append(
            TEST_SNIPPET.format(
                id=_escape(tc.get("id", "TC-XXX")),
                title=_escape(tc.get("title", "")),
                step_lines=render_steps(tc.get("steps")),
                expected_result=_escape(tc.get("expected_result", "")),
                priority=_escape(tc.get("priority", "")),
                type=_escape(tc.get("type", "")),
                tags=_escape(",".join(tc.get("tags", [])) if isinstance(tc.get("tags"), list) else tc.get("tags", "")),
                feature=_escape(tc.get("feature", "")),
                trace=_escape(tc.get("trace", "")),
                page="page",
            )
        )
    return "\n\n".join(tests)


def render_spec(plan: dict) -> str:
    title = plan.get("document_title") or plan.get("summary") or "Test Plan"
    tests = build_tests(plan)
    return TEMPLATE.format(title=title, tests=tests)


def generate(in_path: Path, out_dir: Path) -> Path:
    plan = json.loads(in_path.read_text())
    content = render_spec(plan)
    title = plan.get("document_title") or plan.get("summary") or "Test Plan"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{slug(title)}.spec.ts"
    out_file.write_text(content)
    return out_file


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate Playwright test scaffold from JSON test plan.")
    parser.add_argument("plan_json", help="Path to test plan JSON file.")
    parser.add_argument("--out-dir", default="generated", help="Output directory for .spec.ts (default: generated)")
    args = parser.parse_args()

    out_file = generate(Path(args.plan_json), Path(args.out_dir))
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
