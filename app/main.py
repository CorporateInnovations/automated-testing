import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import json
import os

from playwright.async_api import async_playwright

from app.config import Settings
from app.services.document_loader import read_text
from app.services.llm_client import TestPlanExtractor
from app.run_manager import run_manager, RunRecord
from tools.generate_playwright import render_spec

logger = logging.getLogger("app")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def get_settings() -> Settings:
    return Settings()


app = FastAPI(title="AI Test Plan Extractor", version="0.1.0")

# Resolve static directory relative to project root and ensure it exists
STATIC_DIR = (Path(__file__).resolve().parent.parent / "static").resolve()
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/extract-tests")
async def extract_tests(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    include_playwright: bool = Query(False, description="If true, return a Playwright spec alongside the plan."),
):
    text = read_text(file)
    logger.info("Received extract request: name=%s chars=%s", file.filename, len(text))
    extractor = TestPlanExtractor(settings)
    plan = extractor.extract(text)

    out_dir = Path("generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug_name = _slug(plan.document_title or 'plan') or 'plan'
    plan_path = out_dir / f"{slug_name}.json"
    plan_path.write_text(plan.model_dump_json(indent=2))
    (out_dir / "latest_plan.json").write_text(plan.model_dump_json(indent=2))

    if include_playwright:
        spec = render_spec(plan.model_dump())
        spec_path = out_dir / f"{slug_name}.spec.ts"
        spec_path.write_text(spec)
        (out_dir / "latest_plan.spec.ts").write_text(spec)
        return {
            "plan": plan,
            "playwright_spec": spec,
            "plan_path": str(plan_path),
            "spec_path": str(spec_path),
            "latest_plan_path": str(out_dir / "latest_plan.json"),
            "latest_spec_path": str(out_dir / "latest_plan.spec.ts"),
        }

    return {
        "plan": plan,
        "plan_path": str(plan_path),
        "latest_plan_path": str(out_dir / "latest_plan.json"),
    }


@app.post("/runs")
async def start_run(
    body: Dict[str, Any],
):
    """
    Start a Playwright run. Expects `spec_path` (path to .spec.ts) and optional `headed` (bool).
    """
    spec_path = body.get("spec_path", "generated/google-smoke.spec.ts")
    headed = body.get("headed", False)
    extra_args = body.get("extra_args", [])
    command = ["npx", "playwright", "test", spec_path]
    if headed:
        command.append("--headed")
    if isinstance(extra_args, list):
        command.extend(extra_args)

    record: RunRecord = await run_manager.start_run(command)
    return {"run_id": record.id, "status": record.status, "command": record.command}


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    record = await run_manager.get_run(run_id)
    if not record:
        return {"error": "run not found"}, 404
    return record.to_dict()


@app.get("/runs")
async def list_runs():
    runs = await run_manager.list_runs()
    return {"runs": runs}


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text or "").strip("-").lower() or "spec"


@app.post("/specs")
async def save_spec(body: Dict[str, Any]):
    content = body.get("content")
    name = body.get("name") or "plan"
    if not content:
        return {"error": "content required"}, 400

    out_dir = Path("generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(name)}.spec.ts"
    path = out_dir / filename
    path.write_text(content)
    return {"path": str(path)}


def _normalize_url(url: str) -> str:
    if not url:
        return "https://example.com"
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


@app.websocket("/ws/preview")
async def preview_stream(websocket: WebSocket, url: str = Query("https://example.com")):
    await websocket.accept()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        try:
            norm_url = _normalize_url(url)
            await page.goto(norm_url)
            await websocket.send_text(f"meta {page.viewport_size['width']} {page.viewport_size['height']}")

            async def send_frames():
                while True:
                    try:
                        shot = await page.screenshot(type="jpeg", quality=55, full_page=False)
                        await websocket.send_bytes(shot)
                    except Exception:
                        await websocket.send_text("error: screenshot failed")
                        break
                    await asyncio.sleep(0.3)

            async def recv_commands():
                while True:
                    try:
                        msg = await websocket.receive_text()
                    except WebSocketDisconnect:
                        break
                    if msg.startswith("click"):
                        parts = msg.split()
                        if len(parts) == 3:
                            try:
                                x = float(parts[1])
                                y = float(parts[2])
                                await page.mouse.click(x, y)
                            except Exception:
                                continue
                    elif msg.startswith("wheel"):
                        parts = msg.split()
                        if len(parts) == 3:
                            try:
                                dx = float(parts[1])
                                dy = float(parts[2])
                                await page.mouse.wheel(dx, dy)
                            except Exception:
                                continue
            await asyncio.gather(send_frames(), recv_commands())
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                msg = str(exc)
                if len(msg) > 120:
                    msg = msg[:117] + "..."
                await websocket.send_text(f"error: {msg}")
            finally:
                await websocket.close(code=1011, reason="error")
        finally:
            await browser.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
