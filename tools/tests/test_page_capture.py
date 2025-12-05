import asyncio
import unittest

from playwright.async_api import async_playwright

from tools.agent_runner.page_model import build_page_map, collect_clickable_elements
from tools.agent_runner.run import capture_state, digest_for_action


class PageCaptureTests(unittest.TestCase):
    def test_build_page_map_basic_structure(self):
        html = "<html><body><a href='/about'>About</a><button>Accept</button></body></html>"
        elements = [
            {"index": 0, "tag": "a", "text": "About", "ariaLabel": "", "title": "", "alt": "", "role": "", "selector": "a", "parentRef": None},
            {"index": 1, "tag": "button", "text": "Accept", "ariaLabel": "", "title": "", "alt": "", "role": "", "selector": "button", "parentRef": None},
        ]
        page_map = build_page_map(html, elements, url="https://example.com", title="Example")
        self.assertIn("digest", page_map)
        self.assertTrue(page_map["digest"]["nav"])
        self.assertTrue(page_map["digest"]["cookies"])

    def test_capture_state_round_trip(self):
        async def _run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.set_content("<html><body><button aria-label='Close'>X</button></body></html>")
                state = await capture_state(page)
                self.assertIn("screenshotBase64", state)
                self.assertTrue(state["clickables"])
                await browser.close()

        asyncio.run(_run())

    def test_collect_clickables_strategies(self):
        async def _run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.set_content("<html><body><button aria-label='Close'>X</button><a href='/about'>About</a></body></html>")
                clickables = await collect_clickable_elements(page)
                self.assertGreaterEqual(len(clickables), 2)
                # Ensure strategies are populated for targeting
                self.assertTrue(any(el.get("strategies") for el in clickables))
                await browser.close()

        asyncio.run(_run())

    def test_digest_for_action_prioritizes_cookies_and_nav(self):
        html = "<html><body><a href='/about'>About</a><button>Accept</button><button>Other</button></body></html>"
        elements = [
            {"index": 0, "tag": "a", "text": "About", "ariaLabel": "", "title": "", "alt": "", "role": "", "selector": "a", "parentRef": None},
            {"index": 1, "tag": "button", "text": "Accept", "ariaLabel": "", "title": "", "alt": "", "role": "", "selector": "button:nth-of-type(1)", "parentRef": None},
            {"index": 2, "tag": "button", "text": "Other", "ariaLabel": "", "title": "", "alt": "", "role": "", "selector": "button:nth-of-type(2)", "parentRef": None},
        ]
        page_map = build_page_map(html, elements, url="https://example.com", title="Example")
        digest = digest_for_action(page_map, limit=3)
        refs = [d["ref"] for d in digest]
        # Cookie and nav refs should be present in the top results
        self.assertIn(1, refs)
        self.assertIn(0, refs)

    def test_capture_state_real_site_smoke(self):
        async def _run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(viewport={"width": 1280, "height": 720})
                await page.goto("https://www.cigroup.co.uk", wait_until="networkidle", timeout=30000)
                state = await capture_state(page)
                self.assertTrue(state["clickables"])
                await browser.close()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
