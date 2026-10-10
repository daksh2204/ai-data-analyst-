"""Record the README screenshots and the 10-30 s demo video with a real browser.

Start the app first (with your LLM key in .env), then run:

    pip install playwright && python -m playwright install chromium
    python scripts/record_demo.py                      # app on http://localhost:8000
    python scripts/record_demo.py --url http://localhost:5173 --skip-video

Outputs
    docs/screenshots/*.png     one screenshot per feature
    docs/demo.mp4              the demo video, sped up to fit --max-seconds (needs ffmpeg)
    docs/demo.gif              a small GIF for the README (needs ffmpeg)
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from playwright.async_api import Page, async_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
SCRATCH = ROOT / "scripts" / ".demo"
VIEWPORT = {"width": 1440, "height": 900}

SCREENSHOT_FLOW = [
    ("02-answer-and-chart", "Which region generated the highest revenue?"),
    ("03-monthly-trend", "Show monthly sales trends."),
    ("04-underperforming", "Which products are underperforming?"),
    ("05-top-customers-sql", "What are the top five customers? Show the SQL and pandas code."),
    ("06-anomalies", "Detect anomalies in the dataset."),
    ("07-forecast", "Forecast revenue for the next 6 months."),
    ("08-targets-multi-file", "Which regions missed their revenue targets?"),
]
VIDEO_FLOW = [
    "Which region generated the highest revenue?",
    "Show monthly sales trends.",
    "Detect anomalies in the dataset.",
]


async def wait_idle(page: Page, timeout_s: float) -> None:
    """Wait until the live answer has finished streaming."""
    await page.wait_for_timeout(400)
    await page.wait_for_selector("article.turn.is-live", state="detached", timeout=timeout_s * 1000)
    await page.wait_for_timeout(900)  # let charts finish animating


async def ask(page: Page, question: str, timeout_s: float, *, type_delay: int = 0) -> None:
    box = page.locator("textarea#question")
    await box.click()
    if type_delay:
        await box.press_sequentially(question, delay=type_delay)
    else:
        await box.fill(question)
    await box.press("Enter")
    await wait_idle(page, timeout_s)


async def scroll_last_turn_to_top(page: Page) -> None:
    await page.evaluate(
        "() => { const t = [...document.querySelectorAll('article.turn')].pop();"
        " if (t) t.scrollIntoView({block: 'start'}); }"
    )
    await page.wait_for_timeout(300)


async def load_samples(page: Page) -> None:
    await page.get_by_role("button", name="Load sample data").first.click()
    await page.wait_for_selector("text=sales_orders.csv", timeout=30_000)
    await page.wait_for_timeout(800)
    # Dismiss the "loaded" toast so it doesn't cover the screenshots.
    for btn in await page.locator(".toast button").all():
        await btn.click()
    await page.wait_for_timeout(400)


async def screenshots(url: str, timeout_s: float, theme: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport=VIEWPORT, color_scheme=theme, device_scale_factor=2)
        await page.goto(url)
        await page.wait_for_timeout(1000)
        await load_samples(page)
        await page.screenshot(path=SHOTS / "01-overview.png")
        for name, question in SCREENSHOT_FLOW:
            await ask(page, question, timeout_s)
            await scroll_last_turn_to_top(page)
            if name == "05-top-customers-sql":
                working = page.locator("article.turn").last.get_by_role("button", name="Show working")
                if await working.count():
                    await working.first.click()
                    await page.wait_for_timeout(400)
                    code_tab = page.locator("article.turn").last.locator("[role=tab], button").filter(has_text="Code")
                    if await code_tab.count():
                        await code_tab.first.click()
                        await page.wait_for_timeout(400)
                        await code_tab.first.evaluate("el => el.scrollIntoView({block: 'start'})")
                        await page.mouse.wheel(0, -120)
                        await page.wait_for_timeout(500)
            await page.screenshot(path=SHOTS / f"{name}.png")
            print(f"  saved {name}.png")
        await page.get_by_role("button", name="Dashboard").click()
        await page.wait_for_timeout(2500)
        await page.screenshot(path=SHOTS / "09-dashboard.png")
        await page.get_by_role("button", name="Data").click()
        await page.wait_for_timeout(1500)
        await page.screenshot(path=SHOTS / "10-data-quality.png")
        await browser.close()
    print(f"Screenshots in {SHOTS}")


async def video(url: str, timeout_s: float, theme: str, max_seconds: float) -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(viewport=VIEWPORT, color_scheme=theme,
                                            record_video_dir=str(SCRATCH), record_video_size=VIEWPORT)
        page = await context.new_page()
        await page.goto(url)
        await page.wait_for_timeout(1200)
        await load_samples(page)
        for q in VIDEO_FLOW:
            await ask(page, q, timeout_s, type_delay=25)
            await scroll_last_turn_to_top(page)
            await page.wait_for_timeout(1500)
        await page.get_by_role("button", name="Dashboard").click()
        await page.wait_for_timeout(2500)
        await page.mouse.wheel(0, 500)
        await page.wait_for_timeout(1500)
        await context.close()
        await browser.close()
    raw = next(SCRATCH.glob("*.webm"))
    if not shutil.which("ffmpeg"):
        target = ROOT / "docs" / "demo.webm"
        shutil.copy(raw, target)
        print(f"ffmpeg not found; saved the raw recording to {target}")
        return
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(raw)]).strip() or 0)
    speed = max(1.0, duration / max_seconds)
    mp4 = ROOT / "docs" / "demo.mp4"
    gif = ROOT / "docs" / "demo.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-filter:v", f"setpts=PTS/{speed:.3f},fps=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    "fps=8,scale=880:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)
    print(f"Video: {mp4} ({duration:.1f}s raw, {speed:.2f}x speed-up)\nGIF:   {gif}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=120, help="Seconds to wait for each answer")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--max-seconds", type=float, default=28, help="Target video length")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-screenshots", action="store_true")
    args = parser.parse_args()
    if not args.skip_screenshots:
        asyncio.run(screenshots(args.url, args.timeout, args.theme))
    if not args.skip_video:
        asyncio.run(video(args.url, args.timeout, args.theme, args.max_seconds))


if __name__ == "__main__":
    sys.exit(main())
