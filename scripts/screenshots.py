"""The README's pictures, taken from the real app on this PC: run it again for a release.

    uv run --with pillow python scripts/screenshots.py [--gguf <a .gguf to benchmark>]

Drives the app headless (no window): Check a model (typed, checked, the pick), then a real
benchmark with quality on the Benchmark page. Saves assets/check.png, assets/benchmark.png and
assets/demo.gif (a frame about every 0.8 s, played faster, still stretches shortened).
Needs Chrome (or rsvg-convert) to turn the app's SVG screenshots into PNG, and Pillow. The
settings it uses are temporary: the app's own are not touched.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SIZE = (118, 46)  # columns, rows
CHECK_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_GGUF = "ocq-output/Qwen--Qwen3.5-0.8B/artifact/Qwen--Qwen3.5-0.8B-Q4_K_M.gguf"
FRAME_SECONDS = 0.8
WIDTH = 1500  # pixels of the PNGs


async def drive(frames: Path, shots: Path, gguf: str) -> None:
    from ocq.tui.app import QuantPilotApp

    settings = Path(tempfile.mkdtemp()) / "tui.json"
    app = QuantPilotApp(settings_path=settings, look_on_start=False)
    filming = True
    count = 0

    async def film() -> None:
        nonlocal count
        while filming:
            app.save_screenshot(filename=f"{count:04d}.svg", path=str(frames))
            count += 1
            await asyncio.sleep(FRAME_SECONDS)

    async def settle(seconds: float = 900) -> None:
        for _ in range(int(seconds / 0.5)):
            await asyncio.sleep(0.5)
            if not app.workers or all(w.is_finished for w in app.workers):
                break
        await asyncio.sleep(1.5)

    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        camera = asyncio.create_task(film())
        await asyncio.sleep(1.5)

        # Check a model: typed, checked, the pick.
        app.show_page("check", focus=True)
        await asyncio.sleep(1)
        for key in CHECK_MODEL:
            await pilot.press(key if key != "/" else "slash")
            await asyncio.sleep(0.06)
        await asyncio.sleep(0.6)
        await pilot.press("enter")
        await settle()
        app.save_screenshot(filename="check.svg", path=str(shots))
        await asyncio.sleep(2)

        # Benchmark: speed and quality, for real.
        app.open_bench(gguf, "fixed")
        await settle()
        app.save_screenshot(filename="benchmark.svg", path=str(shots))
        await asyncio.sleep(2.5)
        filming = False
        await camera


CHROME = next(
    (
        path
        for path in (
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            shutil.which("google-chrome") or "",
            shutil.which("chromium") or "",
        )
        if path and Path(path).exists()
    ),
    "",
)


def to_png(svg: Path, png: Path) -> None:
    """A browser draws the app's SVG as it is meant to look (rsvg-convert drops the spaces at
    the start of a piece of text: "Fits easily1.1 GB"); rsvg-convert when there is none."""
    if not CHROME:
        subprocess.run(["rsvg-convert", "-w", str(WIDTH), str(svg), "-o", str(png)], check=True)
        return
    import re

    from PIL import Image

    box = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg.read_text(encoding="utf-8"))
    width, height = (round(float(n)) for n in box.groups())
    page = svg.with_suffix(".html")
    page.write_text(
        f'<html><body style="margin:0;background:transparent"><img src="{svg.name}" '
        f'width="{width}" height="{height}"></body></html>'
    )
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--force-device-scale-factor=2", "--default-background-color=00000000",
         f"--window-size={width},{height}", f"--screenshot={png}", page.as_uri()],
        check=True, capture_output=True,
    )  # fmt: skip
    image = Image.open(png)
    image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.LANCZOS).save(png)


SHOWN_MS = 300  # each frame (taken every 0.8 s) on screen: played about 2.5 times as fast
STILL_MS = 1500  # the longest a frame stays when nothing changes


def make_gif(frames: Path, gif: Path) -> int:
    """The frames as a GIF, faster than it happened: a frame that looks like the one before
    only lengthens it, up to STILL_MS."""
    from PIL import Image

    images: list = []
    durations: list[int] = []
    previous = b""
    for svg in sorted(frames.glob("*.svg")):
        png = svg.with_suffix(".png")
        to_png(svg, png)
        data = png.read_bytes()
        if data == previous:
            durations[-1] = min(durations[-1] + SHOWN_MS, STILL_MS)
            continue
        previous = data
        image = Image.open(png).convert("RGB")
        image = image.resize((1100, round(image.height * 1100 / image.width)), Image.LANCZOS)
        images.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        durations.append(SHOWN_MS)
    durations[-1] = 4000  # the result stays on screen before it starts again
    images[0].save(gif, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True, disposal=1)  # fmt: skip
    return len(images)


BANNER = """<!doctype html><html><head><meta charset="utf-8"><style>
* { margin: 0; box-sizing: border-box; }
body { width: 1280px; height: 640px; overflow: hidden; font-family: -apple-system,
  "SF Pro Display", "Segoe UI", Inter, "Helvetica Neue", Arial, sans-serif; color: #e8ecf3;
  background: radial-gradient(1100px 700px at 85% 30%, #12304d 0%, rgba(18,48,77,0) 60%),
  radial-gradient(800px 600px at 0% 100%, #2a1f4a 0%, rgba(42,31,74,0) 55%), #0b0f17; }
.grid { position: absolute; inset: 0; background-image:
  radial-gradient(rgba(255,255,255,.06) 1px, transparent 1px); background-size: 22px 22px; }
.text { position: absolute; left: 72px; top: 92px; width: 560px; }
.kicker { font-size: 15px; letter-spacing: .22em; font-weight: 700; color: #38bdf8; }
h1 { margin-top: 14px; font-size: 92px; line-height: 1; font-weight: 800; letter-spacing: -.03em;
  background: linear-gradient(90deg, #ffffff 0%, #bfe6ff 55%, #a78bfa 100%);
  -webkit-background-clip: text; color: transparent; }
.lead { margin-top: 22px; font-size: 25px; line-height: 1.38; color: #c9d3e1; }
.lead b { color: #ffffff; font-weight: 650; }
.chips { margin-top: 26px; width: 640px; display: flex; flex-wrap: wrap; gap: 7px; }
.chip { font-size: 13px; padding: 5px 10px; border-radius: 999px; color: #cfe8ff;
  background: rgba(56,189,248,.10); border: 1px solid rgba(56,189,248,.30); }
.chip.run { color: #e4d9ff; background: rgba(167,139,250,.10);
  border-color: rgba(167,139,250,.32); }
.install { position: absolute; left: 72px; bottom: 64px; font: 18px "SF Mono", Menlo,
  Consolas, monospace; color: #e8ecf3; background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.12); padding: 12px 18px; border-radius: 10px; }
.install span { color: #38bdf8; }
.shot { position: absolute; right: -30px; top: 70px; width: 610px; border-radius: 12px;
  box-shadow: 0 30px 80px rgba(0,0,0,.6), 0 0 0 1px rgba(255,255,255,.08); }
</style></head><body><div class="grid"></div>
<div class="text">
  <div class="kicker">LOCAL LLMS, MEASURED AND EXPLAINED</div>
  <h1>QuantPilot</h1>
  <div class="lead">Which model, format and runtime to run <b>on your PC</b>:
    recommended with reasons, quantized, and benchmarked for speed and quality.</div>
  <div class="chips">
    <span class="chip">GGUF</span><span class="chip">AWQ</span><span class="chip">GPTQ</span>
    <span class="chip">bitsandbytes</span><span class="chip">NVFP4</span>
    <span class="chip run">llama.cpp</span><span class="chip run">vLLM</span>
    <span class="chip run">transformers</span>
  </div>
</div>
<div class="install"><span>$</span> uv tool install quantpilot</div>
<img class="shot" src="SHOT">
</body></html>"""


def make_banner(shot: Path, banner: Path) -> None:
    """assets/banner.png: what QuantPilot is, next to the real Check page (1280 x 640, also
    GitHub's social preview size), drawn by Chrome at twice the size for sharp text."""
    from PIL import Image

    page = Path(tempfile.mkdtemp()) / "banner.html"
    page.write_text(BANNER.replace("SHOT", shot.resolve().as_uri()), encoding="utf-8")
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--force-device-scale-factor=2", "--window-size=1280,640",
         "--allow-file-access-from-files", f"--screenshot={banner}", page.as_uri()],
        check=True, capture_output=True,
    )  # fmt: skip
    Image.open(banner).convert("RGB").save(banner, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", default=DEFAULT_GGUF, help="the GGUF the benchmark measures")
    parser.add_argument(
        "--banner-only", action="store_true", help="only the banner, from assets/check.png"
    )
    args = parser.parse_args()
    if args.banner_only:
        make_banner(ASSETS / "check.png", ASSETS / "banner.png")
        return 0
    if not CHROME and not shutil.which("rsvg-convert"):
        print("Chrome or rsvg-convert is needed (brew install librsvg, apt install librsvg2-bin).")
        return 1
    work = Path(tempfile.mkdtemp())
    frames, shots = work / "frames", work / "shots"
    frames.mkdir()
    shots.mkdir()
    asyncio.run(drive(frames, shots, args.gguf))
    ASSETS.mkdir(exist_ok=True)
    for name in ("check", "benchmark"):
        to_png(shots / f"{name}.svg", ASSETS / f"{name}.png")
    count = make_gif(frames, ASSETS / "demo.gif")
    if CHROME:
        make_banner(ASSETS / "check.png", ASSETS / "banner.png")
    for name in ("check.png", "benchmark.png", "demo.gif"):
        size = (ASSETS / name).stat().st_size / 1e6
        print(
            f"assets/{name}: {size:.1f} MB" + (f" ({count} frames)" if name.endswith("gif") else "")
        )
    shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
