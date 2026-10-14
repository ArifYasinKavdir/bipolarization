import asyncio, subprocess, sys, time, math
from playwright.async_api import async_playwright
HTML, OUT, WORKERS = sys.argv[1], sys.argv[2], int(sys.argv[3])
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
FPS = 30
async def worker(p, k, f0, f1):
    b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium", args=["--disable-gpu-vsync"])
    page = await (await b.new_context(viewport={"width": 1920, "height": 1080})).new_page()
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    await page.goto("file://" + HTML)
    await page.wait_for_function("window.__videoReady === true", timeout=90000)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS),
                           f"{OUT}/seg_{k:02d}.mp4"], stdin=subprocess.PIPE)
    t0 = time.time()
    for fr in range(f0, f1):
        await page.evaluate(f"renderAt({fr / FPS})")
        ff.stdin.write(await page.screenshot(type="png"))
        if (fr - f0) % 300 == 0:
            print(f"worker {k}: frame {fr} ({fr - f0}/{f1 - f0}) {time.time() - t0:.0f}s", flush=True)
    ff.stdin.close(); ff.wait()
    await b.close()
    print(f"worker {k} done {f1 - f0} frames in {time.time() - t0:.0f}s errors={errs}", flush=True)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        page = await (await b.new_context(viewport={"width": 1920, "height": 1080})).new_page()
        await page.goto("file://" + HTML)
        await page.wait_for_function("window.__videoReady === true", timeout=90000)
        frames = await page.evaluate("__video.frames")
        await b.close()
        if LIMIT: frames = min(frames, LIMIT)
        per = math.ceil(frames / WORKERS)
        ranges = [(k, k * per, min(frames, (k + 1) * per)) for k in range(WORKERS)]
        print("frames", frames, "ranges", ranges, flush=True)
        await asyncio.gather(*[worker(p, k, a, z) for k, a, z in ranges if a < z])
asyncio.run(main())
