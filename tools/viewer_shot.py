"""
Screenshot the live viewer with a headless browser.

    python tools/viewer_shot.py                                # chase view -> /tmp/viewer.png
    python tools/viewer_shot.py --view aerial --out aerial.png
    python tools/viewer_shot.py --diag                         # also print scene diagnostics

For checking what the viewer looks like on a box with no display (the same
headless GPU box that runs training and `main.py serve`), and for verifying
rendering changes without a human in front of a browser. Needs Playwright:

    pip install playwright && python -m playwright install chromium

The server must already be running (`python main.py serve`). The page is loaded
with `?debug=1` so the scene's diagnostics block exists; it is hidden again
before the screenshot unless --show-diag is given.
"""

import argparse
import sys

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--view", choices=["chase", "aerial", "driver"], default=None)
    ap.add_argument("--wait", type=float, default=6.0, help="seconds to let the page connect, load models and settle")
    ap.add_argument("--out", default="/tmp/viewer.png")
    ap.add_argument("--size", default="1600x1000", help="viewport WxH")
    ap.add_argument("--diag", action="store_true", help="print the scene diagnostics JSON")
    ap.add_argument("--show-diag", action="store_true", help="leave the diagnostics overlay visible in the shot")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    sep = "&" if "?" in args.url else "?"
    url = f"{args.url}{sep}debug=1" + (f"&view={args.view}" if args.view else "")

    problems = []
    with sync_playwright() as p:
        # Prefer an installed Chrome: its headless mode uses the real GPU, so the scene
        # renders at full speed. Playwright's bundled headless shell falls back to a
        # software rasteriser (SwiftShader), which does work -- e.g. on a Linux box with
        # no Chrome -- but a shadowed 1600x1000 scene then takes seconds per frame, so
        # pass a smaller --size and a longer --wait there.
        try:
            browser = p.chromium.launch(channel="chrome")
            engine = "chrome (gpu)"
        except Exception:
            browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            engine = "bundled chromium (software gl)"
        page = browser.new_page(viewport={"width": w, "height": h})
        page.set_default_timeout(90000)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: problems.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.goto(url)
        page.wait_for_timeout(int(args.wait * 1000))
        webgl = page.evaluate("!!document.createElement('canvas').getContext('webgl2')")
        diag = page.evaluate("(document.getElementById('debug') || {}).textContent || ''")
        if not args.show_diag:
            page.evaluate("const d = document.getElementById('debug'); if (d) d.hidden = true")
            page.wait_for_timeout(100)
        page.screenshot(path=args.out)
        browser.close()

    print(f"wrote {args.out}  ({w}x{h}, {engine}, webgl2={'yes' if webgl else 'NO'})")
    if args.diag:
        print(diag or "(no diagnostics block -- is this the viewer page?)")
    for line in problems:
        print("!", line)
    return 0 if webgl else 1


if __name__ == "__main__":
    sys.exit(main())
