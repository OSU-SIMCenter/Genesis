#!/usr/bin/env python3
"""Probe OpenGL backends available in this environment (WSL/Linux diagnostics)."""

from __future__ import annotations

import argparse
import importlib
import os
import sys


def _is_software_renderer(renderer: str) -> bool:
    r = renderer.lower()
    return any(tag in r for tag in ("llvmpipe", "softpipe", "swiftshader", "software"))


def probe_platform(platform: str, *, verbose: bool = False) -> dict:
    result = {
        "platform": platform,
        "ok": False,
        "renderer": "",
        "vendor": "",
        "version": "",
        "software": False,
        "error": "",
    }

    prev = os.environ.get("PYOPENGL_PLATFORM")
    os.environ["PYOPENGL_PLATFORM"] = platform
    try:
        importlib.reload(__import__("OpenGL.platform"))
        import pyglet
        from OpenGL.GL import GL_VENDOR, GL_RENDERER, GL_VERSION, glGetString

        confs = [
            pyglet.gl.Config(double_buffer=True, depth_size=24),
            pyglet.gl.Config(double_buffer=True),
            None,
        ]
        window = None
        last_exc: Exception | None = None
        for conf in confs:
            try:
                window = pyglet.window.Window(
                    width=64,
                    height=64,
                    visible=False,
                    config=conf,
                )
                break
            except Exception as exc:
                last_exc = exc
                continue
        if window is None:
            raise last_exc or RuntimeError("Could not create pyglet window")
        try:
            window.switch_to()
            result["vendor"] = (glGetString(GL_VENDOR) or b"").decode(errors="replace")
            result["renderer"] = (glGetString(GL_RENDERER) or b"").decode(errors="replace")
            result["version"] = (glGetString(GL_VERSION) or b"").decode(errors="replace")
            result["software"] = _is_software_renderer(result["renderer"])
            result["ok"] = bool(result["renderer"])
        finally:
            if window is not None:
                window.close()
            try:
                pyglet.app.platform_event_loop.stop()
            except Exception:
                pass
    except Exception as exc:
        result["error"] = str(exc)
        if verbose:
            import traceback

            traceback.print_exc()
    finally:
        if prev is None:
            os.environ.pop("PYOPENGL_PLATFORM", None)
        else:
            os.environ["PYOPENGL_PLATFORM"] = prev
        importlib.reload(__import__("OpenGL.platform"))

    return result


def default_platforms() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("wgl",)
    if sys.platform == "linux":
        if os.environ.get("WSL_DISTRO_NAME"):
            return ("glx", "native", "egl", "osmesa")
        return ("native", "egl", "glx", "osmesa")
    return ("native",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        help="OpenGL platform to test (repeatable). Default: environment-appropriate set.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=== Environment ===")
    print(f"DISPLAY={os.environ.get('DISPLAY', '')!r}")
    print(f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')!r}")
    print(f"WSL_DISTRO_NAME={os.environ.get('WSL_DISTRO_NAME', '')!r}")
    print(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', 'unset')!r}")
    print()

    platforms = tuple(args.platforms) if args.platforms else default_platforms()
    rows = [probe_platform(platform, verbose=args.verbose) for platform in platforms]
    best = None
    print("=== Platform probe ===")
    for row in rows:
        status = "OK" if row["ok"] else "FAIL"
        hw = "software" if row["software"] else "hardware?"
        print(f"[{status}] {row['platform']:8s}  renderer={row['renderer']!r}  ({hw})")
        if row["error"]:
            print(f"         error: {row['error']}")
        if row["ok"] and not row["software"] and best is None:
            best = row

    print()
    if best:
        print(
            f"Recommended: PYOPENGL_PLATFORM={best['platform']!r} "
            f"({best['renderer']})"
        )
        print(f"  cfg.performance.opengl_platform = {best['platform']!r}")
        return 0

    if any(row["ok"] for row in rows):
        print(
            "Only software OpenGL backends succeeded. Viewer will be slow.\n"
            "  WSL2: update Windows GPU drivers; in WSL run:\n"
            "    sudo apt install mesa-utils mesa-vulkan-drivers libgl1-mesa-dri\n"
            "  try: pixi run python agforge/teleop_socket.py --opengl glx\n"
            "  or:  pixi run python agforge/teleop_socket.py --headless"
        )
        return 1

    print("No OpenGL backend succeeded. Install Mesa/GPU drivers or use --headless.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
