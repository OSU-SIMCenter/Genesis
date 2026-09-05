"""WSLg OpenGL/Gallium defaults that must run before OpenGL is imported.

Kept free of OpenGL/genesis imports so callers can invoke this at module
load (e.g. before ``import genesis``).
"""

from __future__ import annotations

import os
import sys


def apply_early_wsl_graphics_defaults() -> None:
    """Prefer Mesa D3D12 + NVIDIA adapter + GLX on WSLg.

    Intel UHD as the default D3D12 adapter with CUDA on NVIDIA often
    segfaults during visualizer build; pin NVIDIA when unset.
    """
    if not sys.platform.startswith("linux"):
        return
    if not os.environ.get("WSL_DISTRO_NAME"):
        return
    if not os.path.exists("/dev/dxg"):
        return
    if not os.environ.get("GALLIUM_DRIVER") and not os.environ.get(
        "MESA_LOADER_DRIVER_OVERRIDE"
    ):
        os.environ["GALLIUM_DRIVER"] = "d3d12"
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
