# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sharp 4D Video Plugin for LichtFeld Studio.

Uses SHARP to generate 4D Gaussian Splats from video input.
"""

import lichtfeld as lf

from .panels import SharpVideoPanel

_panel_class = None


def on_load():
    """Called when plugin loads."""
    global _panel_class

    _panel_class = SharpVideoPanel
    lf.ui.register_panel(SharpVideoPanel)
    lf.log.info("Sharp 4D Video plugin loaded")


def on_unload():
    """Called when plugin unloads."""
    global _panel_class

    if _panel_class:
        lf.ui.unregister_panel(_panel_class)
        _panel_class = None
    lf.log.info("Sharp 4D Video plugin unloaded")


__all__ = [
    "SharpVideoPanel",
]