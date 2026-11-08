"""The QUANTPILOT banner at the top of the TUI.

Only plain ASCII characters (space to ~), so it looks the same in Windows Terminal, the old
Windows console, macOS Terminal and Linux terminals: no block or box-drawing characters that
an old code page would turn into question marks. Drawn with figlet's "standard" font.
"""

from __future__ import annotations

BANNER = (
    r"  ___  _   _   _    _   _ _____ ____ ___ _     ___ _____",
    r" / _ \| | | | / \  | \ | |_   _|  _ \_ _| |   / _ \_   _|",
    r"| | | | | | |/ _ \ |  \| | | | | |_) | || |  | | | || |",
    r"| |_| | |_| / ___ \| |\  | | | |  __/| || |__| |_| || |",
    r" \__\_\\___/_/   \_\_| \_| |_| |_|  |___|_____\___/ |_|",
)

# What a terminal too narrow or too short for the big banner shows instead.
COMPACT = "Q U A N T P I L O T"

TAGLINE = "See what your PC can run. Then quantize it to fit."

BANNER_WIDTH = max(len(line) for line in BANNER)


def banner_text(width: int, height: int) -> str:
    """The big banner when the terminal has room for it, else the one-line name.
    Lines are padded to one length: the TUI centers each line, which would shift short ones."""
    if width >= BANNER_WIDTH + 4 and height >= 30:
        return "\n".join(line.ljust(BANNER_WIDTH) for line in BANNER)
    return COMPACT
