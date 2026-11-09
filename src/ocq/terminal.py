from __future__ import annotations

import sys


def prepare_terminal() -> None:
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not reconfigure:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def status_glyph(unicode_char: str, ascii_char: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        unicode_char.encode(enc)
    except (UnicodeEncodeError, LookupError):
        return ascii_char
    return unicode_char


def rich_status(unicode_char: str, ascii_char: str, style: str) -> str:
    return f"[{style}]{status_glyph(unicode_char, ascii_char)}[/{style}]"
