from ocq.terminal import rich_status, status_glyph


def test_status_glyph_falls_back_on_cp1252(monkeypatch):
    class FakeOut:
        encoding = "cp1252"

    monkeypatch.setattr("ocq.terminal.sys.stdout", FakeOut())
    assert status_glyph("\u2713", "+") == "+"
    assert status_glyph("!", "!") == "!"
    assert rich_status("\u2713", "+", "green") == "[green]+[/green]"
