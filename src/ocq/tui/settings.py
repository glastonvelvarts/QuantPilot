"""What the TUI remembers between runs, saved as JSON in the user's config folder.

The folder is the usual one for each system (platformdirs):
  Linux    ~/.config/ocq/tui.json
  macOS    ~/Library/Application Support/ocq/tui.json
  Windows  %LOCALAPPDATA%\\ocq\\tui.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from platformdirs import user_config_dir

from ocq.model_analyzer import DEFAULT_CONTEXT
from ocq.types import KV_CACHE_TYPES, OptimizationGoal

SETTINGS_FILE = Path(user_config_dir("ocq", appauthor=False)) / "tui.json"
RECENT_LIMIT = 10


@dataclass
class Settings:
    theme: str = "textual-dark"
    show_banner: bool = True
    ascii_borders: bool = False  # +--+ borders for terminals that cannot draw lines
    context: int = DEFAULT_CONTEXT
    kv_cache: str = "f16"
    goal: str = OptimizationGoal.BALANCED.value
    model_folders: list[str] = field(default_factory=list)  # searched besides Ollama, LM Studio...
    llmfit_bin: str = ""  # empty: the llmfit that came with QuantPilot
    recent_models: list[str] = field(default_factory=list)  # newest first

    def remember_model(self, ref: str) -> None:
        """Puts a model the user just ran something on at the top of the recent list."""
        ref = ref.strip()
        if ref:
            self.recent_models = [ref] + [m for m in self.recent_models if m != ref]
            del self.recent_models[RECENT_LIMIT:]

    def folders(self) -> list[Path]:
        return [Path(folder).expanduser() for folder in self.model_folders if folder.strip()]


def load_settings(path: Path = SETTINGS_FILE) -> Settings:
    """The saved settings, or the defaults when there is no file or it cannot be read.
    Values of the wrong type are ignored one by one, so one bad line does not reset the rest."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    if not isinstance(data, dict):
        return Settings()
    settings = Settings()
    for item in fields(Settings):
        value = data.get(item.name)
        # Exact type, so true/false is not taken for a number (bool is an int in Python).
        if type(value) is type(getattr(settings, item.name)):
            setattr(settings, item.name, value)
    if settings.kv_cache not in KV_CACHE_TYPES:
        settings.kv_cache = "f16"
    if settings.goal not in {goal.value for goal in OptimizationGoal}:
        settings.goal = OptimizationGoal.BALANCED.value
    if settings.context <= 0:
        settings.context = DEFAULT_CONTEXT
    settings.model_folders = [str(f) for f in settings.model_folders if isinstance(f, str)]
    settings.recent_models = [str(m) for m in settings.recent_models if isinstance(m, str)]
    return settings


def save_settings(settings: Settings, path: Path = SETTINGS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")
