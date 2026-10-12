"""Settings loaded from environment variables (and an optional .env file).

We read .env ourselves with a few lines of code instead of adding a dependency.
Real environment variables always win over values in .env.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Read KEY=VALUE lines from a .env file into os.environ (if not already set)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


load_env_file()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini").strip()
