"""Small portable artifact helpers shared by training entry points."""

import json
from pathlib import Path


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
