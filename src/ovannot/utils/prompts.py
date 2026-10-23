"""Prompt templates: prompts live in files (``prompts/*.txt`` or ``*.jinja``), never in code.

Templates use Jinja2 with ``StrictUndefined``: rendering fails loudly if a variable is
missing, instead of silently sending an incomplete prompt to a model.
"""

from __future__ import annotations

from pathlib import Path

import jinja2
from jinja2 import meta

_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
    autoescape=False,  # prompts are plain text, not HTML
)


def render_prompt(path: str | Path, **variables: object) -> str:
    """Render the template at ``path`` with ``variables``."""
    return _ENV.from_string(Path(path).read_text(encoding="utf-8")).render(**variables)


def prompt_variables(path: str | Path) -> set[str]:
    """Names of the variables a template expects (used by docs and tests)."""
    return meta.find_undeclared_variables(_ENV.parse(Path(path).read_text(encoding="utf-8")))
