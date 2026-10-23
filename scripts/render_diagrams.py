"""Render pipeline diagrams (diagram.mmd -> diagram.svg) with mermaid-cli.

Usage::

    uv run python scripts/render_diagrams.py [--check] [--force] [MMD ...]

Without arguments, renders every ``pipelines/*/diagram.mmd`` (templates included). Each SVG is
stamped with the SHA-256 of its source, so an SVG is re-rendered only when its ``.mmd``
changed (or with ``--force``). ``--check`` renders nothing and exits with 1 if an SVG is
missing or stale; the test suite runs the same check.

mermaid-cli is looked up in ``$MMDC``, then ``.tools/`` (see scripts/setup_mermaid.sh), then
``PATH``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
STAMP_PREFIX = "<!-- source-sha256: "


def source_hash(mmd: Path) -> str:
    return hashlib.sha256(mmd.read_bytes()).hexdigest()


def stamp_of(svg: Path) -> str | None:
    """The source hash stamped on the first line of ``svg``, if any."""
    if not svg.exists():
        return None
    first_line = svg.read_text(encoding="utf-8").split("\n", 1)[0]
    if first_line.startswith(STAMP_PREFIX) and first_line.endswith(" -->"):
        return first_line[len(STAMP_PREFIX) : -len(" -->")]
    return None


def is_up_to_date(mmd: Path) -> bool:
    return stamp_of(mmd.with_suffix(".svg")) == source_hash(mmd)


def find_mmdc() -> tuple[str, dict[str, str]]:
    """Path to mmdc and the environment to run it with."""
    env = dict(os.environ)
    if "MMDC" in env:
        return env["MMDC"], env
    local = TOOLS / "mermaid" / "node_modules" / ".bin" / "mmdc"
    if local.exists():
        node_bins = sorted(TOOLS.glob("node-*/bin"))
        if node_bins:
            env["PATH"] = f"{node_bins[-1]}{os.pathsep}{env.get('PATH', '')}"
        env.setdefault("PUPPETEER_CACHE_DIR", str(TOOLS / "puppeteer"))
        return str(local), env
    found = shutil.which("mmdc")
    if found:
        return found, env
    sys.exit("mmdc not found: run scripts/setup_mermaid.sh (or set $MMDC)")


def render(mmd: Path, mmdc: str, env: dict[str, str]) -> None:
    svg = mmd.with_suffix(".svg")
    subprocess.run([mmdc, "--quiet", "-i", str(mmd), "-o", str(svg)], check=True, env=env)
    body = svg.read_text(encoding="utf-8")
    svg.write_text(f"{STAMP_PREFIX}{source_hash(mmd)} -->\n{body}", encoding="utf-8")


def default_sources() -> list[Path]:
    return sorted((ROOT / "pipelines").glob("*/diagram.mmd"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", type=Path, help="diagram.mmd files (default: all)")
    parser.add_argument("--check", action="store_true", help="only report missing/stale SVGs")
    parser.add_argument("--force", action="store_true", help="re-render even if up to date")
    args = parser.parse_args(argv)

    sources = [p.resolve() for p in args.sources] or default_sources()
    stale = [p for p in sources if args.force or not is_up_to_date(p)]
    if args.check:
        for p in stale:
            print(f"stale or missing SVG: {p.with_suffix('.svg').relative_to(ROOT)}")
        return 1 if stale else 0

    if stale:
        mmdc, env = find_mmdc()
        for p in stale:
            render(p, mmdc, env)
            print(f"rendered {p.with_suffix('.svg').relative_to(ROOT)}")
    print(f"{len(sources) - len(stale)} up to date, {len(stale)} rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
