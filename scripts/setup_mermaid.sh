#!/usr/bin/env bash
# Install mermaid-cli under .tools/ (git-ignored), for scripts/render_diagrams.py.
#
# Everything is pinned: the Node.js tarball (version and SHA-256, checked) and the whole npm
# dependency tree (scripts/mermaid/package-lock.json, installed with `npm ci`; it was resolved
# with `npm install --before=2026-10-09`). Puppeteer downloads chrome-headless-shell into
# .tools/puppeteer (about 950 MB in total under .tools/). Re-running is a no-op once installed.
#
# Rendered SVGs are committed (decision D6): this is only needed where diagrams are edited.
#
# Usage: scripts/setup_mermaid.sh
set -euo pipefail

NODE_VERSION=v24.21.0  # LTS "Krypton", released 2026-09-07
declare -A NODE_SHA256=(  # from https://nodejs.org/dist/v24.21.0/SHASUMS256.txt
  [x64]=fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6
  [arm64]=6ad1325edbdb5649c379b75a237147a666c95d4f9ae8d340fef2d1575d289ad2
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/.tools"
case "$(uname -m)" in
  x86_64) ARCH=x64 ;;
  aarch64 | arm64) ARCH=arm64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

NODE_DIR="$TOOLS/node-$NODE_VERSION-linux-$ARCH"
if [[ ! -x "$NODE_DIR/bin/node" ]]; then
  mkdir -p "$TOOLS/downloads"
  tarball="$TOOLS/downloads/node-$NODE_VERSION-linux-$ARCH.tar.xz"
  curl -fsSL --retry 3 -o "$tarball" \
    "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-$ARCH.tar.xz"
  echo "${NODE_SHA256[$ARCH]}  $tarball" | sha256sum --check --quiet
  tar -xJf "$tarball" -C "$TOOLS"
  rm -f "$tarball"
fi
export PATH="$NODE_DIR/bin:$PATH"
export PUPPETEER_CACHE_DIR="$TOOLS/puppeteer"
export PUPPETEER_CHROME_SKIP_DOWNLOAD=true  # mmdc launches chrome-headless-shell only

MERMAID_DIR="$TOOLS/mermaid"
LOCK="$ROOT/scripts/mermaid/package-lock.json"
if [[ ! -x "$MERMAID_DIR/node_modules/.bin/mmdc" ]] || ! cmp -s "$LOCK" "$MERMAID_DIR/package-lock.json"; then
  mkdir -p "$MERMAID_DIR"
  cp "$ROOT/scripts/mermaid/package.json" "$LOCK" "$MERMAID_DIR/"
  npm ci --prefix "$MERMAID_DIR" --no-audit --no-fund
fi

echo "mmdc ready: $MERMAID_DIR/node_modules/.bin/mmdc ($("$MERMAID_DIR/node_modules/.bin/mmdc" --version))"
