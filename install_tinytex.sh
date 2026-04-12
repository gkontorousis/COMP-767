#!/usr/bin/env bash
# install.sh — macOS setup for COMP-767 LaTeX projects
# Installs TinyTeX and all packages required by poster/ and project_paper/
# Usage: bash install.sh

set -e

TINYTEX_BIN="$HOME/Library/TinyTeX/bin/universal-darwin"

# ── 1. Install TinyTeX ────────────────────────────────────────────────────────
if command -v tlmgr &>/dev/null; then
  echo "✓ TinyTeX already installed ($(tlmgr --version | head -1))"
else
  echo "→ Installing TinyTeX..."
  curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh

  # Add to PATH for the rest of this script
  export PATH="$TINYTEX_BIN:$PATH"
  echo "✓ TinyTeX installed"

  echo ""
  echo "  Add TinyTeX to your PATH by adding this line to ~/.zshrc or ~/.bash_profile:"
  echo "    export PATH=\"$TINYTEX_BIN:\$PATH\""
  echo "  Then run: source ~/.zshrc"
  echo ""
fi

# Ensure tlmgr is on PATH even if TinyTeX was already installed
export PATH="$TINYTEX_BIN:$PATH"

# ── 2. Update tlmgr ───────────────────────────────────────────────────────────
echo "→ Updating tlmgr..."
tlmgr update --self --all --no-auto-install 2>/dev/null || true

# ── 3. Install packages for poster/ (lualatex + biblatex/biber) ───────────────
echo "→ Installing poster packages (lualatex)..."
tlmgr install \
  beamerposter beamer pgf \
  lato raleway \
  type1cm fp \
  changepage ragged2e \
  booktabs caption \
  cm-super \
  biblatex biber

# ── 4. Install packages for project_paper/ (pdflatex + bibtex) ───────────────
echo "→ Installing project_paper packages (pdflatex)..."
tlmgr install \
  microtype \
  natbib \
  eso-pic \
  forloop \
  psnfss \
  float \
  xcolor \
  preprint

echo ""
echo "✓ All done. Both projects are ready to build."
echo ""
echo "  poster/        → lualatex via latexmk (outputs to .latex-out/)"
echo "  project_paper/ → pdflatex via latexmk (outputs to .latex-out/)"
