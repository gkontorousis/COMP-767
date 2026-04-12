# Gemini Poster — Setup Guide

A 24×36in portrait LaTeX poster based on the [Gemini beamerposter theme](https://github.com/anishathalye/gemini).

The working file for this project is `gemini_portrait_24x36_3col.tex`.

---

## Prerequisites

### 1. Install TinyTeX

TinyTeX is a lightweight, cross-platform LaTeX distribution. Install it with:

```bash
curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
```

After installation, add the TinyTeX binaries to your PATH (add this to `~/.zshrc` or `~/.bash_profile`):

```bash
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
```

Reload your shell:

```bash
source ~/.zshrc
```

Verify the installation:

```bash
lualatex --version
tlmgr --version
```

### 2. Install Required LaTeX Packages

Run the following command to install all packages needed by the Gemini theme:

```bash
tlmgr install \
  beamerposter beamer pgf \
  lato raleway \
  type1cm fp \
  changepage ragged2e \
  booktabs caption \
  cm-super \
  biblatex biber
```

| Package | Why it's needed |
|---------|-----------------|
| `beamerposter` | Core poster sizing and layout |
| `beamer` | Document class used by beamerposter |
| `pgf` | Graphics/TikZ backend |
| `lato` | Lato OTF font (body text) |
| `raleway` | Raleway OTF font (headlines) |
| `type1cm` | Required internally by beamerposter |
| `fp` | Fixed-point arithmetic used by beamerposter for scaling |
| `changepage` | Required by `beamerthemegemini.sty` for alerted block margins |
| `ragged2e` | `\justifying` command used throughout the theme |
| `booktabs` | `\toprule`, `\midrule`, `\bottomrule` for tables |
| `caption` | `\captionsetup` for figure/table captions |
| `cm-super` | Type1 versions of Computer Modern (math fonts) |
| `biblatex` | Bibliography management (`\cite`, `\printbibliography`) |
| `biber` | Bibliography processor backend for biblatex |

### 3. Install the VS Code Extension

Install the **LaTeX Workshop** extension by James Yu:

1. Open VS Code
2. Go to Extensions (`Cmd+Shift+X`)
3. Search for `James-Yu.latex-workshop`
4. Click **Install**

Or install from the command line:

```bash
code --install-extension James-Yu.latex-workshop
```

---

## Building the Poster

### Option A: VS Code (recommended)

The workspace includes a `.vscode/settings.json` that configures LaTeX Workshop to use **lualatex** automatically (required by the Gemini theme's `fontspec`-based fonts).

1. Open the repo root in VS Code
2. Open `poster/gemini_portrait_24x36_3col.tex`
3. Press `Cmd+Shift+P` → **LaTeX Workshop: Build with recipe**  
   Select **latexmk (lualatex)**  
   (or just save the file — it will build automatically)
4. The PDF preview opens in a side tab

> **Note on build warnings:** You may see "Illegal unit of measure" errors in the LaTeX Compiler output. These are non-fatal — they come from `beamerposter`'s internal `fp` arithmetic and are recovered automatically. The PDF is produced correctly.

### Option B: Command line (latexmk)

```bash
cd poster
latexmk -lualatex -f gemini_portrait_24x36_3col.tex
```

The `-f` flag forces latexmk to finish even when `beamerposter` triggers non-fatal `fp` errors. The output PDF will be `gemini_portrait_24x36_3col.pdf`.

---

## Why LuaLaTeX?

The Gemini theme uses the `fontspec` package (`beamerthemegemini.sty`) to load the **Raleway** and **Lato** OTF fonts. `fontspec` requires either **LuaLaTeX** or **XeLaTeX** — standard `pdflatex` will fail with a `fontspec` error.

The `.latexmkrc` in the `poster/` directory sets `$pdf_mode = 4` (lualatex), and the VS Code workspace settings enforce this recipe.

---

## Files

| File | Description |
|------|-------------|
| `gemini_portrait_24x36_3col.tex` | Main poster source (24×36in, 3 columns) |
| `poster.bib` | BibTeX bibliography |
| `beamerthemegemini.sty` | Gemini layout theme |
| `beamercolorthemegemini.sty` | Default blue color theme |
| `beamercolorthememit.sty` | MIT color theme |
| `beamercolorthemelabsix.sty` | LabSix color theme |
| `.latexmkrc` | latexmk config (lualatex engine + biber) |

---

## Troubleshooting

**`File 'changepage.sty' not found`**  
Run: `tlmgr install changepage`

**`File 'fontspec.sty' not found`** or font errors  
Make sure you're using **lualatex**, not pdflatex. Check that the recipe in VS Code is set to `latexmk (lualatex)`.

**VS Code is using pdflatex instead of lualatex**  
Open the workspace from the repo root so `.vscode/settings.json` is picked up. Alternatively, add the `% !TEX program = lualatex` magic comment at the top of your `.tex` file (already present in `gemini_portrait_24x36_3col.tex`).

**`tlmgr: permission denied`**  
If TinyTeX was installed as root, prefix with `sudo`. If installed as the current user (default), it should not require sudo.
