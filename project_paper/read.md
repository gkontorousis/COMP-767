# TinyTeX + VS Code Setup on macOS

This guide sets up a macOS machine to build the paper PDF locally with TinyTeX and VS Code.

## 1. Install TinyTeX

Install TinyTeX with the official script:

```bash
curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
```

Add TinyTeX to your shell `PATH` in `~/.bashrc`:

```bash
export PATH="$HOME/bin:$PATH"
```

Reload your shell:

```bash
source ~/.bashrc
```

Verify the install:

```bash
which pdflatex
which latexmk
tlmgr --version
```

Update TinyTeX packages once after install:

```bash
tlmgr update --self --all
```

## 2. Install VS Code extension

Install the `LaTeX Workshop` extension in VS Code.

Recommended optional extension:

- `LTeX` or `Code Spell Checker` for prose/spelling help

## 3. Recommended VS Code settings

Open VS Code settings JSON and add:

```json
{
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.latex.outDir": "%DIR%/build",
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk (pdf)",
      "tools": ["latexmk"]
    }
  ],
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": [
        "-pdf",
        "-interaction=nonstopmode",
        "-synctex=1",
        "-outdir=%OUTDIR%",
        "%DOC%"
      ]
    }
  ]
}
```

## 4. Build the PDF

If the paper source is in a file like `main.tex` or `paper.tex`, build it from `project_paper/` with:

```bash
latexmk -pdf -interaction=nonstopmode -synctex=1 -outdir=build main.tex
```

If the main file has a different name, replace `main.tex` with the actual `.tex` entrypoint.

To clean build artifacts:

```bash
latexmk -c -outdir=build
```

## 5. If TinyTeX reports missing packages

Install missing packages with `tlmgr`. Common useful ones:

```bash
tlmgr install latexmk
tlmgr install collection-latexrecommended
tlmgr install collection-latexextra
tlmgr install collection-fontsrecommended
tlmgr install bibtex
```

Then run the build again.

## 6. Important prerequisite

This folder currently contains `comp-767-final-paper.pdf`, but I do not see the LaTeX source file in the workspace view. To regenerate the PDF, you also need:

- the main `.tex` file
- any bibliography files such as `.bib`
- any figures or other included assets
- any custom class/style files such as `.cls`, `.sty`, or `.bst`

If only the PDF exists, it cannot be rebuilt until the original LaTeX source is added to `project_paper/`.

## 7. Typical workflow in VS Code

1. Open the LaTeX source file in VS Code.
2. Save the file to trigger auto-build, or run `LaTeX Workshop: Build LaTeX project`.
3. Open the PDF preview from LaTeX Workshop.
4. If the build fails, read the LaTeX Workshop log and install any missing package with `tlmgr`.
