# ICLR 2027 submission source

Build:

    latexmk -pdf main.tex

or, without latexmk:

    pdflatex main && bibtex main && pdflatex main && pdflatex main

## Files

- `main.tex` — the paper. Submission mode (author names suppressed, line-number
  rulers on). Uncomment `\iclrfinalcopy` for the camera-ready.
- `references.bib` — bibliography, 19 entries.
- `iclr2027_conference.{sty,bst}`, `fancyhdr.sty`, `natbib.sty`,
  `math_commands.tex` — unmodified ICLR 2027 style files.

## Red markup

Everything typeset in red is an open item. Two macros, defined at the top of
`main.tex`:

- `\chknote{note}{label}` — a value or definition to verify against the code.
- `\pend{text}{label}`    — a claim a scheduled run may change; the text stays
                            readable, with a red superscript tag.

Each red marker carries a bracketed tag (`[C3]`, `[T1.2]`, `[L1]`, `[FIG1]`)
pointing at the matching entry in **Appendix A, "Working notes: outstanding
items"**, which holds the checks, the Tier 1/2/3 run queue, the facts to look
up, the figures to produce, and the acknowledge-only list.

To produce a clean copy with no red and no working notes: redefine `\red` and
`\opentag` as no-ops and delete the `\section{Working notes...}` block.

## Figures

Left out for now. The two figure slots are marked in red in the text
(§4.2 and §4.5) and itemised as FIG1/FIG2 in Appendix A.

## Page count

ICLR 2027 allows 9 pages of main text (references, appendices and the
reproducibility statement are excluded). The clean copy currently runs to
about 10.7 pages of body before any figures are added, so roughly 1.5–2 pages
need to come out. §3's six-check list and §5's limitation lists are the
longest blocks.

## Local toolchain

TinyTeX was installed at `~/Library/TinyTeX` and is not on PATH. To build:

    export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
    pdflatex main && bibtex main && pdflatex main && pdflatex main
