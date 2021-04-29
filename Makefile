# Subroutines.
PYTHON = PYTHONPATH="`pwd`/src:$(PYTHONPATH)" python3
LATEXMK = latexmk -pdf -interaction=nonstopmode -cd

# Main paper.
paper/scalar_coverings.pdf: tex/scalar_coverings/*.tex tex/scalar_coverings/*.sty tex/scalar_coverings/*.cls tex/scalar_coverings/tikz/* tex/scalar_coverings/ratio_intervals_proof.tex figs
	$(LATEXMK) tex/scalar_coverings/main.tex
	mv tex/scalar_coverings/main.pdf paper/scalar_coverings.pdf

# Abridged version of main paper (no appendix, references arXiv URL, has JMLR header).
paper/scalar_coverings_noappdx.pdf: tex/scalar_coverings/*.tex tex/scalar_coverings/*.bib tex/scalar_coverings/*.sty tex/scalar_coverings/*.cls tex/scalar_coverings/tikz/* tex/scalar_coverings/ratio_intervals_proof.tex figs
	ABRIDGED=true $(LATEXMK) tex/scalar_coverings/main.tex
	mv tex/scalar_coverings/main.pdf paper/scalar_coverings_noappdx.pdf

# LaTeX source code for arXiv, stripped of comments.
tex/scalar_coverings_arXiv: figs
	cd tex/scalar_coverings; latexmk -c; rm -f main.pdf
	arxiv_latex_cleaner tex/scalar_coverings
	cp -r figures tex/scalar_coverings_arXiv

# Zipping the arXiv source for upload.
paper/scalar_coverings_arXiv.zip: tex/scalar_coverings_arXiv
	rm -f paper/scalar_coverings_arXiv.zip
	zip -r paper/scalar_coverings_arXiv.zip tex/scalar_coverings_arXiv

# All the source code for github.
github: tex/scalar_coverings_arXiv
	rm -rf github
	mkdir -p github
	cp conda_env.yaml github
	cp Makefile github
	cp matplotlibrc_template github
	cp .gitignore github
	mkdir -p github/src
	cp -r src/*.py github/src
	cp -r src/scalar_coverings github/src/scalar_coverings
	py3clean src
	mkdir github/tex
	cp tex/preamble.tex github/tex
	cp -r tex/scalar_coverings_arXiv github/tex/scalar_coverings
	mkdir -p github/data
	cp data/.gitignore github/data
	mkdir -p github/figures
	cp figures/.gitignore github/figures
	mkdir -p github/paper
	cp paper/.gitignore github/paper


# Tell Make about all figures manually. TODO: automate based on data/plot code.
figs: \
figures/covering_grid_ratios.pgf \
figures/covering_quadrotor.pgf \
figures/geomgrid.pgf \
figures/neighborhoods_2x2.pgf \
figures/neighborhoods_3x3.pgf \


# Rules to make LaTeX from code.
# LaTeX-generating scripts should write to stdout.

tex/scalar_coverings/ratio_intervals_proof.tex: src/scalar_coverings/scalar_quasiconvexity.py
	$(PYTHON) src/scalar_coverings/scalar_quasiconvexity.py > tex/scalar_coverings/ratio_intervals_proof.tex


# Rules to make figures from dataframes.
# Figure-generating scripts should take the input data path and output image
# path as the last two command-line args.

figures/%.pgf: data/%.feather src/scalar_coverings/%_plot.py matplotlibrc tex/preamble.tex
	$(PYTHON) src/scalar_coverings/$*_plot.py data/$*.feather $@


# Rule to make dataframes from code.
# Data-generating scripts should take the output data path as the last
# command-line arg.

.PRECIOUS: data/%.feather data/%.npz data/%.pickle

data/%.feather: src/*.py src/scalar_coverings/%_data.py
	$(PYTHON) src/scalar_coverings/$*_data.py $@


# Get around annoying flaw of no relative "imports" in matplotlibrc
matplotlibrc: matplotlibrc_template
	sed "s|%PWD%|`pwd`|g" matplotlibrc_template > matplotlibrc


tex/packages.txt:
	grep -r --include "*.tex" --include "*.sty" -h usepackage ./tex \
		| awk -F"{|}" '{print $$2}' \
		| sed 's/,/\n/' \
		| sed 's/ //' \
		> ./tex/packages.txt


clean:
	rm -rf data/*
	rm -rf figures/*
	rm -rf paper/*
