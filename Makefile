# Subroutines.
PYTHON = PYTHONPATH="`pwd`/src:$(PYTHONPATH)" python3
LATEXMK = latexmk -pdf -interaction=nonstopmode -cd

# Main paper.
paper/scalar_coverings.pdf: tex/scalar_coverings/*.tex tex/scalar_coverings/*.bib tex/scalar_coverings/*.sty tex/scalar_coverings/*.cls tex/scalar_coverings/tikz/* tex/scalar_coverings/ratio_intervals_proof.tex figs
	$(LATEXMK) tex/scalar_coverings/main.tex
	mv tex/scalar_coverings/main.pdf paper/scalar_coverings.pdf

abridged: paper/scalar_coverings_noappdx.pdf

paper/scalar_coverings_noappdx.pdf: tex/scalar_coverings/*.tex tex/scalar_coverings/*.bib tex/scalar_coverings/*.sty tex/scalar_coverings/*.cls tex/scalar_coverings/tikz/* tex/scalar_coverings/ratio_intervals_proof.tex figs
	ABRIDGED=true $(LATEXMK) tex/scalar_coverings/main.tex
	mv tex/scalar_coverings/main.pdf paper/scalar_coverings_noappdx.pdf

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


clean:
	rm -rf data/*
	rm -rf figures/*
	rm -rf paper/*

