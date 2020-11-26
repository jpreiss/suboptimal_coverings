# 	Suboptimal coverings for continuous spaces of control tasks

This repository contains the source code to reproduce the paper
*Suboptimal coverings for continuous spaces of control tasks*
by James A. Preiss and Gaurav S. Sukhatme.

Instructions
------------
- Create the conda environment: `conda env create -f conda_env.yaml`
- Activate the conda environment: `conda activate lqr`
- Build the paper: `make -j8`.

This runs all of the computations for the empirical results from scratch. It will take a long time, around 15 minutes.

The result is stored in `papers/`.
