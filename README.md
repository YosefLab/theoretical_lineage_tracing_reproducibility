Reproducibility Repo for `Theoretical Guarantees for Phylogeny Inference from Single-Cell Lineage Tracing`
=============================================================

This repository stores the analysis and results needed to reproduce the figures found in this [work](https://www.biorxiv.org/content/10.1101/2021.11.21.469464v1).

The analysis makes use of the Cassiopeia package for lineage tracing found [here](https://github.com/YosefLab/Cassiopeia).

The main simulation scripts to produce the simulations showing the empirical performance of the algorithmic approaches are in the `scripts` folder.

Bash scripts that interface with TORQUE were used to parallelize runs in the original analysis: `run_sim_validate_k.sh`, `run_sim_vary_lambda_q.sh`, `run_sim_vary_n.sh`. Any parallelization system that interfaces with the scripts can be used.

The `notebooks` folder contains .ipynb notebooks that are used to generate the figures and intermediate files used in the analysis. 

Documentation is included in each individual file.


