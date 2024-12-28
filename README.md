# Information Theory of Composite Motifs

Computational simulations in support of our work on the information theory of composite motifs

## Installation

The dependencies are listed in the `ITCM_env.yml` file. Conda users can create a specific environment with

	conda env create -f ITCM_env.yml

## Usage

### Evolutionary simulations

Run evolutionary simulations by running the `evolve_reg_sys.py` script. The settings can be chosen by editing the `settings.json` file.

### Reproduce the figures in our pre-print

The results of all our *in silico* experiments can be obtained by unzipping the `results.zip` folder. They can be analyzed by running the `analyze_results.py` script, which will regenerate all the plots in [Mascolo & Erill (2024)](https://doi.org/10.1101/2024.11.11.623117)


