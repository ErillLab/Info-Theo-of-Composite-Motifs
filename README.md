# Information Theory of Composite Motifs

Computational simulations in support of our work on the information theory of composite motifs [(Mascolo & Erill, 2024)](https://doi.org/10.1101/2024.11.11.623117)

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```

2. Install Python requirements
   
   The dependencies are listed in the `itcm_env_minimal.yml` file.

   [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) users can create a ready-to-use environment with
   ```sh
   conda env create -f itcm_env_complete.yml
   ```

## Usage

### Run evolutionary simulations

Evolutionary simulations can be run by running the `evolve_reg_sys.py` script.

The settings can be chosen by editing the `settings.json` file.

### Reproduce the figures in our pre-print

#### Gather data

The results of all our *in silico* experiments can be obtained by unzipping the `results.zip` folder.

#### Analyze data

The data can be analyzed to regenerate all the Figures in [Mascolo & Erill (2024)](https://doi.org/10.1101/2024.11.11.623117). They will be saved as PNG files.

This can be done in two ways:

1. By running the `analyze_results.py` script in the `src` folder.

2. By running the `regenerate_figures.ipynb` jupyter notebook in the `src` folder. The notebook allows for step-by-step serial regeneration of all the figures and provides brief descriptions and explanations.



