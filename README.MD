# raildelays

## Reproducing the Results

### Setup
Note: The computer we use to run this model runs Ubuntu 18.04 and uses CUDA V9.1 to facilitate model training using a GPU. While this code may work with other configurations, we did not test other configurations during our development.

1. Clone this repository to your local machine

1. Create a data folder in the project base directory with the following subdirectories
    - ./data/raw
    - ./data/interim
    - ./data/processed

1. Download and unzip the raw data from this [Google Drive link](https://drive.google.com/file/d/1G1LzyftvTsMUPmKT2fMtPN-ceanhytEW/view?usp=sharing) to ./data/raw

1. Create and activate the raildelays conda environment using the following command

        $ conda env create -f environment.yml

1. Run jupyter notebook in the notebooks directory

        $ cd ./notebooks
        $ jupyter notebook

    - Open the jupyter notebook in a browser, and open the "create_data" notebook
    - Change the base_dir in the first cell to the base directory for this repository
    - Run all the cells in "Imports" and "Link-Based Node Formulation" sections. This creates the input dataset for the STGCN model.

### Running Models

1. Open the src/main.py and src/model_comparison.py and change the base dir to the raildelays base directory

1. Run the STGCN model using the following command

        $ cd src
        $ python main.py --n_timesteps_in=6 --n_timesteps_future=1

where n_timesteps_in can take values {6, 12} and n_timesteps_future can take values {1, 3, 6} for the experiments presented in the paper. This trains the model for a default of 25 epochs. The accuracy metrics are shown in the terminal after inference.

1. Run the comparison models using the following command

        $ python model_comparison.py --n_timesteps_in=6 --n_timesteps_future=1 --model_type=MLP


where n_timesteps_in can take values {6, 12}, n_timesteps_future can take values {1, 3, 6}, model_type can be {LR, MLP} for the experiments presented in the paper. The accuracy metrics are shown in the terminal after inference.
