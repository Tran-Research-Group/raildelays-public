############################
# imports
############################
# external libraries
import os
import sys
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir = r"/home/jacobheglund/dev/raildelays-public"
# change directory for loading files from disk and loading custom dictionaries
# these commands MUST be in this order to get the desired effect
sys.path.append(base_dir)
os.chdir(base_dir)
print(os.getcwd())

# custom libraries
from src.data.data_processing import data_interface
import src.utils.model_comparison_utils as model_comparison_utils

parser = argparse.ArgumentParser()
# model training code
## using the F1 features
# data_train = np.load("./data/processed/baseline_train.npy")
# data_test = np.load("./data/processed/baseline_test.npy")
dataset = "raildelays"
data_dir = "./data/processed/" + dataset + "/"
model_type = "MLP"

if dataset == "raildelays":
    n_nodes = 40
    n_timesteps_per_day = 42
    n_timesteps_in = 12
    n_timesteps_future = 1
    n_features_in = 1
    n_features_out = 1

# data parameters
parser.add_argument("--n_nodes", type=int, default=n_nodes)
parser.add_argument("--n_timesteps_per_day", type=int, default=n_timesteps_per_day)
parser.add_argument("--n_timesteps_in", type=int, default=n_timesteps_in)
parser.add_argument("--n_timesteps_future", type=int, default=n_timesteps_future)
parser.add_argument("--n_features_in", type=int, default=1)
parser.add_argument("--approx", type=str, default="cheb_poly", choices={"cheb_poly", "first_order"})
parser.add_argument("--ks", type=int, default=3)
parser.add_argument("--model_type", type=str, default="LR", choices={"LR", "MLP"})
parser.add_argument("--n_epochs", type=int, default=50)

# GPU setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using Device: CUDA")

else:
    device = torch.device("cpu")
    print("Using Device: CPU")
parser.add_argument("--device", type=str, default=device)
args = parser.parse_args()


_, data_train, data_test, data_val, output_stats = data_interface(data_dir,
                                                    dataset,
                                                    args.n_nodes,
                                                    args.ks,
                                                    args.approx,
                                                    device,
                                                    args.n_timesteps_per_day,
                                                    args.n_timesteps_in,
                                                    args.n_timesteps_future,
                                                    args.n_features_in)


mae_node = []
rmse_node = []

if args.model_type == "LR":
    for i in range(args.n_nodes):
        # pick out particular node's data
        model_data_train = data_train[0][:, :, i, :].squeeze(), data_train[1][:, :, i, :].squeeze()
        model_data_test = data_test[0][:, :, i, :].squeeze(), data_test[1][:, :, i, :].squeeze()

        # fit model, report results
        err = model_comparison_utils.linear_regression(model_data_train, model_data_test)
        mae_node.append(err[0])
        rmse_node.append(err[1])


elif args.model_type == "MLP":

    model = model_comparison_utils.MLP(args.n_timesteps_in, 100, 1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    for i in range(args.n_nodes):
        # pick out particular node's data
        model_data_train = data_train[0][:, :, i, :].squeeze().to(device).float(), data_train[1][:, :, i, :].squeeze(axis=2).float()
        model_data_test = data_test[0][:, :, i, :].squeeze().to(device).float(), data_test[1][:, :, i, :].squeeze(axis=2).float()
        # fit model, report results
        err = model_comparison_utils.model_train(model_data_train, model_data_test, model, optimizer, criterion, args.n_epochs, args.device)
        mae_node.append(err[0])
        rmse_node.append(err[1])

print(args.model_type, "n_past = {}".format(args.n_timesteps_in), " n_future = {}".format(args.n_timesteps_future))
print("MAE", np.mean(mae_node))
print("RMSE:", np.mean(rmse_node))
