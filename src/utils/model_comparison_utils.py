from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


###################################
# Data Formatting
###################################


def process_data(df_data, filepath, hour_start, hour_end, verbose=False):
    # remove rows outside of [hour_start, hour_end]
    df_data = df_data.loc[df_data["hour_of_day"]%24 >= hour_start]
    df_data = df_data.loc[df_data["hour_of_day"]%24 <= hour_end]

    date_list = df_data["datetime"].unique()
    n_timesteps = len(date_list)
    n_nodes = 30
    #TODO arr_delay, dep_delay, scheduled_flights
    drop_list_1 = ["arr_delay", "carrier_delay", "day_of_week", "day_of_year", "hour_of_day",
                "late_aircraft_delay", "weather_delay", "arr_delay_class", "day_of_week_cos",
                "day_of_week_sin", "day_of_year_cos", "day_of_year_sin", "hour_of_data",
                "hour_of_day_cos", "hour_of_day_sin"]

    df_data = df_data.drop(labels=drop_list_1, axis=1)

    # drop everything except feature_cols
    drop_list_2 = ["datetime", "datetime_est", "origin_airport_code", "timedelta_est"]

    n_features = np.shape(df_data)[1] - len(drop_list_2)
    dataset = np.zeros((n_timesteps, n_nodes, n_features))
    if verbose:
        print("Extracting Data")

    for i in range(n_timesteps):
        if verbose:
            if i % 2000 == 0:
                print("{}/{}".format(i, len(date_list)))

        # get the data for all airports at each unique datetime
        df_tmp = df_data.loc[df_data["datetime"] == date_list[i]]
        df_tmp.sort_values("origin_airport_code")

        # drop unnecessary features
        df_tmp = df_tmp.drop(columns = drop_list_2)
        feature_vec = df_tmp.to_numpy()
        dataset[i] = feature_vec

    np.save(filepath, dataset)
    del dataset


def sequence_data(dataset, n_timesteps_per_day, n_timesteps_in, n_timesteps_out, n_nodes, n_features_in):
    """loads data from disk and processes into sequences

    Args:
        dataset.shape = (n_timesteps_total, n_nodes, n_features_in)
        n_nodes (int): number of nodes on the graph
        n_timesteps_per_day (int): number of timesteps included in each day of the data (EX: 0400 - 2400: 20 hours)
        n_timesteps_in (int): number of timesteps to include as model input
        n_timesteps_out (int): number of timesteps to include as model labels, also number of timesteps that are predicted

    Returns:
        torch.Tensor: input and labels for the model, only the first dimension chances for the other outputs
        shape(train_input) = (n_sequences_train, n_nodes, n_timesteps_in, n_features_in)
        shape(train_label) = (n_sequences_train, n_nodes, n_timesteps_out, n_features_out = 1)
    """
    n_days = int(np.shape(dataset)[0] / n_timesteps_per_day)
    n_timesteps_seq = n_timesteps_in + n_timesteps_out

    # number of sequences per day
    n_slot = n_timesteps_per_day - n_timesteps_seq + 1

    # total number of sequences
    n_sequences = (n_slot-1) * n_days
    dataset_seq = np.zeros((n_sequences, n_timesteps_seq, n_nodes, n_features_in))

    # get to the correct day
    counter = 0
    for i in range(n_days):
        curr_data = dataset[i * n_timesteps_per_day : (i + 1) * n_timesteps_per_day]

        # get the input-output sequences within the day
        for j in range(n_slot-1):
            input_seq = curr_data[j : j + n_timesteps_in]
            output_seq = curr_data[j + n_timesteps_in : j + n_timesteps_in + n_timesteps_out]
            tmp_data = np.expand_dims(np.concatenate((input_seq, output_seq)), 0)
            dataset_seq[counter] = tmp_data
            counter += 1

    return dataset_seq


def generate_input_label(dataset, n_timesteps_in):
    # data_seq.shape = (n_seq, n_timesteps_seq = n_timesteps_in + n_timesteps_out, n_airports, n_features_in)
    data_input = dataset[:, 0:n_timesteps_in, :, :]
    #TODO this is where the output label is chosen
    data_label = dataset[:, -1:, :, 0:1]

    return data_input, data_label


def format_data(dataset):
    # dataset.shape = (n_data, n_nodes, n_features_in)
    if len(dataset.shape) == 2:
        dataset = np.expand_dims(dataset, 1)

    # produce (X, y) pairs of the data (sequences with 1 timestep each,
    # select 2 hours in the future as prediction variable)
    n_nodes = 40
    n_timesteps_per_day = 15
    n_timesteps_in = 4
    n_timesteps_out = 1
    n_features_in = 1

    data_seq = sequence_data(dataset, n_timesteps_per_day, n_timesteps_in, n_timesteps_out, n_nodes, n_features_in)
    # data_seq.shape = (n_seq, n_timesteps_seq = n_timesteps_in + n_timesteps_out, n_airports,n_features_in)

    delay_mean = np.mean(data_seq[:, :, 0])
    delay_median = np.median(data_seq[:, :, 0])
    delay_std = np.std(data_seq[:, :, 0])

    del_idx = []
    thres = delay_mean + 3*delay_std
    # print("Delay Threshold: ", thres, " minutes")
    for i in range(len(data_seq)):

        # remove sequences that have 0 flights in X or y
        if any(data_seq[i, :, :, 1] == 0):
            del_idx.append(i)

        # remove sequences with outlier delays in X or y
        # if any(data_seq[i, :, 0] > thres):
        if any(data_seq[i, :, :, 0] > thres):
            del_idx.append(i)
    data_seq = np.delete(data_seq, del_idx, axis=0)

    data_input, data_label = generate_input_label(data_seq, 1)
    # data_input.shape = (n_training_seq, n_timesteps_in, n_airports, n_features_in)
    # data_output.shape = (n_training_seq, n_timesteps_out=1, n_airports, n_features_out=1)
    data_input = data_input[:, :, :, 0]
    # data_label = np.expand_dims(data_label, 1)

    return data_input, data_label


###################################
# Models
###################################
def linear_regression(data_train, data_test):
    train_input, train_label = data_train
    test_input, test_label = data_test

    if len(train_input.shape) < 2:
        train_input = np.expand_dims(train_input, 1)
        train_label = np.expand_dims(train_label, 1)
        test_input = np.expand_dims(test_input, 1)
        test_label = np.expand_dims(test_label, 1)
    # returns the median absolute error for a linear model developed for a particular airport's data
    model = LinearRegression().fit(train_input, train_label)
    y_hat_train = model.predict(train_input)
    y_hat_test = model.predict(test_input)

    # med_ab_err = median_absolute_error(test_label, y_hat_test)
    mae = mean_absolute_error(test_label, y_hat_test)
    rmse = mean_squared_error(test_label, y_hat_test, squared=False)

    return mae, rmse


class MLP(torch.nn.Module):
    def __init__(self, c_in, c_hid, c_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(c_in, c_hid)
        self.fc2 = nn.Linear(c_hid, c_hid)
        self.fc3 = nn.Linear(c_hid, c_out)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # out = self.sigmoid(self.fc1(x))
        # out = self.sigmoid(self.fc2(out))
        # out = self.sigmoid(self.fc3(out))

        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))


        return out


def model_train(data_train, data_test, model, optimizer, criterion, n_epochs, device):
    # training
    train_input, train_label = data_train
    test_input, test_label = data_test

    if len(train_input.shape) < 2:
        train_input = np.expand_dims(train_input, 1)
        train_label = np.expand_dims(train_label, 1)
        test_input = np.expand_dims(test_input, 1)
        test_label = np.expand_dims(test_label, 1)

    # perm = torch.randperm(train_input.shape[0])

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # randomize and batch data
        # idx = perm[i:i+args.batch_size]
        # X, y = train_input[idx].to(args.device), train_label[idx].to(args.device)
        y_hat = model(train_input)
        y_hat = y_hat.cpu()
        loss = criterion(y_hat, train_label)
        loss.backward()
        optimizer.step()

    # validation
    if epoch == n_epochs-1:
        with torch.no_grad():
            model.eval()
            y_hat = model(test_input)
            y_hat = y_hat.cpu()

            # med_ab_err = median_absolute_error(test_label, y_hat)
            mae = mean_absolute_error(test_label, y_hat)
            rmse = mean_squared_error(test_label, y_hat, squared=False)

    return mae, rmse