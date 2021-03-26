############################
# imports
############################
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
############################
# STGCN Building Blocks
############################


class TemporalConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kt=3):
        """
        Args:
            c_in (int): number of input channels per node
            c_out (int): number of output channels per node
            kt (int, optional): size of temporal kernel (1D convolution). Defaults to 3.
        """
        super(TemporalConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        # for input size NCWH, kernel acts on the WH channels
        self.conv1 = nn.Conv2d(c_in, 2 * c_out, kernel_size=(kt, 1), padding = (1, 0))
        self.conv_reshape = nn.Conv2d(c_in, c_out, kernel_size=(kt, 1), padding = (1, 0))

    def forward(self, X):
        """
        Args:
            X (Torch.Tensor): X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
        Returns:
            [Torch.Tensor]: out shape  = (batch_size, c_out, n_timesteps_in, n_nodes)
        """
        x_residual = X
        # change size of channel dimension of input so it works as part of GLU

        if self.c_in < self.c_out:
            n_pad = round((self.c_out - x_residual.shape[1])/2)
            # odd number of channels
            if (x_residual.shape[1] % 2 != 0):
                pad = nn.ZeroPad2d((0, 0, n_pad, n_pad-1))
            else:
                pad = nn.ZeroPad2d((0, 0, n_pad, n_pad))
            x_residual = x_residual.permute(0, 2, 1, 3).contiguous()
            x_residual = pad(x_residual)
            x_residual = x_residual.permute(0, 2, 1, 3).contiguous()
            # x_residual shape = (batch_size, c_out, n_timesteps_in, n_nodes)

        elif self.c_in > self.c_out:
            x_residual = self.conv_reshape(x_residual)

        x_conv = self.conv1(X)
        # x_conv shape = (batch_size, 2*c_out, n_timesteps_in, n_nodes)

        # cut along the "channels" dimension to form P, Q for GLU activation
        P = x_conv[:, 0:self.c_out, :, :]
        Q = x_conv[:, -self.c_out:, :, :]

        out = (P + x_residual) * torch.sigmoid(Q)
        return out


class SpatialConvLayer(nn.Module):
    def __init__(self, c_in, c_out, device, ks=5):
        """
        Args:
            c_in (int): number of input channels per node
            c_out ([type]): number of output channels per node
            ks (int, optional): size of spatial kernel. Defaults to 5.
        """

        super(SpatialConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.ks = ks

        self.theta = nn.Parameter(torch.FloatTensor(ks * c_in, c_out)).double().to(device)
        self.conv_reshape = nn.Conv2d(c_in, c_out, kernel_size=(ks, 1), padding = (1, 0))
        self.init_parameters()

    def init_parameters(self):
        # kaiming uniform initialization
        std = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-std, std)


    def graph_conv(self, X, theta, graph_kernel, ks, c_in, c_out):
        """performs graph convolution operation

        Args:
            X (Torch.Tensor): input data, X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
            theta (Torch.Tensor): learnable parameters, theta shape  = (ks * c_in, c_out)
            graph_kernel (Torch.Tensor): Chebyshev or 1st order approximation of scaled graph Laplacien, graph_kernel shape  = (n_nodes, ks * n_nodes)
            ks (int): size of spatial kernel
            c_in (int): number of input channels per node
            c_out (int): number of output channels per node

        Returns:
            [Torch.Tensor]: x_gconv shape  = [batch_size * n_timesteps_in, n_nodes, c_out]
        """
        batch_size, n_timesteps_in = X.shape[0], X.shape[2]
        n_nodes = graph_kernel.shape[0]

        # reshape 1
        x_conv = X.reshape(-1, n_nodes)
        # x_conv shape  = [batch_size * n_timesteps_in * c_in, n_nodes]

        # spatial convolution
        x_conv = torch.spmm(x_conv, graph_kernel)
        # x_conv  shape  = [batch_size * n_timesteps_in * c_in, ks * n_nodes]

        # reshape 2
        x_conv = x_conv.reshape(-1, c_in * ks)
        # x_conv shape  = [batch_size * n_timesteps_in * n_nodes, c_in * ks]

        # multiply by learned params
        x_conv = torch.spmm(x_conv, theta)
        # x_conv shape  = [batch_size * n_timesteps_in * n_nodes, c_out]

        # reshape 3
        x_conv = x_conv.reshape(-1, self.c_out, n_timesteps_in, n_nodes)
        # x_conv shape  = (batch_size, n_timesteps_in, n_nodes, c_out)

        return x_conv


    def forward(self, X, graph_kernel):
        """
        Args:
            X (Torch.Tensor): input data, X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
            graph_kernel (Torch.Tensor): Chebyshev or 1st order approximation of scaled graph Laplacien, graph_kernel shape  = (n_nodes, ks * n_nodes)
        Returns:
            [Torch.Tensor]: out shape  = (batch_size, n_timesteps_in, n_nodes, c_out)
        """
        batch_size, n_timesteps_in, n_nodes = X.shape[0], X.shape[2], X.shape[3]

        x_residual = X
        x_conv = X


        # change size of channel dimension of input so it works as part of GLU
        if self.c_in < self.c_out:
            n_pad = round((self.c_out - x_residual.shape[1])/2)
            # odd number of channels
            if (x_residual.shape[1] % 2 != 0):
                pad = nn.ZeroPad2d((0, 0, n_pad, n_pad-1))
            else:
                pad = nn.ZeroPad2d((0, 0, n_pad, n_pad))

            x_residual = x_residual.permute(0, 2, 1, 3).contiguous()
            x_residual = pad(x_residual)
            x_residual = x_residual.permute(0, 2, 1, 3).contiguous()
            # x_residual shape = (batch_size, c_out, n_timesteps_in, n_nodes)

        elif self.c_in > self.c_out:
            x_residual = self.conv_reshape(x_residual)

        x_conv = self.graph_conv(X, self.theta, graph_kernel, self.ks, self.c_in, self.c_out)
        # x_conv shape  = (batch_size * n_timesteps_in, n_nodes, c_out)

        # add outputs, relu
        out = F.relu(x_conv + x_residual)

        return out


class OutputLayer(nn.Module):
    def __init__(self, channels, n_timesteps_in, n_timesteps_out, kt=3):
        """
        Args:
            channels (array): channel size for output, channels[0] must be equal to the channel size of input to this layer, channels[1] is number of output features for network
            n_timesteps_in (int): number of timesteps in the input data
            n_timesteps_out (int): number of timesteps in the labeled data
            kt (int, optional): size of temporal kernel. Defaults to 3.
        """
        super(OutputLayer, self).__init__()
        c_in, c_out = channels
        self.temporal_conv_layer = TemporalConvLayer(n_timesteps_in, n_timesteps_out, kt=kt).double()
        self.fc = nn.Conv2d(c_in, c_out, kernel_size = (1, 1)).double()

    def forward(self, X):
        """
        Args:
            X (Torch.Tensor): X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
        Returns:
            Torch.Tensor: out shape  = (batch_size, c_out, n_timesteps_in, n_nodes)
        """
        # reduce the time dimension to be same as n_timesteps_out
        X = X.permute(0, 2, 1, 3).contiguous()
        out = self.temporal_conv_layer(X)
        out = out.permute(0, 2, 1, 3).contiguous()
        # out shape  = (batch_size, n_channels, n_timesteps_out, n_nodes)

        # reduce to 1 output feature per graph node
        out = self.fc(out)
        # out shape  = (batch_size, n_features_out, n_timesteps_out, n_nodes)
        return out


############################
# ST-Conv Block
############################


class STConvBlock(nn.Module):
    def __init__(self, channels, n_nodes, device, ks=5, kt=3, drop_prob=0.0):
        super(STConvBlock, self).__init__()
        c_in, c_hid, c_out = channels
        self.temporal_layer_1 = TemporalConvLayer(c_in, c_hid, kt=kt).double()
        self.spatial_layer = SpatialConvLayer(c_hid, c_hid, device, ks=ks).double()
        self.temporal_layer_2 = TemporalConvLayer(c_hid, c_out, kt=kt).double()

        # layer norm is batch norm along a different dimension
        # https://nealjean.com/assets/blog/group-norm.png
        self.layer_norm = nn.BatchNorm2d(n_nodes).double()
        self.dropout = nn.Dropout2d(drop_prob)


    def forward(self, X, graph_kernel):
        """
        Args:
            X (Torch.Tensor): input data, X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
            graph_kernel (Torch.Tensor): Chebyshev or 1st order approximation of scaled graph Laplacien, graph_kernel shape  = (n_nodes, ks * n_nodes)

        Returns:
            Torch.Tensor: out shape  = (batch_size, c_out, n_timesteps_in, n_nodes)
        """
        out = self.temporal_layer_1(X)
        out = self.spatial_layer(out, graph_kernel)
        out = self.temporal_layer_2(out)
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = out.permute(0, 3, 2, 1).contiguous()

        return out


############################
# STGCN Model
############################


class STGCN(nn.Module):
    def __init__(self, blocks, n_timesteps_in, n_timesteps_out, n_nodes, device, ks=5, kt=3, drop_prob=0.0):
        """STGCN Model as implemented in https://arxiv.org/abs/1709.04875
        Original Tensorflow code may be found here: https://github.com/VeritasYin/STGCN_IJCAI-18

        Args:
            blocks (array): array of arrays giving channel sizes for each STConvBlock,
            the original paper used [[n_features_in, 32, 64], [64, 32, 128], [128, n_features_out]]
            n_timesteps_in (int): number of timesteps in the input data
            n_timesteps_out (int): number of timesteps in the labelled data
            ks (int, optional): size of spatial kernel. Defaults to 5.
            kt (int, optional): size of temporal kernel. Defaults to 3.
        """
        super(STGCN, self).__init__()

        self.st_conv_block_1 = STConvBlock(blocks[0], n_nodes, device, ks, kt, drop_prob=drop_prob)
        self.st_conv_block_2 = STConvBlock(blocks[1], n_nodes, device, ks, kt, drop_prob=drop_prob)
        # use the "individual" inf mode from the original model
        self.output_layer = OutputLayer(blocks[2], n_timesteps_in, 1)


    def forward(self, X, graph_kernel):
        """
        Args:
            X (Torch.Tensor): input data, X shape  = (batch_size, c_in, n_timesteps_in, n_nodes)
            graph_kernel (Torch.Tensor): Chebyshev or 1st order approximation of scaled graph Laplacien, graph_kernel shape  = (n_nodes, ks * n_nodes)
        Returns:
        Torch.Tensor: model prediction, shape  = (batch_size, n_features_out, 1, n_nodes)
        use the "individual" inf mode from the original model
        """
        out = self.st_conv_block_1(X, graph_kernel)
        out = self.st_conv_block_2(out, graph_kernel)
        y_hat = self.output_layer(out)

        return y_hat

