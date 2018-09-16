import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN_2048_MODEL(nn.Module):
    def __init__(self, c_in_1, c_in_2, c_out_1, c_out_2):
        super(CNN_2048_MODEL, self).__init__()
        self.__c_in_1 = c_in_1
        self.__c_in_2 = c_in_2
        self.__c_out_1 = c_out_1
        self.__c_out_2 = c_out_2

        self.__expanded_size = (
            2 * 4 * c_out_2 * 2 + 3 * 3 * c_out_2 * 2 + 4 * 3 * c_out_1 * 2
        )

        self.__dense_value_1 = nn.Linear(self.__expanded_size, 256).double()
        self.__dense_value_2 = nn.Linear(256, 1).double()
        self.__dense_advantage_1 = nn.Linear(self.__expanded_size, 256).double()
        self.__dense_advantage_2 = nn.Linear(256, 4).double()

        self.__cnn_1 = nn.Conv2d(
            c_in_1,
            c_out_1,
            kernel_size=(1, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        ).double()

        self.__cnn_1_2 = nn.Conv2d(
            c_out_1,
            c_out_2,
            kernel_size=(1, 2),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        ).double()

        self.__cnn_2 = nn.Conv2d(
            c_in_2,
            c_out_2,
            kernel_size=(2, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        ).double()

        self.__cnn_2_2 = nn.Conv2d(
            c_out_1,
            c_out_2,
            kernel_size=(2, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
        ).double()

    def forward(self, features, batch_size, size_board):
        features_view = features.view(batch_size, 16, size_board, size_board)
        conv1_output = F.elu(self.__cnn_1(features_view))
        conv2_output = F.elu(self.__cnn_2(features_view))
        conv1_2_1_output = F.elu(self.__cnn_1_2(conv1_output))
        conv1_2_2_output = F.elu(self.__cnn_1_2(conv2_output))
        conv2_2_1_output = F.elu(self.__cnn_2_2(conv1_output))
        conv2_2_2_output = F.elu(self.__cnn_2_2(conv2_output))

        conv1_output_shape = list(conv1_output.shape)
        conv2_output_shape = list(conv2_output.shape)
        conv1_2_1_output_shape = list(conv1_2_1_output.shape)
        conv1_2_2_output_shape = list(conv1_2_2_output.shape)
        conv2_2_1_output_shape = list(conv2_2_1_output.shape)
        conv2_2_2_output_shape = list(conv2_2_2_output.shape)

        hidden1 = conv1_output.view(
            batch_size,
            (conv1_output_shape[1] * conv1_output_shape[2] * conv1_output_shape[3]),
        )

        hidden2 = conv2_output.view(
            batch_size,
            (conv2_output_shape[1] * conv2_output_shape[2] * conv2_output_shape[3]),
        )

        hidden1_2_1 = conv1_2_1_output.view(
            batch_size,
            (
                conv1_2_1_output_shape[1]
                * conv1_2_1_output_shape[2]
                * conv1_2_1_output_shape[3]
            ),
        )

        hidden1_2_2 = conv1_2_2_output.view(
            batch_size,
            (
                conv1_2_2_output_shape[1]
                * conv1_2_2_output_shape[2]
                * conv1_2_2_output_shape[3]
            ),
        )

        hidden2_2_1 = conv2_2_1_output.view(
            batch_size,
            (
                conv2_2_1_output_shape[1]
                * conv2_2_1_output_shape[2]
                * conv2_2_1_output_shape[3]
            ),
        )

        hidden2_2_2 = conv2_2_2_output.view(
            batch_size,
            (
                conv2_2_2_output_shape[1]
                * conv2_2_2_output_shape[2]
                * conv2_2_2_output_shape[3]
            ),
        )

        hidden = torch.cat(
            (hidden1, hidden2, hidden1_2_1, hidden1_2_2, hidden2_2_1, hidden2_2_2), 1
        )

        hidden_value_1 = F.elu(self.__dense_value_1(hidden))
        hidden_value_2 = self.__dense_value_2(hidden_value_1)

        advantage_action_1 = F.elu(self.__dense_advantage_1(hidden))
        advantage_action_2 = self.__dense_advantage_2(advantage_action_1)

        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        # print(advantage_action_2, hidden_value_2)
        reduced_mean = torch.mean(advantage_action_2, dim=1, keepdim=True)
        # print(reduced_mean)
        # reduced_array = advantage_action_2 - reduced_mean
        # print(reduced_array)
        Q = hidden_value_2 + (advantage_action_2 - reduced_mean)
        # print(Q)
