import torch
from torch import nn
import numpy as np


class GRUCellV2(nn.Module):
    """
    Implements a GRU cell

    Args:
        input_size (int): The size of the input layer
        hidden_size (int): The size of the hidden layer
        activation (callable, optional): The activation function for a new gate. Defaults to torch.tanh.
    """

    def __init__(self, input_size, hidden_size, activation=torch.tanh):
        super(GRUCellV2, self).__init__()
        self.activation = activation

        # initialize weights by sampling from a uniform distribution between -K and K
        K = 1 / np.sqrt(hidden_size)
        # weights
        self.w_ih = nn.Parameter(torch.rand(3 * hidden_size, input_size) * 2 * K - K)
        self.w_hh = nn.Parameter(torch.rand(3 * hidden_size, hidden_size) * 2 * K - K)
        self.b_ih = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)
        self.b_hh = nn.Parameter(torch.rand(3 * hidden_size) * 2 * K - K)

    def forward(self, x, h):
        """
        Performs a forward pass through a GRU cell and returns the current hidden state h_t for every datapoint in batch.

        Args:
            x (torch.Tensor): an element x_t in a sequence
            h (torch.Tensor): previous hidden state h_{t-1}

        Returns:
            torch.Tensor: current hidden state h_t for every datapoint in batch.
        """
        #
        # YOUR CODE HERE
        #
        # reset update current
        reset_input, update_input, memory_input = torch.chunk(x @ self.w_ih.t() + self.b_ih, 3, dim=1)
        reset_hidden, update_hidden, memory_hidden = torch.chunk(h @ self.w_hh.t() + self.b_hh, 3, dim=1)

        z_reset = torch.sigmoid(reset_input + reset_hidden)
        z_update = torch.sigmoid(update_input + update_hidden)
        current_memory = self.activation(memory_input + z_reset * memory_hidden)

        final_memory_content = z_update * h + (1 - z_update) * current_memory

        return final_memory_content


class GRU2(nn.Module):
    """
    Implements a GRU network.

    Args:
        input_size (int): The size of the input layer
        hidden_size (int): The size of the hidden layer
        bias (bool, optional): Bias term. Defaults to True.
        activation (callable, optional): The activation function for a new gate. Defaults to torch.tanh.
        bidirectional (bool, optional): Whether the network is bidirectional. Defaults to False.
    """

    # FIXME: who left the bias flag unused?
    def __init__(self, input_size, hidden_size, bias=True, activation=torch.tanh, bidirectional=False):
        super(GRU2, self).__init__()
        self.bidirectional = bidirectional
        self.fw = GRUCellV2(input_size, hidden_size, activation=activation)  # forward cell
        if self.bidirectional:
            self.bw = GRUCellV2(input_size, hidden_size, activation=activation)  # backward cell
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Performs a forward pass through the whole GRU network, consisting of a number of GRU cells.

        Args:
            x (torch.Tensor): a batch of sequences of dimensionality (B, T, D)

        Returns:
            3-tuple (if bidirectional is True) or 2-tuple (otherwise) of torch.Tensor:
            - outputs: a tensor containing the output features h_t for each t in each sequence (the same as in PyTorch native GRU class).
                If bidirectional is True, then it should contain a concatenation of hidden states of forward and backward cells for each sequence element.
            - h_fw: the last hidden state of the forward cell for each sequence, i.e. when t = length of the sequence.
            - h_bw: the last hidden state of the backward cell for each sequence, i.e. when t = 0 (because the backward cell processes a sequence backwards).
        """
        #
        # YOUR CODE HERE
        #
        batches, time_steps, _ = x.shape

        outputs_fw = torch.zeros((batches, time_steps, self.hidden_size))

        h_fw = torch.zeros((batches, self.hidden_size))

        for t in range(time_steps):
            outputs_fw[:, t] = self.fw(x[:, t], h_fw)
            h_fw = outputs_fw[:, t]

        if not self.bidirectional:
            return outputs_fw, h_fw

        outputs_bw = torch.zeros((batches, time_steps, self.hidden_size))
        h_bw = torch.zeros_like(h_fw)

        for t in reversed(range(time_steps)):
            outputs_bw[:, t] = self.bw(x[:, t], h_bw)
            h_bw = outputs_bw[:, t]

        return torch.concat((outputs_fw, outputs_bw), dim=-1), h_fw, h_bw


def is_identical(a, b):
    return "Yes" if np.all(np.abs(a - b) < 1e-6) else "No"


if __name__ == '__main__':
    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = nn.GRU(10, 20, bidirectional=False, batch_first=True)
    outputs, h = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = GRU2(10, 20, bidirectional=False)
    outputs, h_fw = gru2(x)

    print("Checking the unidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru = GRU2(10, 20, bidirectional=True)
    outputs, h_fw, h_bw = gru(x)

    torch.manual_seed(100304343)
    x = torch.randn(5, 3, 10)
    gru2 = nn.GRU(10, 20, bidirectional=True, batch_first=True)
    outputs, h = gru2(x)

    print("Checking the bidirectional GRU implementation")
    print("Same hidden states of the forward cell?\t\t{}".format(
        is_identical(h[0].detach().numpy(), h_fw.detach().numpy())
    ))
    print("Same hidden states of the backward cell?\t{}".format(
        is_identical(h[1].detach().numpy(), h_bw.detach().numpy())
    ))
