import torch
import torch.nn as nn
from attn import AttnDecoder

class Decoder(nn.Module):


    def __init__(self, hidden_dim, output_dim, num_layer):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layer = num_layer

        self.rnn = nn.LSTMCell(self.hidden_dim, self.output_dim, self.num_layers=num_layer, dropout=0.1, bidirectional=True)

        self.fc = nn.Sequential(
                nn.Linear(output_dim, 2, bias=2),
                nn.Softmax()
                )

        self.attn = AttnDecoder(hidden_decoder)

    def forward(self, inputs):
        output, output_state = self.rnn(inputs)
        return (output, output_state)
