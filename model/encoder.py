import torch
import torch.nn as nn

class Encoder(nn.Module):


    def __init__(self, frames, input_dim, hidden_dim, num_layer):
        super(Encoder, self).__init__()
        """
        self.conv = nn.Sequential(
                nn.Conv2d(frames, int(frames/2), kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(int(frames/2), int(frames/4), kernel_size=3, stride=2, padding=1),
                nn.ReLU())
        """
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layer, dropout=0.0, bidirectional=False)
        """
        self.encoder = nn.Sequential(
                self.conv,
                self.rnn
                )
        """
        self.linear = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, inputs, init_state):
        #conv_outputs = self.conv(inputs)
        #outputs = self.rnn(conv_outputs, init_state)
        #outputs = self.rnn(inputs, init_state)
        outputs = self.rnn(inputs)
        return outputs
