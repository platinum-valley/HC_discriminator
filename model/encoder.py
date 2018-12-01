import torch
import torch.nn

class Encoder(nn.Module):


    def __init__(self, frames ,input_dim, hidden_dim, num_layer):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(frames, int(frames/2), kernel_size=3, stride=2, padding=1),
                nn.ReLU()
                nn.Conv2d(int(frames/2), int(frames/4), kernel_size=3, stride=2, padding=1),
                nn.ReLU())
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layer, dropout=0.1, bidirectional=True)

        self.encoder = nn.Sequence(
                self.conv,
                self.rnn
                )

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        return outputs
