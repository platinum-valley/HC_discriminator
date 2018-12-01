import torch
import torch.nn

class decoder(nn.Module):


    def __init__(self, hidden_dim, output_dim, num_layer):
        self.rnn = nn.LSTM(hidden_dim, output_dim, num_layers=num_layer, dropout=0.1, bidirectional=True)

        self.fc = nn.Sequential(
                nn.Linear(output_dim, 2, bias=2),
                nn.Softmax()
                )

    def forward(self, inputs):
        outputs = self.rnn(inputs)
        return outputs
