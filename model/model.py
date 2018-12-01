import torch
import torch.nn
from encoder import Encoder
from decoder import Decoder

class Model(nn.Module):

    def __init__(self, args):
        self.encoder = Encoder(args.input_frame ,args.input_dim, args.hidden_dim, args.num_input_layers)
        self.decoder = Decoder(args.hidden_dim, args.output_dim, args.num_hidden_layers)

        self.model = nn.Sequential(
                self.encoder,
                self.decoder
                )

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
