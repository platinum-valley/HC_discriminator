import torch
import torch.nn
from encoder import Encoder
from decoder import Decoder

class Recognizer(nn.Module):

    def __init__(self, args):
        self.encoder = Encoder(args.input_frame ,args.input_dim, args.encoder_hidden_dim, args.num_encoder_layers)
        self.decoder = Decoder(args.encoder_hidden_dim, args.output_dim, args.num_decoder_layers)

        self.model = nn.Sequential(
                self.encoder,
                self.decoder
                )

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
