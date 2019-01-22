import torch
import torch.nn as nn
from torch.autograd import Variable
from .encoder import Encoder
from .attn import Attention

class Recognizer(nn.Module):

    def __init__(self, args):
        super(Recognizer, self).__init__()
        self.encoder = Encoder(args.input_frame, args.input_dim, args.encoder_hidden_dim, args.num_encoder_layers)
        self.attn = Attention(args.output_dim)

        self.linear1 = nn.Linear(args.output_dim, args.label_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        """
        self.discriminator = nn.Sequential(
                nn.Linear(args.output_dim, args.label_dim, bias=True),
                nn.Sigmoid()
        )
        """
    def forward(self, inputs, init_encoder_state, sequence_length, device):
        encoder_outputs, encoder_state = self.encoder(inputs, init_encoder_state)
        encoder_last_output = encoder_outputs[-1].detach()
        #print(encoder_last_output)
        #encoder_last_output = torch.unsqueeze(encoder_last_output, 1)
        #eye = torch.eye(sequence_length)

        #encoder_attn = torch.tensor([encoder_last_output.tolist() for i in range(sequence_length)]).type(torch.cuda.DoubleTensor).to(device)
        encoder_last_output = torch.unsqueeze(encoder_last_output, 2)
        #print(encoder_last_output)
        mask = torch.ones((encoder_last_output.size()[0], 1, sequence_length)).type(torch.DoubleTensor).to(device)
        #print(mask)
        encoder_attn = torch.bmm(encoder_last_output, mask)
        encoder_attn = torch.transpose(encoder_attn, 0, 1)
        encoder_attn = torch.transpose(encoder_attn, 0, 2)
        #attn_output = Variable(self.attn(encoder_attn, encoder_outputs, device))
        attn_output = self.attn(encoder_attn, encoder_outputs, device)
        #pred = self.discriminator(attn_output)
        pred = self.linear1(attn_output)
        pred = self.sigmoid(pred)
        pred = torch.squeeze(pred)
        return pred
