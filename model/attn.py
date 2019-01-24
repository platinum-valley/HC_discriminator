import torch
import torch.nn as nn
import sys
from torch.autograd import Variable

class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        """
        self.attn = nn.Sequential(
                nn.Linear(self.hidden_dim*2, 1, bias=True),
                nn.Softmax()
        )
        """
        self.linear1 = nn.Linear(self.hidden_dim*2, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, hidden, encoder_output, device):
        concatenated_attn = torch.cat((hidden, encoder_output), dim=2).to(device)
        #attn_weights = self.attn(concatenated_attn)
        batch_size = concatenated_attn.size()[1]
        concatenated_attn = concatenated_attn.view(-1, concatenated_attn.size()[2])
        attn_weights = self.linear1(concatenated_attn)
        attn_weights = attn_weights.view(-1, batch_size, 1)
        attn_weights = torch.transpose(attn_weights, 0, 2)
        attn_weights = torch.transpose(attn_weights, 0, 1)
        attn_weights = self.softmax(attn_weights)
        #print(attn_weights.size())
        #print(attn_weights)
        encoder_output = torch.transpose(encoder_output, 0, 1)
        #attn_hidden = Variable(torch.bmm(attn_weights.type(torch.FloatTensor).to(device), encoder_output.type(torch.FloatTensor))).to(device)
        attn_hidden = torch.bmm(attn_weights, encoder_output)
        attn_hidden = Variable(torch.squeeze(attn_hidden)).to(device)
        #attn_hidden = torch.squeeze(attn_hidden)
        return attn_hidden
