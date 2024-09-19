import torch
import torch.nn as nn


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output

class ConcatFusion_N(nn.Module):
    def __init__(self, input_dim=3072, output_dim=100):
        super(ConcatFusion_N, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
    def forward(self, encoder_res):
        output = torch.cat(list(encoder_res.values()),dim = 1)
        output = self.fc_out(output)
        return output

class ConcatFusion_3(nn.Module):
    def __init__(self, input_dim=3072, output_dim=100):
        super(ConcatFusion_3, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim

    def forward(self, x, y, z):
        output = torch.cat((x, y), dim=1)
        output = torch.cat((output, z),dim = 1)
        output = self.fc_out(output)
        # x = (torch.mm(x, torch.transpose(self.fc_out.weight[:, self.input_dim // 3: 2 * self.input_dim // 3], 0, 1))
        #              + self.fc_out.bias / 2)

        # y = (torch.mm(y, torch.transpose(self.fc_out.weight[:, 2* self.input_dim // 3: self.input_dim ], 0, 1))
        #             + self.fc_out.bias / 2)
        
        return x, y, z, output
    
class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output

