import torch
import math
import torch.nn as nn
import torch.nn.functional as F
class Config():
    def __init__(self,input_dim, layer, hidden_size, dropout_r, multi_head, ff_size, seq_len):
        self.input_dim = input_dim
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout_r = dropout_r
        self.multi_head = multi_head
        self.ff_size = ff_size
        self.seq_len = seq_len

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

class FC(nn.Module):
    def __init__(self,in_size,out_size,dropout_r = 0.,use_relu=True) -> None:
        super(FC,self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size,out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self,x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)
        
        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self,size,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2 *(x-mean) /(std+ self.eps) + self.b_2

class MLP(nn.Module):
    def __init__(self,in_size,mid_size,out_size,dropout_r = 0.,use_relu = True):
        super(MLP,self).__init__()

        self.fc = FC(in_size,mid_size,dropout_r=dropout_r,use_relu = use_relu)
        self.linear = nn.Linear(mid_size,out_size)
    def forward(self,x):
        return self.linear(self.fc(x))

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class AttFlat(nn.Module):
    def __init__(self, args, flat_glimpse, merge=False):
        super(AttFlat, self).__init__()
        self.args = args
        self.merge = merge
        self.flat_glimpse = flat_glimpse
        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=flat_glimpse,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        if self.merge:
            self.linear_merge = nn.Linear(
                args.hidden_size * flat_glimpse,
                args.hidden_size * 2
            )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        if self.merge:
            x_atted = torch.cat(att_list, dim=1)
            x_atted = self.linear_merge(x_atted)

            return x_atted

        return torch.stack(att_list).transpose_(0, 1)

class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class SGA(nn.Module):
    def __init__(self, args):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(args)
        self.mhatt2 = MHAtt(args)
        self.ffn = FFN(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = LayerNorm(args.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class Block(nn.Module):
    def __init__(self, args, i):
        super(Block, self).__init__()
        self.args = args
        self.sa1 = SA(args)
        self.sa3 = SGA(args)

        self.last = (i == args.layer-1)
        if not self.last:
            self.att = AttFlat(args, args.seq_len, merge=False)
            self.norm = LayerNorm(args.hidden_size)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask):

        ax = self.sa1(x, x_mask)
        x = ax + x
        if self.last:
            return x
        ax = self.att(x, x_mask)

        return self.norm(x + self.dropout(ax))


class Transformer_LAEncoder(nn.Module):
    def __init__(self, input_dim, layer, hidden_size, dropout_r, multi_head, ff_size, seq_len, modality):
        super(Transformer_LAEncoder,self).__init__()
        args = Config(input_dim, layer, hidden_size, dropout_r, multi_head, ff_size, seq_len)
        self.modality = modality
        self.enc_list = nn.ModuleList([Block(args, i) for i in range(layer)])
        # self.enc_list = nn.ModuleList([nn.Transformer(d_model=hidden_size)])
        self.adapter = nn.Linear(input_dim, hidden_size)
        if self.modality == "text":
            self.lstm = nn.LSTM(
                input_size = input_dim,
                hidden_size = hidden_size,
                num_layers=1,
                batch_first=True
            )
        self.attflat = AttFlat(args, 1, merge=True)
    
    def forward(self,x):
        x_mask = make_mask(x)
        if self.modality == "text":
            x, _ = self.lstm_x(x)
        else:
            x = self.adapter(x)

        for i, dec in enumerate(self.enc_list):
            x_m= None
            if i == 0:
                x_m= x_mask
            x = dec(x, x_m)

        x = self.attflat(
            x,
            None
        )
        return x
    