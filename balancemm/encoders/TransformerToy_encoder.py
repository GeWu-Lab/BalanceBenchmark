import torch.nn as nn
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(Transformer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力层
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class TransformerEncoder(nn.Module):
    """
    Extend to nn.Transformer.
    """
    def __init__(self, input_dim = 300, n_features = 512,dim = 1024,n_head = 4,n_layers = 2):
        super(TransformerEncoder,self).__init__()
        self.embedding = nn.Linear(input_dim, n_features)
        self.embed_dim = dim
        self.conv = nn.Conv1d(n_features,self.embed_dim,kernel_size=1,padding=0,bias=False)
        layer = Transformer(self.embed_dim,num_heads=n_head, dim_feedforward = n_features)
        self.transformer = nn.TransformerEncoder(layer,num_layers=n_layers)


    def forward(self,x):
        """
        Apply transorformer to input tensor.

        """
        if type(x) is list:
            x = x[0]
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        x = self.conv(x.permute([0,2,1]))
        x = x.permute([2,0,1])
        x = self.transformer(x)[0]
        # print(x.shape)
        return x