import torch
import torch.nn as nn

from module import *
from utils import *

class CPC(nn.Module):
    def __init__(self, input_dim, feat_dim, reduce_times=2, lstm_layers=2, prediction_num=12):
        super(CPC, self).__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.reduce_times = reduce_times
        self.lstm_layers = lstm_layers
        self.prediction_num = prediction_num

        self.projection_layer = nn.Linear(input_dim, feat_dim) #project the input feature to dimention feat_dim
        self.reduce_layers = [concat_nn(feat_dim) for _ in range(self.reduce_times)]
        self.recurrent_layer = nn.LSTM(input_size=self.feat_dim, hidden_size=self.feat_dim, num_layers=self.lstm_layers, batch_first=True)
        self.NCE_loss_layer = count_NCE_loss(self.feat_dim, self.feat_dim, prediction_num=self.prediction_num)

        self.layers = nn.ModuleList(self.reduce_layers)

    def forward(self, feat, length, neg_shift):
        # length: reverse order
        z = self.projection_layer(feat)
        for i in range(self.reduce_times):
            z = self.reduce_layers[i](z)
            length = (length/2).int()
            neg_shift = (neg_shift/2).int()

        z, _ = mask_with_length(z, length)
        z_packed = nn.utils.rnn.pack_padded_sequence(z, lengths=length, batch_first=True)
        c_packed, _ = self.recurrent_layer(z_packed)
        c, _ = nn.utils.rnn.pad_packed_sequence(c_packed, batch_first=True, total_length=z.size()[-2]) # _ is length
        return self.NCE_loss_layer(c, z, neg_shift, length)

if __name__ == '__main__':
    batch = 3
    len_ = 100
    input_dim = 4
    feat_dim = 6

    test_feat = torch.rand([batch, len_, input_dim], dtype=torch.float32).cuda()
    shift = torch.tensor([1,2,3]).cuda()
    length = torch.tensor([90, 80, 60]).cuda()
    m = CPC(input_dim, feat_dim)
    m.cuda()
    loss = m(test_feat, length, shift)
    print(loss)
    """
    test_feat = torch.rand([batch, len_, dim], dtype=torch.float32).cuda()
    print(test_feat)
    space = torch.tensor(1).cuda()
    m = shift(test_feat, space)
    m = m.cpu()
    print(m)
    """