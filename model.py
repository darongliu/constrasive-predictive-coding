import torch
import torch.nn as nn

from module import *

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

    def forward(self, feat, length):
        z = self.projection_layer(feat)
        for i in range(self.reduce_times):
            z = self.reduce_layers[i](z)
            length = (length/2).int()

        z_packed = nn.utils.rnn.pack_padded_sequence(input=z, lengths=length, batch_first=True)
        c_packed, _ = self.recurrent_layer(z_packed)
        c, _ = pad_packed_sequence(c_packed) # _ is length

        #TODO
        #mask
