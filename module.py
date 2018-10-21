import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class concat_nn(nn.Module):
    '''
    concat two neighbor feature and pass through an nn
    '''
    def __init__(self, dim):
        super(concat_nn, self).__init__()
        self.linear = nn.Linear(2*dim , dim)

    def forward(self, feat): #feat: [batch, len, dim]
        feat_batch = feat.size()[0]
        feat_len = feat.size()[1]
        feat_dim = feat.size()[2]
        if feat_len%2 == 1:
            feat = feat[:, :-1, :]
            feat_len -= 1
        feat = feat.view([feat_batch, int(feat_len/2), feat_dim*2])

        return self.linear(F.relu(feat)) #relu -> linear

class gated_concat_nn(nn.Module):
    '''
    concat two neighbor feature and pass through an nn
    '''
    def __init__(self, dim):
        super(gated_concat_nn, self).__init__()
        self.linear1 = nn.Linear(2*dim , dim)
        self.linear2 = nn.Linear(2*dim , dim)

    def forward(self, feat): #feat: [batch, len, dim]
        feat_batch = feat.size()[0]
        feat_len = feat.size()[1]
        feat_dim = feat.size()[2]
        if feat_len%2 == 1:
            feat = feat[:, :-1, :]
            feat_len -= 1
        feat = feat.view([feat_batch, int(feat_len/2), feat_dim*2])

        return F.tanh(self.linear1(feat))*F.sigmoid(self.linear2(feat)) #relu -> linear

class count_NCE_loss(nn.Module):
    def __init__(self, c_dim, z_dim, prediction_num=12):
        super(count_NCE_loss, self).__init__()
        self.prediction_num = prediction_num
        self.bilinear_matrix = [nn.Parameter(torch.rand([c_dim, z_dim], requires_grad=True)) for _ in range(self.prediction_num)]
        self.params = nn.ParameterList(self.bilinear_matrix)
        #self.layers = nn.ModuleList(self.bilinear_matrix)

    def forward(self, c, z, neg_shift, length):
        '''
        neg_shift: 1d array with number in range[self.prediction_num, len]
        z: [batch x len x z_dim]
        c: [batch x len x c_dim]
        '''
        neg_num = neg_shift.size()[0]
        negative_z = [self.shift(z, neg_shift[i]) for i in range(neg_num)]

        total_loss = 0
        for i in range(self.prediction_num):
            positive_z = self.shift(z, i+1)
            all_z = torch.cat([positive_z, *negative_z], dim=0) #[batch*(self.negative+1) x len x z_dim]

            Wc = torch.matmul(c, self.bilinear_matrix[i]) #[batch x len x z_dim]
            Wc = Wc.repeat(neg_num+1, 1, 1)
            zWc = torch.sum((Wc*all_z), -1) #[batch*(self.negative+1) x len]
            zWc = zWc.view([-1, neg_num+1, zWc.size()[-1]])
            f = torch.exp(zWc) # [batch x (self.negative+1) x len]
            loss  = -torch.log(f[:,0,:]/torch.sum(f,1)) #[batch x len]
            loss, mask = mask_with_length(loss, length)

            total_loss += torch.sum(loss)/torch.sum(mask)

        return total_loss

    def shift(self, feat, shift):
        # do not shift vertical
        # shift hotizontal
        # shift left
        before = feat[:,:shift,:]
        after = feat[:,shift:,:]
        return torch.cat([after, before], dim=1)

if __name__ == '__main__':
    batch = 2
    len_ = 5
    dim_c = 4
    dim_z = 6

    test_feat_c = torch.rand([batch, len_, dim_c], dtype=torch.float32).cuda()
    test_feat_z = torch.rand([batch, len_, dim_z], dtype=torch.float32).cuda()
    shift = torch.tensor([1,2,3]).cuda()
    length = torch.tensor([3,4])
    m = count_NCE_loss(dim_c, dim_z, prediction_num=3)
    m.cuda()
    print(m(test_feat_c, test_feat_z, shift, length))
    """
    test_feat = torch.rand([batch, len_, dim], dtype=torch.float32).cuda()
    print(test_feat)
    space = torch.tensor(1).cuda()
    m = shift(test_feat, space)
    m = m.cpu()
    print(m)
    """
