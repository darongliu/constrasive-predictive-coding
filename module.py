import torch
import torch.nn as nn
import torch.nn.functional as F

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
            feat = torch.cat([feat, torch.zeros(feat_batch, 1, feat_dim)], dim=1)
            feat_len += 1
        feat = feat.view([feat_batch, int(feat_len/2), feat_dim*2])

        return F.relu(self.linear(feat))

class count_positive_prob(nn.Module):
    def __init__(self, c_dim, z_dim):
        #TODO


def shift(feat, shift):
    # do not shift vertical
    # shift hotizontal
    before = feat[:,:shift,:]
    after = feat[:,shift:,:]
    return torch.cat([after, before], dim=1)

if __name__ == '__main__':
    batch = 2
    len_ = 2
    dim = 4

    test_feat = torch.rand([batch, len_, dim], dtype=torch.float32).cuda()
    print(test_feat)
    space = torch.tensor(1).cuda()
    m = shift(test_feat, space)
    m = m.cpu()
    print(m)
