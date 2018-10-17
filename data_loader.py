import torch
import torch.utils.data
import numpy as np
import os
import _pickle as pk
import random

class myDataset(torch.utils.data.Dataset):
    def __init__(self, feat_dir):
        super(myDataset).__init__()
        self.feat_dir = feat_dir
        print('finish loading data')

    def __len__(self):
        return len(os.listdir(self.feat_dir))

    def __getitem__(self, idx):
        feat = pk.load(open(os.path.join(self.feat_dir, str(idx+1)+'.mfcc.pkl'),'rb'))
        return feat, len(feat)

    @staticmethod
    def get_collate_fn(prediction_num, neg_num, reduce_times):

        def _pad_sequence(np_list, length=None):
            tensor_list = [torch.from_numpy(np_array) for np_array in np_list]
            #print('tensor length: ', len(tensor_list))
            #for tensor in tensor_list:
            #    print('shape', tensor.size())
            pad_tensor = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
            if length is None:
                return pad_tensor
            else:
                pad_length = pad_tensor.size()[1]
                if pad_length >= length:
                    return pad_tensor[:, :length, :]
                else:
                    pad = torch.zeros([pad_tensor.size()[0], length-pad_length, pad_tensor.size()[2]])
                    return torch.cat([pad_tensor, pad], 1)

        def collate_fn(batch):
            # for dataloader
            all_feat, all_length = zip(*batch)
            all_feat = _pad_sequence(all_feat)
            all_length = torch.tensor(all_length)

            tensor_length = all_feat.size()[1]
            neg_shift = random.sample(range((prediction_num+1)*reduce_times, tensor_length), neg_num)
            neg_shift = torch.tensor(neg_shift)

            return all_feat, all_length, neg_shift

        return collate_fn

if __name__ == '__main__':
    path = '/home/darong/darong/data/constrasive/processed_ls/feature'
    prediction_num=3
    neg_num=5
    reduce_times=2

    dataset = myDataset(path)
    collate_fn = myDataset.get_collate_fn(prediction_num, neg_num, reduce_times)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for i, data in enumerate(data_loader):
        print('i', i)
        print('feat shape:', data[0].shape)
        print('len shape:', data[1])
        print('neg shape:', data[2])
        if i == 2:
            break
