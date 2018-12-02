import torch

def mask_with_length(tensor, length):
    '''
    mask the last two dimensin of the tensor with length
    tensor: rank>2 [... length x feat_dim]
    length: 1D array
    the first dimension of the tensor and length must be the same
    '''
    assert (len(tensor.size()) >= 2)
    assert (tensor.size()[0] == length.size()[0])

    tensor_length = tensor.size()[-2]
    rank = len(tensor.size())

    a = torch.arange(tensor_length).unsqueeze(-1).expand_as(tensor).int()
    b = length.view(-1, *([1]*(rank-1))).int()
    if tensor.is_cuda:
        a = a.cuda()
        b = b.cuda()

    mask = torch.ge(a, b) #mask: where to pad zero
    #if tensor.is_cuda:
        #mask = mask.cuda()
    tensor[mask] = 0.
    return tensor, (1.-mask).float()

def put_to_cuda(tensor_list):
    return [x.cuda() for x in tensor_list]

def print_and_logging(f, log):
    print(log)
    f.write(log)
    f.write('\n')

if __name__ == '__main__':
    feat = torch.ones(2, 3, 3, 2).cuda()
    length = torch.tensor([2,1]).cuda()
    a, mask = mask_with_length(feat, length)
    print(a[0])
    print(a[1])

    print(mask[0])

