from collections import Counter
import os
import tqdm
import _pickle as pk

def inference(model, test_data_loader, result_dir, reduce_num):
    model.eval()

    if not os.isdir(result_dir):
        os.makedirs(result_dir)
    feature_dir = os.path.join(result_dir, 'feature')
    if not os.isdir(feature_dir):
        os.makedirs(feature_dir)
    all_phn_align = []

    count = 0
    for i, data in tqdm.tqdm(enumerate(test_data_loader)):
        feat, length, neg_shift, phn_boundary = data
        feat, length, neg_shift = put_to_cuda([feat, length, neg_shift])

        _, z = model(feat, length, neg_shift, train=False)
        z = z.cpu().numpy()
        for i in feat.size()[0]:
            phn_seq, transform_length = get_phn_seq(phn_boundary[i], length.cpu().numpy()[0], reduce_num)
            all_phn_align.append(phn_seq)

            encode_feat = z[i][:transform_length]
            pk.dump(encode_feat, open(os.path.join(feature_dir, str(count+1)+'.encode.feat'), 'wb'))
            count += 1

    pk.dump(all_phn_align, open(os.path.join(result_dir, 'phn_list'), 'wb'))

def get_phn_seq(phn_boundary, length, reduce_num=2):
    interval = 2**reduce_num
    transform_length = (length+interval-1)//interval

    counter = 0
    phn_idx = 0

    all_phn = []
    for i in range(transform_length):
        phn_temp = []
        for j in range(interval):
            if (phn_boundary[phn_idx][-1] <= counter) and \
               (phn_idx != len(phn_boundary)-1):
                phn_idx += 1
            phn_temp.append(phn_boundary[phn_idx][0])
            counter += 1
        all_phn.append(get_most_phn(phn_temp))

    return all_phn, transform_length

def get_most_phn(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

if __name__ == '__main__':
    phn_boundary = pk.load(open('../data/constrasive/processed_ls/phn_align.pkl', 'rb'))
    one_phn_boundary = phn_boundary[1]
    phn_list, l = get_phn_seq(one_phn_boundary, one_phn_boundary[-1][-1])
    print(len(phn_list), l)

