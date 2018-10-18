from collections import Counter
import os
import tqdm
import _pickle as pk

def inference(model, test_data_loader, result_dir, reduce_num):
    model.eval()

    feature_dir = os.path.join(result_dir, 'feature')
    os.mkdir(feature_dir)
    all_phn_align = []

    count = 0
    for i, data in tqdm.tqdm(enumerate(train_data_loader)):
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

def get_phn_seq(phn_boundary, length, reduce_num=2):
    interval = 2**reduce_num
    transform_length = (length+interval-1)//interval

    counter = 0
    phn_idx = 0

    all_phn = []
    for i in range(transform_length):
        for j in range(interval):
            if (phn_boundary[phn_idx][-1] <= counter) and \
               (phn_idx != len(phn_boundary)-1):
                phn_idx += 1
            all_phn.append(phn_boundary[phn_idx][0])
            counter += 1

    return all_phn, transform_length

def get_most_phn(phn_list):
    data = Counter(lst)
    return data.most_common(1)[0][0]



