gpu_id=0

mode='train' # or 'inference'

epochs=100
batch_size=32
learning_rate=0.001

prediction_num=12
neg_num=8
reduce_num=2

feat_dim=256

train_dir='/home/darong/darong/data/constrasive/processed_ls'
test_dir='/home/darong/darong/data/constrasive/processed_ls'
#train_dir='/home/kgb/qacnn_1d/data/movie_qa/train_part.json'
#test_dir='/home/kgb/qacnn_1d/data/movie_qa/dev_part.json'

#resume_dir='./model/cpc'
save_dir='./model/cpc'
result_dir='./new_data/'

CUDA_VISIBLE_DEVICES=$gpu_id python main.py $mode \
--batch_size $batch_size --epochs $epochs --learning_rate $learning_rate \
--prediction_num $prediction_num --neg_num $neg_num --reduce_num $reduce_num \
--feat_dim $feat_dim \
--train_dir $train_dir --test_dir $test_dir \
--save_dir $save_dir \
--result_dir $result_dir \

