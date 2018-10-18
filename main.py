import argparse
import torch
import sys
import os

from model import CPC
from data_loader import myDataset
from saver import pytorch_saver

from train import train
from inference import inference

parser = argparse.ArgumentParser(description='pytorch constrasive predictive coding')
parser.add_argument('pos1', default='train', type=str,
                    help='train or inference (default: train)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 50)')
parser.add_argument('--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')

parser.add_argument('--prediction_num', default=12, type=int,
                    help='the length of question (default: 12)')
parser.add_argument('--neg_num', default=12, type=int,
                    help='the length of option (default: 8)')
parser.add_argument('--reduce_num', default=2, type=int,
                    help='the reduce number in the encoder (default: 2)')

parser.add_argument('--input_dim', default=39, type=int,
                    help='the original dimension of the feature')
parser.add_argument('--feat_dim', default=256, type=int,
                    help='the project dimention of the feature')

parser.add_argument('--train_dir', default='', type=str, metavar='PATH',
                    help='The dir of the training (default: none)')
parser.add_argument('--test_dir', default='', type=str, metavar='PATH',
                    help='The path of the testing (default: none)')
parser.add_argument('--resume_dir', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir',
                    help='The directory used to save the trained models',
                    default='model/', type=str)
parser.add_argument('--log',help='training log file',
                    default='./log', type=str)
parser.add_argument('--result_dir', dest='result_dir',
                    help='The output path of the inference result',
                    default='', type=str)

def main(args):
    if args.pos1 == 'train':
        train_dataset = myDataset(os.path.join(args.train_dir, 'feature'))
        #prepare dataloader
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=myDataset.get_collate_fn(args.prediction_num, args.neg_num, args.reduce_num))
        saver = pytorch_saver(10, args.save_dir)
        #build model
        model = CPC(args.input_dim, args.feat_dim)
        if args.resume_dir != '':
            model.load_state_dict(pytorch_saver.load_dir(args.resume_dir)['state_dict'])

        model.train()
        model.cuda()
        args.log = os.path.join(args.save_dir, args.log)
        train(model, train_data_loader, saver, args.epochs, args.learning_rate, args.log)

    else:
        test_dataset = myDataset(os.path.join(args.train_dir, 'feature'), os.path.join(args.train_dir, 'phn_align.pkl'))
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        if args.resume_dir == '':
            print("resume should exist in inference mode", file=sys.stderr)
            sys.exit(-1)
        else:
            model = CPC(args.input_dim, args.feat_dim)
            model.load_state_dict(pytorch_saver.load_dir(args.resume_dir)['state_dict'])
            model.eval()
            model.cuda()

            inference(model, test_data_loader, args.result_dir, args.reduce_num)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


#filter need deeper
#add dropout
