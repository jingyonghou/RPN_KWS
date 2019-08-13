#!/usr/bin/env python

# Copyrigh 2018 houjingyong@gmail.com

# MIT Licence

from __future__ import print_function

import os, sys, argparse, datetime, shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from bbox_transform import get_out_utt_boxes
from config import cfg
from config import cfg_from_file
from streaming_special_torch_dataset import * 
from kaldi_io import *
from RNNs import GRU
from RPN import RPN
from RPN_KWS import RPN_KWS
from utils import AverageMeter, count_parameters
from loss import loss_frame_fn_ce, acc_frame

def get_args():
    """Get arguments from stdin."""
    parser = argparse.ArgumentParser(description='Pytorch acoustic model.')
    parser.add_argument('--encoder', type=str, default='gru',
                        help='encoder type {default: gru}')
    parser.add_argument('--num-anchor', type=int, default=10, metavar='HF',
                        help='Num anchors per frame {default: 10.0}')
    parser.add_argument('--lambda-factor', type=float, default=5.0, metavar='HF',
                        help='Balance factor between classification and regression loss (default: 5.0).')
    parser.add_argument('--input-dim', type=int, default=40, metavar='N',
                        help='Input feature dimension without context (default: 40).')
    parser.add_argument('--kernel-size', type=int, default=3, metavar='N',
                        help='Kernel size of Wavenet or CNN (default:3).')
    parser.add_argument('--hidden-dim', type=int, default=128, metavar='N',
                        help='Hidden dimension of feature extractor (default: 128).')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                        help='Numbers of hidden layers of feature extractor (default: 2).')
    parser.add_argument('--output-dim', type=int, default=2000, metavar='N',
                        help='Output dimension, number of classes (default: 2000).')
    parser.add_argument('--dropout', type=float, default=0.0001, metavar='DR',
                        help='dropout of feature extractor (default: 0.0001).')
    parser.add_argument('--left-context', type=int, default=5, metavar='N',
                        help='Left context length for splicing feature (default: 5).')
    parser.add_argument('--right-context', type=int, default=5, metavar='N',
                        help='Right context length for splicing feature (default: 5).')
    parser.add_argument('--max-epochs', type=int, default=20, metavar='N',
                        help='Maximum epochs to train (default: 20).')
    parser.add_argument('--min-epochs', type=int, default=0, metavar='N',
                        help='Minimum epochs to train (default: 0).')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='Batch size for training (default: 8).')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
                        help='Initial learning rate (default: 0.001).')
    parser.add_argument('--halving-factor', type=float, default=0.5, metavar='HF',
                        help='Half factor for learning rate (default: 0.5).')
    parser.add_argument('--start-halving-impr', type=float, default=0.01, metavar='S',
                        help='Improvement threshold to half the learning rate (default: 0.01).')
    parser.add_argument('--end-halving-impr', type=float, default=0.001, metavar='E',
                        help='Improvement threshold to stop half learning rate (default: 0.001).')
    parser.add_argument('--init-weight-decay', type=float, default=1e-5, metavar='E',
                                    help='Weight decay of L2 normalization (default: 1e-5).')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='Random seed (default: 1234).')
    parser.add_argument('--use-cuda', type=int, default=1, metavar='C',
                        help='Use cuda (1) or cpu(0).')
    parser.add_argument('--multi-gpu', type=int, default=0, metavar='G',
                        help='Use multi gpu (1) or not (0).')
    parser.add_argument('--train', type=int, default=1,
                        help='Executing mode, train (1) or test (0).')
    parser.add_argument('--train-scp', type=str, default='',
                        help='Training data file.')
    parser.add_argument('--dev-scp', type=str, default='',
                        help='Development data file.')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Directory to output the model.')
    parser.add_argument('--load-model', type=str, default='',
                        help='Previous model to load.')
    parser.add_argument('--test', type=int, default=0,
                        help='Executing mode, 1 for test, 0 no test')
    parser.add_argument('--test-scp', type=str, default='',
                        help='Test data file.')
    parser.add_argument('--output-file', type=str, default='',
                        help='Test output file')
    parser.add_argument('--region-output-file', type=str, default='',
                        help='Region output file')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='How many batches to wait before logging training status.')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                        help='How many workers used to load data')
    parser.add_argument('--config-file', type=str, default='',
                        help='config file in yaml format')
    args = parser.parse_args()

    if args.config_file != '':
        cfg_from_file(args.config_file)
    
    return args

def get_new_target(device, target, num_p, num_n):
    new_target=[]
    for i in range(target.size(0)):
        if target[i][0] == 0:
            new_target += ([target[i][0]] * num_n)
        else:
            new_target += ([target[i][0]] * num_p)
    return torch.LongTensor(new_target).to(device)

def adjust_learning_rate(args, optimizer):
    """Half the learning rate when relative improvement is too low.
    Args:
        args: Arguments for training.
        optimizer: Optimizer for training.
    """
    args.learning_rate *= args.halving_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate

def train(args, model, device, train_loader, optimizer, epoch):
    """Train one epoch."""
    tr_rpn_loss_bbox = AverageMeter()
    tr_rpn_loss_cls = AverageMeter()
    tr_loss = AverageMeter()
    tr_rpn_acc = AverageMeter()
    model.train()
    total_step = len(train_loader)
    balance_weight=args.lambda_factor
    for batch_idx, (utt_id, act_lens, data, target) in enumerate(train_loader):
        act_lens, data, target = act_lens.to(device), data.to(device), target.to(device)
        target = target.reshape(target.size(0), 1, target.size(1)).float()
        # Forward pass
        batch_size = data.shape[0]
        outputs = model(epoch, data, act_lens, target, 100)
        rois, rpn_cls_score, rpn_label, rpn_loss_cls, rpn_loss_bbox = outputs
        rpn_acc = acc_frame(rpn_cls_score, rpn_label)
        
        # Backward and optimize
        loss = rpn_loss_cls + balance_weight * rpn_loss_bbox
        optimizer.zero_grad()
        loss.backward()
        #name, param=list(model.named_parameters())[1]
        #print('Epoch:[{}/{}], param name:{},\n param:'.format(epoch+1, args.max_epochs, name, param))
        optimizer.step()

        tr_rpn_acc.update(rpn_acc, 1)
        tr_loss.update(loss, 1)
        tr_rpn_loss_cls.update(rpn_loss_cls, 1)
        tr_rpn_loss_bbox.update(rpn_loss_bbox, 1)

        if batch_idx % args.log_interval == 0:
            print('Epoch: [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train RPN Acc: {:.4f}%'
                  .format(epoch+1, args.max_epochs, batch_idx+1, total_step, tr_loss.cur,  tr_rpn_acc.cur))
            print('Epoch: [{}/{}], Step [{}/{}], Train RPN cls Loss: {:.4f}, Train RPN bbox Loss: {:.4f} '
                  .format(epoch+1, args.max_epochs, batch_idx+1, total_step, tr_rpn_loss_cls.cur, tr_rpn_loss_bbox.cur))

    print('Epoch: [{}/{}], Average Train Loss: {:.4f}, Average Train RPN cls Loss: {:.4f}, Average Train RPN bbox Loss: {:.4f}, AverageAverage Train RPN Acc: {:.4f}%'
         .format(epoch+1, args.max_epochs, tr_loss.avg, tr_rpn_loss_cls.avg, tr_rpn_loss_bbox.avg, tr_rpn_acc.avg))
    return float("{:.4f}".format(tr_loss.avg))


def validate(args, model, device, dev_loader, epoch):
    """Cross validate the model."""
    meter_rpn_loss_bbox = AverageMeter()
    balance_weight = args.lambda_factor
    meter_rpn_loss_cls = AverageMeter()
    meter_loss = AverageMeter()
    meter_rpn_acc = AverageMeter()
    balance_weight = args.lambda_factor
    with torch.no_grad():
        total_step = len(dev_loader)
        for batch_idx, (utt_id, act_lens, data, target) in enumerate(dev_loader):
            act_lens, data, target = act_lens.to(device), data.to(device), target.to(device)
            target = target.reshape(target.size(0), 1, target.size(1)).float()
            # Forward pass
            batch_size = data.shape[0]
            outputs = model(epoch, data, act_lens, target, 100)
            rois, rpn_cls_score, rpn_label, rpn_loss_cls, rpn_loss_bbox = outputs
            rpn_acc = acc_frame(rpn_cls_score, rpn_label)
            # Backward and optimize
            loss = rpn_loss_cls + balance_weight * rpn_loss_bbox 
            meter_rpn_acc.update(rpn_acc, 1)
            meter_loss.update(loss, 1)
            meter_rpn_loss_cls.update(rpn_loss_cls, 1)
            meter_rpn_loss_bbox.update(rpn_loss_bbox, 1)

            if batch_idx % args.log_interval == 0:
                print('Epoch: [{}/{}], Step [{}/{}], Val Loss: {:.4f}, Val RPN Acc: {:.4f}% '
                      .format(epoch+1, args.max_epochs, batch_idx+1, total_step, meter_loss.cur, meter_rpn_acc.cur))
                print('Epoch: [{}/{}], Step [{}/{}], Val RPN cls Loss: {:.4f}, Val RPN bbox Loss: {:.4f} '
                      .format(epoch+1, args.max_epochs, batch_idx+1, total_step, meter_rpn_loss_cls.cur, meter_rpn_loss_bbox.cur))

        print('Epoch: [{}/{}], Average Val Loss: {:.4f}, Average Val RPN cls Loss: {:.4f}, Average Val RPN bbox Loss: {:.4f}, Average Val RPN Acc: {:.4f}%'
             .format(epoch+1, args.max_epochs, meter_loss.avg, meter_rpn_loss_cls.avg, meter_rpn_loss_bbox.avg, meter_rpn_acc.avg))
    return float("{:.4f}".format(meter_loss.avg))

def test(args, model, device, test_loader, output_file, region_output_file):
    """Test the model"""
    write_post = open_or_fd(output_file, "wb")                                  
    fid = open(region_output_file, "w")
    model.eval()
    with torch.no_grad():
        total_step = len(test_loader)
        for batch_idx, (utt_ids, act_lens, data, target) in enumerate(test_loader):
            act_lens, data, target = act_lens.to(device), data.to(device), target.to(device)
            target = target.reshape(target.size(0), 1, target.size(1)).float()
            # Forward pass
            batch_size = data.shape[0]
            max_lens = data.shape[1]
            num_anchors_per_frame = args.num_anchor
            num_classes = args.output_dim
            outputs = model(0, data, act_lens, target, 100)
            rois, rpn_cls_score, anchors_per_utt = outputs
            rpn_cls_prob = F.softmax(rpn_cls_score, dim=2)
            disable_indexes = get_out_utt_boxes(anchors_per_utt, act_lens, batch_size)
            rpn_cls_prob[disable_indexes] = 0
            rpn_cls_prob = rpn_cls_prob.view(batch_size, max_lens, num_anchors_per_frame, num_classes)
            rois = rois.view(batch_size, max_lens, num_anchors_per_frame, 2)
            anchors_per_utt = anchors_per_utt.view(max_lens, num_anchors_per_frame, 2)
            rpn_cls_prob, arg_max_anchor = torch.max(rpn_cls_prob, dim=2)
            max_score, arg_max_score = torch.max(rpn_cls_prob, dim=1) # get the index of each utterance
            data_write = rpn_cls_prob.cpu().numpy()
            for i in range (len(utt_ids)):
                utt_id = utt_ids[i]
                act_len = act_lens[i]
                write_mat(write_post, data_write[i,0:act_len,:], utt_id)
                fid.writelines(utt_id)
                label = target[i][0].cpu().numpy()
                fid.writelines(", %f %f %f"%(label[0],label[1],label[2])) 
                for j in range(num_classes-1):
                    best_score1 = max_score[i][1+j]
                    best_frame1 = arg_max_score[i][1+j]
                    best_anchor1 = arg_max_anchor[i][best_frame1][1+j]
                    roi1 = rois[i][best_frame1][best_anchor1] # anchor of keyword 1
                    anchor1 = anchors_per_utt[best_frame1][best_anchor1]
                    roi1=roi1.cpu().numpy()
                    anchor1 = anchor1.cpu().numpy()
                    fid.writelines(", %f %f %f, %f %f %f"%(best_score1, anchor1[0], anchor1[1], best_score1, roi1[0], roi1[1]))
                fid.writelines("\n")
    write_post.close()
    fid.close()
def main():
    args = get_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if args.encoder=='gru':
        feature_extractor = GRU(input_size=args.input_dim, 
                output_size=args.hidden_dim, 
                hidden_size=args.hidden_dim, 
                num_layers=args.num_layers, 
                bias=True, batch_first=True, 
                dropout=args.dropout, 
                bidirectional=False, 
                output_layer=False)
    else:
        print("unsupported feature extractor: %s"%args.encoder)
        exit(1)

    rpn = RPN(128, args.num_anchor, args.output_dim)

    model = RPN_KWS(feature_extractor, rpn, args.output_dim).to(device)

    params = count_parameters(model)                                            
    print("Num parameters: %d, Num Flops: %d\n"%(params,0))
    
    if args.multi_gpu:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.init_weight_decay)

    print("Global Config:\n {}".format(cfg))
    print("Training Arguments:\n {}".format(args))
    print("Training Model:\n {}".format(model))
    print("Training Optimizer:\n {}".format(optimizer))

    # Load previous trained model
    if args.load_model != '':
        print("=> Loading previous checkpoint to train: {}".format(args.load_model))
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        prev_val_loss = checkpoint['prev_val_loss']
    elif not args.train:
        sys.exit("Option --load-model should not be empty for testing.")
    else:
        print("=> No checkpoint found.")
        prev_val_loss = float('inf')

    # For training
    if args.train:
        if args.train_scp == '' or args.dev_scp == '':
            sys.exit("Options --train-scp and --dev-scp are required for training.")

        if args.save_dir == '':
            sys.exit("Option --save-dir is required to save model.")

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        halving = 0
        best_model = args.load_model
        kwargs = {'num_workers': 3, 'pin_memory': True} if args.use_cuda else {}

        # Training data loader
        train_set = StreamingTorchDataset(args.train_scp, ["kaldi_reader", "raw_list_reader"], args.left_context, args.right_context)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn)

        # Dev data loader
        dev_set = StreamingTorchDataset(args.dev_scp,["kaldi_reader", "raw_list_reader"], args.left_context, args.right_context)
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn)

        for epoch in range(args.max_epochs):
            cur_tr_loss = train(args, model, device, train_loader,optimizer, epoch)
            cur_val_loss = validate(args, model, device, dev_loader, epoch)
            rel_impr = (prev_val_loss - cur_val_loss) / prev_val_loss

            model_name = 'nnet_epoch' + str(epoch+1) + '_lr' \
                        + str(args.learning_rate) + '_tr' + str(cur_tr_loss) \
                        + '_cv' + str(cur_val_loss) + '.ckpt'
            model_path = args.save_dir + '/' + model_name

            if cur_val_loss < prev_val_loss:

                prev_val_loss = cur_val_loss
                torch.save({
                    'prev_val_loss': prev_val_loss,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_path)
                best_model = model_path

                print("Model {} accepted. Time: {}".format(model_name,
                                                           datetime.datetime.now()))

            else:
                print ("Model {} rejected. Time: {}".format(model_name,
                                                            datetime.datetime.now()))
                if best_model != '':
                    print("=> Loading best checkpoint: {}".format(best_model))
                    checkpoint = torch.load(best_model)
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    prev_val_loss = checkpoint['prev_val_loss']
                else:
                    sys.exit("Error training neural network.")

            # Stopping training criterion
            if halving and rel_impr < args.end_halving_impr:
                if epoch < args.min_epochs:
                    print("We were supposed to finish, but we continue as min_epochs"
                          .format(args.min_epochs))
                    continue
                else:
                    print("Finished, too small relative improvement {}".format(rel_impr))
                    break

            # Start halving when improvement is low
            if rel_impr < args.start_halving_impr:
                halving = 1

            if halving:
                adjust_learning_rate(args, optimizer)
                print("Halving learning rate to {}".format(args.learning_rate))

        if best_model != args.load_model:
            final_model = args.save_dir + "/final.mdl"
            shutil.copyfile(best_model, final_model)
            print("Succeeded training the neural network: {}/final.mdl"
                  .format(args.save_dir))
        else:
            sys.exit("Error training neural network.")
    # For testing
    if args.test:
        # Test data loader
        if args.test_scp == '' or args.output_file == '':
            sys.exit("Options --test-scp and --output-file are required for testing")
        test_set = StreamingTorchDataset(args.test_scp,["kaldi_reader", "raw_list_reader"], args.left_context, args.right_context)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn)
        test(args, model, device, test_loader, args.output_file, args.region_output_file) 


if __name__ == '__main__':
    main()

