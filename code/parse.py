import os
import torch
import random
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="TopKformer")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--similarity', type=str, default='dot')
    # TopKformer
    parser.add_argument('--TopKformer_layers', type=int, default=4)
    # Train
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--show_loss_interval', type=int, default=1)
    # bpr/sl loss
    parser.add_argument('--emb_loss_func', type=str, default='bpr_edge')
    parser.add_argument('--emb_learning_rate', type=float, default=1e-1)
    parser.add_argument('--emb_weight_decay', type=float, default=1e-4)
    parser.add_argument('--emb_reg_lambda', type=float, default=1e-4)
    parser.add_argument('--emb_loss_batch_size', type=int, default=0)
    # Test
    parser.add_argument('--topks', type=str, default='[1,3,5,10,20]')
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--valid_interval', type=int, default=20)
    parser.add_argument('--stopping_step', type=int, default=10)
    # Data
    parser.add_argument('--data', type=str, default="AliEC")
    parser.add_argument('--data_id', type=str, default='5')
    return parser.parse_args()


args = parse_args()
args.topks = eval(args.topks)
args.device = torch.device(f'cuda:{args.device:d}')

if args.seed != -1:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f'seed: {args.seed:d}')

print('Using', args.device)

print('Model Setting')
print(f'    hidden dim: {args.hidden_dim:d}')
print(f'    Using {args.TopKformer_layers:d} layers Rankformer:')

print('Train Setting')
print(f'    max epochs: {args.max_epochs:d}')
print(f'        reg lambda: {args.emb_reg_lambda:f}')
print(f'        learning rate: {args.emb_learning_rate:f}')
print(f'        weight decay: {args.emb_weight_decay:f}')
print(f'        loss batch size: {args.emb_loss_batch_size:d}')
print(f'        train emb & beta')

print('Test Setting')
print(f'    topks: ', args.topks)
print(f'    test batch size: {args.test_batch_size:d}')
print(f'    valid interval: {args.valid_interval:d}')
print(f'    stopping step: {args.stopping_step:d}')

print('Data Setting')
args.data_dir = "./data/"
args.train_file = os.path.join(args.data_dir, args.data, f'train.txt')
args.valid_file = os.path.join(args.data_dir, args.data, f'valid.txt')
args.test_file = os.path.join(args.data_dir, args.data, f'test.txt')
print(f'    train: {args.train_file:s}')
print(f'    valid: {args.valid_file:s}')
print(f'    test: {args.test_file:s}')

print('---------------------------')
