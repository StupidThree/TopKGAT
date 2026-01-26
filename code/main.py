from parse import args
from dataloader import dataset
from model import Model
import torch
import numpy as np


def print_test_result():
    print(f'===== Test Result(at {model.best_epoch:d} epoch) =====')
    for i, k in enumerate(args.topks):
        print(f'ndcg@{k:d} = {model.test_ndcg[i]:f}, recall@{k:d} = {model.test_recall[i]:f}, pre@{k:d} = {model.test_pre[i]:f}')


model = Model(dataset).to(args.device)
model.valid_func(epoch=0)
for epoch in range(1, args.max_epochs+1):
    model.train_func(epoch)
    if args.emb_loss_batch_size or epoch % args.valid_interval == 0:
        model.valid_func(epoch)
    if epoch-model.best_epoch >= args.stopping_step*(1 if args.emb_loss_batch_size else args.valid_interval):
        break
print('---------------------------')
print('done.')
print_test_result()
