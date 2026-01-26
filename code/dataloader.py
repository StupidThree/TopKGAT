from parse import args
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F


class MyDataset():
    def __init__(self, train_file, valid_file, test_file, device):
        self.device = device
        # train dataset
        train_data = pd.read_table(train_file, header=None, sep=' ')
        train_data = torch.from_numpy(train_data.values).to(self.device)
        self.train_data = train_data[(train_data[:, 0]).argsort(), :]
        self.train_user, self.train_item = self.train_data[:, 0], self.train_data[:, 1]
        # valid dataset
        valid_data = pd.read_table(valid_file, header=None, sep=' ')
        valid_data = torch.from_numpy(valid_data.values).to(self.device)
        self.valid_data = valid_data[torch.argsort(valid_data[:, 0]), :]
        self.valid_user, self.valid_item = self.valid_data[:, 0], self.valid_data[:, 1]
        # test dataset
        test_data = pd.read_table(test_file, header=None, sep=' ')
        test_data = torch.from_numpy(test_data.values).to(self.device)
        self.test_data = test_data[torch.argsort(test_data[:, 0]), :]
        self.test_user, self.test_item = self.test_data[:, 0], self.test_data[:, 1]
        # process
        self.num_users = max(self.train_user.max(), self.valid_user.max(), self.test_user.max()).cpu()+1
        self.num_items = max(self.train_item.max(), self.valid_item.max(), self.test_item.max()).cpu()+1
        self.num_nodes = self.num_users+self.num_items
        self.du = torch.zeros(self.num_users).long().to(args.device).index_add(0, self.train_user, torch.ones_like(self.train_user))
        self.di = torch.zeros(self.num_items).long().to(args.device).index_add(0, self.train_item, torch.ones_like(self.train_item))
        print(f'{self.num_users:d} users, {self.num_items:d} items.')
        print(f'train: {self.train_user.shape[0]:d}, valid: {self.valid_user.shape[0]:d}, test: {self.test_user.shape[0]:d}.')
        self.build_batch()
        self.shuffle_batch_users, self.shuffle_train_batch = None, None

    def build_batch(self):
        # for test & valid
        self.train_degree = self.du
        self.test_degree = torch.zeros(self.num_users).long().to(args.device).index_add(0, self.test_user, torch.ones_like(self.test_user))
        self.valid_degree = torch.zeros(self.num_users).long().to(args.device).index_add(0, self.valid_user, torch.ones_like(self.valid_user))
        self.batch_users = [torch.arange(i, min(i+args.test_batch_size, self.num_users)).to(args.device) for i in range(0, self.num_users, args.test_batch_size)]
        self.train_batch = list(self.train_data.split([self.train_degree[batch_user].sum() for batch_user in self.batch_users]))
        self.test_batch = list(self.test_data.split([self.test_degree[batch_user].sum() for batch_user in self.batch_users]))
        self.valid_batch = list(self.valid_data.split([self.valid_degree[batch_user].sum() for batch_user in self.batch_users]))

    def build_shuffle_train_batch(self, batch_size):
        n = self.num_users
        shuffle_user_id = torch.randperm(n).to(args.device)
        shuffle_user = torch.zeros_like(shuffle_user_id)
        shuffle_user[shuffle_user_id] = torch.arange(n).long().to(args.device)
        # shuffle_user_id[user]=idx; shuffle_user[idx]=user
        shuffle_indices = shuffle_user_id[self.train_data[:, 0]].argsort()
        shuffle_train_data = self.train_data[shuffle_indices, :]
        self.shuffle_batch_users = [shuffle_user[i:i+batch_size] for i in range(0, n, batch_size)]
        self.shuffle_train_batch = list(shuffle_train_data.split([self.du[batch_user].sum() for batch_user in self.shuffle_batch_users]))


dataset = MyDataset(args.train_file, args.valid_file, args.test_file, args.device)
