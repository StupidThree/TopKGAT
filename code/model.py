from parse import args
from rec import TopKformer, sparse_sum
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def InfoNCE(x, y, tau=0.15, b_cos=True):
    if b_cos:
        x, y = F.normalize(x), F.normalize(y)
    return -torch.diag(F.log_softmax((x@y.T)/tau, dim=1)).mean()


def test(pred, pred_popularity, test, recall_n):
    pred = torch.isin(pred[recall_n > 0], test)
    recall_n = recall_n[recall_n > 0]
    pre, recall, ndcg = [], [], []
    for k in args.topks:
        right_pred = pred[:, :k].sum(1)
        recall_k = recall_n.clamp(max=k)
        # precision
        pre.append((right_pred/k).sum())
        # recall
        recall.append((right_pred/recall_k).sum())
        # ndcg
        dcg = (pred[:, :k]/torch.arange(2, k+2).to(args.device).unsqueeze(0).log2()).sum(1)
        d_val = (1/torch.arange(2, k+2).to(args.device).log2()).cumsum(0)
        idcg = d_val[recall_k-1]
        ndcg.append((dcg / idcg).sum())
    return recall_n.shape[0], torch.tensor(pre), torch.tensor(recall), torch.tensor(ndcg)


class NegativeSampler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.n, self.m = dataset.num_users, dataset.num_items
        self.edge_id = dataset.train_user*self.m+dataset.train_item

    def negative_sampling(self, v):
        j = torch.randint_like(v, 0, self.m)
        mask = torch.isin(v*self.m+j, self.edge_id)
        while mask.sum() > 0:
            j[mask] = torch.randint_like(j[mask], 0, self.m)
            mask = torch.isin(v*self.m+j, self.edge_id)
        return j


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.dataset = dataset
        self.hidden_dim = args.hidden_dim
        self.embedding_user = nn.Embedding(self.dataset.num_users, self.hidden_dim)
        self.embedding_item = nn.Embedding(self.dataset.num_items, self.hidden_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.TopKformer = TopKformer(dataset)
        self.TopKformer_beta = nn.Embedding(self.dataset.num_users, args.TopKformer_layers)
        self.emb_parameters = [
            {'params': self.embedding_user.parameters()},
            {'params': self.embedding_item.parameters()},
            {'params': self.TopKformer_beta.parameters()},
        ]
        self.emb_optimizer = torch.optim.AdamW(
            self.emb_parameters,
            lr=args.emb_learning_rate,
            weight_decay=args.emb_weight_decay)
        self.negative_sampler = NegativeSampler(self.dataset)
        self._users, self._items = None, None
        self.best_valid_ndcg, self.best_epoch = 0, 0
        self.test_pre, self.test_recall, self.test_ndcg = torch.zeros(len(args.topks)), torch.zeros(len(args.topks)), torch.zeros(len(args.topks))

    @property
    def beta_value(self):
        return F.tanh(self.TopKformer_beta.weight)

    def similarity_func(self, user_emb, item_emb):
        if user_emb.shape[1] == 1 and item_emb.shape[0] == 1:
            return torch.mm(user_emb.squeeze(1), item_emb.squeeze(0).t())
        return torch.sum(torch.mul(user_emb, item_emb), dim=-1)

    def computer(self):
        u, i = self.dataset.train_user, self.dataset.train_item
        users_emb, items_emb = self.embedding_user.weight, self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(args.TopKformer_layers):
            all_emb = self.TopKformer(all_emb,
                                      self.beta_value[:, _])
            embs.append(all_emb)
        self._users, self._items = torch.split(all_emb, [self.dataset.num_users, self.dataset.num_items])

    def evaluate(self, test_batch, test_degree):
        self.eval()
        if self._users is None:
            self.computer()
        user_emb, item_emb = self._users, self._items
        max_K = max(args.topks)
        all_pre = torch.zeros(len(args.topks))
        all_recall = torch.zeros(len(args.topks))
        all_ndcg = torch.zeros(len(args.topks))
        all_cnt = 0
        with torch.no_grad():
            for batch_users, batch_train, ground_true in zip(self.dataset.batch_users, self.dataset.train_batch, test_batch):
                user_e = user_emb[batch_users]
                rating = self.similarity_func(user_e.unsqueeze(1), item_emb.unsqueeze(0))
                rating[batch_train[:, 0]-batch_users[0], batch_train[:, 1]] = -(1 << 10)
                _, pred_items = torch.topk(rating, k=max_K)
                cnt, pre, recall, ndcg = test(
                    batch_users.unsqueeze(1)*self.dataset.num_items+pred_items,
                    self.dataset.di[pred_items]/self.dataset.di.max(),
                    ground_true[:, 0]*self.dataset.num_items+ground_true[:, 1],
                    test_degree[batch_users])
                all_pre += pre
                all_recall += recall
                all_ndcg += ndcg
                all_cnt += cnt
            all_pre /= all_cnt
            all_recall /= all_cnt
            all_ndcg /= all_cnt
        return all_pre, all_recall, all_ndcg

    def train_func(self, epoch):
        print(f'EPOCH {epoch:d}')
        emb_loss = self.train_emb(epoch)
        print(f'  emb Loss = {emb_loss:f}')

    def valid_func(self, epoch):
        valid_pre, valid_recall, valid_ndcg = self.evaluate(self.dataset.valid_batch, self.dataset.valid_degree)
        for i, k in enumerate(args.topks):
            print(f'  Valid Result: ndcg@{k:d} = {valid_ndcg[i]:f}, recall@{k:d} = {valid_recall[i]:f}, pre@{k:d} = {valid_pre[i]:f}')
        if valid_ndcg[-1] > self.best_valid_ndcg:
            self.best_valid_ndcg, self.best_epoch = valid_ndcg[-1], epoch
            self.test_func(epoch)

    def test_func(self, epoch):
        self.test_pre, self.test_recall, self.test_ndcg = self.evaluate(self.dataset.test_batch, self.dataset.test_degree)
        for i, k in enumerate(args.topks):
            print(f'  Test Result: ndcg@{k:d} = {self.test_ndcg[i]:f}, recall@{k:d} = {self.test_recall[i]:f}, pre@{k:d} = {self.test_pre[i]:f}')

    def train_emb(self, epoch):
        self.train()
        self.computer()
        train_loss = self.loss_func(self.dataset.train_user, self.dataset.train_item)
        self.emb_optimizer.zero_grad()
        train_loss.backward()
        self.emb_optimizer.step()
        print(f'  (emb) loss = {train_loss:f}')
        return train_loss

    def loss_func(self, u, i):
        n, m = self.dataset.num_users, self.dataset.num_items
        j = self.negative_sampler.negative_sampling(u)
        scores_ui = self.similarity_func(self._users[u], self._items[i])
        scores_uj = self.similarity_func(self._users[u], self._items[j])
        loss = F.softplus(scores_uj-scores_ui).mean()
        reg_loss = (1/2)*(self.embedding_user(u).norm(dim=-1).pow(2).mean() +
                          self.embedding_item(i).norm(dim=-1).pow(2).mean() +
                          self.embedding_item(j).norm(dim=-1).pow(2).mean())
        return loss+args.emb_reg_lambda*reg_loss
