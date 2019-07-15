import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import pickle

DATASET = 'nyt'
ARCH = '2p'

tr = pickle.load(open('Dataset/dataset_mul-rel/%s/tr.pkl' % (DATASET), 'rb'))
vl = pickle.load(open('Dataset/dataset_mul-rel/%s/vl.pkl' % (DATASET), 'rb'))
ts = pickle.load(open('Dataset/dataset_mul-rel/%s/ts.pkl' % (DATASET), 'rb'))

print(len(tr), len(vl), len(ts))
pre_tr = []
for i in range(6):
    tmp = pickle.load(open('Dataset/dataset_mul-rel/%s/pre_tr-%d.pkl' % (DATASET, i), 'rb'))
    pre_tr += tmp
pre_vl = pickle.load(open('Dataset/dataset_mul-rel/%s/pre_vl.pkl' % (DATASET), 'rb'))
pre_ts = pickle.load(open('Dataset/dataset_mul-rel/%s/pre_ts.pkl' % (DATASET), 'rb'))

print(len(pre_tr), len(pre_vl), len(pre_ts))

NUM_REL = dict()
MXL = 0

for d in tr:
    sent = d[0]
    MXL = max(len(sent) + 4, MXL)

    rels = d[1]

    for rel in rels:
        rel = rel[2]

        if not rel in NUM_REL:
            NUM_REL[rel] = 1

print(NUM_REL)

NUM_REL = len(NUM_REL) + 1  # 0 for NA
print('Num of relation: %d' % (NUM_REL))
print('Max length: %d' % (MXL))

import spacy

NLP = spacy.load('en_core_web_lg')

POS = dict()
for pos in list(NLP.tagger.labels):
    POS[pos] = len(POS) + 1

NUM_POS = len(POS) + 1  # 0 for NA
print(POS)
print('Num of pos: %d' % (NUM_POS))

import torch as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm_notebook as tqdm

class DS(Dataset):
    def __init__(self, dat):
        super(DS, self).__init__()

        self.dat = dat

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        return self.dat[idx]


ld_tr = DataLoader(DS(pre_tr), batch_size=32, shuffle=True)
ld_vl = DataLoader(DS(pre_vl), batch_size=64)
ld_ts = DataLoader(DS(pre_ts), batch_size=64)

for idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel in ld_ts:
    print(idx.shape)
    print(inp.shape, pos.shape, dep_fw.shape, dep_bw.shape)
    print(ans_ne.shape, wgt_ne.shape)
    print(ans_rel.shape, wgt_rel.shape)

    break

import math


class GCN(nn.Module):
    def __init__(self, hid_size=256):
        super(GCN, self).__init__()

        self.hid_size = hid_size

        self.W = nn.Parameter(T.FloatTensor(self.hid_size, self.hid_size // 2).cuda())
        self.b = nn.Parameter(T.FloatTensor(self.hid_size // 2, ).cuda())

        self.init()

    def init(self):
        stdv = 1 / math.sqrt(self.hid_size // 2)

        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj, is_relu=True):
        out = T.matmul(inp, self.W) + self.b
        out = T.matmul(adj, out)

        if is_relu == True:
            out = nn.functional.relu(out)

        return out

    def __repr__(self):
        return self.__class__.__name__ + '(hid_size=%d)' % (self.hid_size)


gcn = GCN().cuda()
print(gcn)


class Model_GraphRel(nn.Module):
    def __init__(self, mxl, num_rel,
                 hid_size=256, rnn_layer=2, gcn_layer=2, dp=0.5):
        super(Model_GraphRel, self).__init__()

        self.mxl = mxl
        self.num_rel = num_rel
        self.hid_size = hid_size
        self.rnn_layer = rnn_layer
        self.gcn_layer = gcn_layer
        self.dp = dp

        self.emb_pos = nn.Embedding(NUM_POS, 15)

        self.rnn = nn.GRU(300 + 15, self.hid_size,
                          num_layers=self.rnn_layer, batch_first=True, dropout=dp, bidirectional=True)
        self.gcn_fw = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.gcn_layer)])
        self.gcn_bw = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.gcn_layer)])

        self.rnn_ne = nn.GRU(self.hid_size * 2, self.hid_size,
                             batch_first=True)
        self.fc_ne = nn.Linear(self.hid_size, 5)

        self.trs0_rel = nn.Linear(self.hid_size * 2, self.hid_size)
        self.trs1_rel = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc_rel = nn.Linear(self.hid_size * 2, self.num_rel)

        if ARCH == '2p':
            self.gcn2p_fw = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.num_rel)])
            self.gcn2p_bw = nn.ModuleList([GCN(self.hid_size * 2) for _ in range(self.num_rel)])

        self.dp = nn.Dropout(dp)

    def output(self, feat):
        out_ne, _ = self.rnn_ne(feat)
        out_ne = self.dp(out_ne)
        out_ne = self.fc_ne(out_ne)

        trs0 = nn.functional.relu(self.trs0_rel(feat))
        trs0 = self.dp(trs0)
        trs1 = nn.functional.relu(self.trs1_rel(feat))
        trs1 = self.dp(trs1)

        trs0 = trs0.view((trs0.shape[0], trs0.shape[1], 1, trs0.shape[2]))
        trs0 = trs0.expand((trs0.shape[0], trs0.shape[1], trs0.shape[1], trs0.shape[3]))
        trs1 = trs1.view((trs1.shape[0], 1, trs1.shape[1], trs1.shape[2]))
        trs1 = trs1.expand((trs1.shape[0], trs1.shape[2], trs1.shape[2], trs1.shape[3]))
        trs = T.cat([trs0, trs1], dim=3)

        out_rel = self.fc_rel(trs)

        return out_ne, out_rel

    def forward(self, inp, pos, dep_fw, dep_bw):
        pos = self.emb_pos(pos)
        inp = T.cat([inp, pos], dim=2)
        inp = self.dp(inp)

        out, _ = self.rnn(inp)

        for i in range(self.gcn_layer):
            out_fw = self.gcn_fw[i](out, dep_fw)
            out_bw = self.gcn_bw[i](out, dep_bw)

            out = T.cat([out_fw, out_bw], dim=2)
            out = self.dp(out)

        feat_1p = out
        out_ne, out_rel = self.output(feat_1p)

        if ARCH == '1p':
            return out_ne, out_rel

        # 2p
        out_ne1, out_rel1 = out_ne, out_rel

        dep_fw = nn.functional.softmax(out_rel, dim=3)
        dep_bw = dep_fw.transpose(1, 2)

        outs = []
        for i in range(self.num_rel):
            out_fw = self.gcn2p_fw[i](feat_1p, dep_fw[:, :, :, i])
            out_bw = self.gcn2p_bw[i](feat_1p, dep_bw[:, :, :, i])

            outs.append(self.dp(T.cat([out_fw, out_bw], dim=2)))

        feat_2p = feat_1p
        for i in range(self.num_rel):
            feat_2p = feat_2p + outs[i]

        out_ne2, out_rel2 = self.output(feat_2p)

        return out_ne1, out_rel1, out_ne2, out_rel2


model = nn.DataParallel(Model_GraphRel(mxl=MXL, num_rel=NUM_REL)).cuda()

if ARCH == '1p':
    out_ne, out_rel = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
    print(out_ne1.shape, out_rel1.shape)

else:
    out_ne1, out_rel1, out_ne2, out_rel2 = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())
    print(out_ne1.shape, out_rel1.shape)

def post_proc(dat, idx, out_ne, out_rel):
    out_ne = np.argmax(out_ne.detach().cpu().numpy(), axis=1)
    out_rel = np.argmax(out_rel.detach().cpu().numpy(), axis=2)

    nes = dict()
    el = -1
    for i, v in enumerate(out_ne):
        if v == 4:
            nes[i] = [i, i]
            el = -1

        elif v == 1:
            el = i

        elif v == 3:
            if not el == -1:
                for p in range(el, i + 1):
                    nes[p] = [el, i]

        elif v == 2:
            pass

        elif v == 0:
            el = -1

    rels = []
    for i in range(MXL):
        for j in range(MXL):
            if not out_rel[i][j] == 0 and i in nes and j in nes:
                rels.append([nes[i][1], nes[j][1], out_rel[i][j]])

    cl = []
    for rel in rels:
        if not rel in cl:
            cl.append(rel)
    rels = cl

    ans = []
    for tmp in dat[idx][1]:
        ans.append([tmp[0][1], tmp[1][1], tmp[2]])

    cl = []
    for rel in ans:
        if not rel in cl:
            cl.append(rel)
    ans = cl

    return rels, ans


class F1:
    def __init__(self):
        self.P = [0, 0]
        self.R = [0, 0]

    def get(self):
        try:
            P = self.P[0] / self.P[1]
        except:
            P = 0

        try:
            R = self.R[0] / self.R[1]
        except:
            R = 0

        try:
            F = 2 * P * R / (P + R)
        except:
            F = 0

        return P, R, F

    def add(self, ro, ra):
        self.P[1] += len(ro)
        self.R[1] += len(ra)

        for r in ro:
            if r in ra:
                self.P[0] += 1

        for r in ra:
            if r in ro:
                self.R[0] += 1


EPOCHS = 200
LR = 0.0008
DECAY = 0.98

W_NE = 2
W_REL = 2
ALP = 3

loss_func = nn.CrossEntropyLoss(reduction='none').cuda()
optim = T.optim.Adam(model.parameters(), lr=LR)


def ls(out_ne, wgt_ne, out_rel, wgt_rel):
    ls_ne = loss_func(out_ne.view((-1, 5)), ans_ne.view((-1,)).cuda()).view(ans_ne.shape)
    ls_ne = (ls_ne * wgt_ne.cuda()).sum() / (wgt_ne > 0).sum().cuda()

    ls_rel = loss_func(out_rel.view((-1, NUM_REL)), ans_rel.view((-1,)).cuda()).view(ans_rel.shape)
    ls_rel = (ls_rel * wgt_rel.cuda()).sum() / (wgt_rel > 0).sum().cuda()

    return ls_ne, ls_rel


for e in tqdm(range(EPOCHS)):
    ls_ep_ne1, ls_ep_rel1 = 0, 0
    ls_ep_ne2, ls_ep_rel2 = 0, 0

    model.train()
    with tqdm(ld_tr) as TQ:
        for i, (idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel) in enumerate(TQ):

            wgt_ne.masked_fill_(wgt_ne == 1, W_NE)
            wgt_ne.masked_fill_(wgt_ne == 0, 1)
            wgt_ne.masked_fill_(wgt_ne == -1, 0)

            wgt_rel.masked_fill_(wgt_rel == 1, W_REL)
            wgt_rel.masked_fill_(wgt_rel == 0, 1)
            wgt_rel.masked_fill_(wgt_rel == -1, 0)

            out_ne1, out_rel1, out_ne2, out_rel2 = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())

            ls_ne1, ls_rel1 = ls(out_ne1, wgt_ne, out_rel1, wgt_rel)
            ls_ne2, ls_rel2 = ls(out_ne2, wgt_ne, out_rel2, wgt_rel)

            optim.zero_grad()
            ((ls_ne1 + ls_rel1) + ALP * (ls_ne2 + ls_rel2)).backward()
            optim.step()

            ls_ne1 = ls_ne1.detach().cpu().numpy()
            ls_rel1 = ls_rel1.detach().cpu().numpy()
            ls_ep_ne1 += ls_ne1
            ls_ep_rel1 += ls_rel1

            ls_ne2 = ls_ne2.detach().cpu().numpy()
            ls_rel2 = ls_rel2.detach().cpu().numpy()
            ls_ep_ne2 += ls_ne2
            ls_ep_rel2 += ls_rel2

            TQ.set_postfix(ls_ne1='%.3f' % (ls_ne1), ls_rel1='%.3f' % (ls_rel1),
                           ls_ne2='%.3f' % (ls_ne2), ls_rel2='%.3f' % (ls_rel2))

            if i % 100 == 0:
                for pg in optim.param_groups:
                    pg['lr'] *= DECAY

        ls_ep_ne1 /= len(TQ)
        ls_ep_rel1 /= len(TQ)

        ls_ep_ne2 /= len(TQ)
        ls_ep_rel2 /= len(TQ)

        print('Ep %d: ne1: %.4f, rel1: %.4f, ne2: %.4f, rel2: %.4f' % (e + 1, ls_ep_ne1, ls_ep_rel1,
                                                                       ls_ep_ne2, ls_ep_rel2))
        T.save(model.state_dict(), 'Model/%s_%s_%d.pt' % (DATASET, ARCH, e + 1))

    f1 = F1()
    model.eval()
    with tqdm(ld_vl) as TQ:
        for idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel in TQ:
            _, _, out_ne, out_rel = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())

            for i in range(idx.shape[0]):
                rels, ans = post_proc(vl, idx[i], out_ne[i], out_rel[i])
                f1.add(rels, ans)

        p, r, f = f1.get()
        print('P: %.4f%%, R: %.4f%%, F: %.4f%%' % (100 * p, 100 * r, 100 * f))
# model.load_state_dict('')

f1 = F1()
model.eval()
with tqdm(ld_ts) as TQ:
    for idx, inp, pos, dep_fw, dep_bw, ans_ne, wgt_ne, ans_rel, wgt_rel in TQ:
        _, _, out_ne, out_rel = model(inp.cuda(), pos.cuda(), dep_fw.cuda(), dep_bw.cuda())

        for i in range(idx.shape[0]):
            rels, ans = post_proc(ts, idx[i], out_ne[i], out_rel[i])
            f1.add(rels, ans)

    p, r, f = f1.get()
    print('Test: P: %.4f%%, R: %.4f%%, F: %.4f%%' % (100 * p, 100 * r, 100 * f))