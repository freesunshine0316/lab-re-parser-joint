from torch import nn
import torch
import math
# this is only p2 as the dep probs are already given
# based on GraphRel

class GNNRel(nn.Module):
    def __init__(self, rel_embs, in_size=256, out_size=256//2):
        super(GNNRel, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.W = nn.Parameter(torch.rand((in_size+rel_embs.embedding_dim, out_size)))
        self.b = nn.Parameter(torch.rand((out_size, )))

        self.rel_embedding = rel_embs.weight

    def init(self):
        stdv = 1 / math.sqrt(self.out_size)

        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    # adj_full: [batch, dep, head, L], where L shows the number of relations
    #           Be careful, sum(seq * L) == 1.0, not sum(L)
    def forward(self, inp, adj_full, is_relu=True):
        adj = adj_full.sum(dim=-1) # [batch, seq, seq]
        neigh_repre = torch.matmul(adj, inp) + inp # [batch, seq, hidden_dim]
        batch_size, seq_num, _, L = list(adj_full.size())
        # rel_repre: [batch, seq, emb_dim]
        rel_repre = torch.matmul(adj_full.view(batch_size, seq_num, -1), # [batch, seq, seq*L]
                self.rel_embedding.repeat(seq_num, 1)) #[seq*L, emb_dim]
        final_repre = torch.cat([neigh_repre, rel_repre], dim=2) # [batch, seq, hidden_dim + emb_dim]
        final_repre = torch.matmul(final_repre, self.W) + self.b
        return nn.functional.relu(final_repre) if is_relu else final_repre

    def __repr__(self):
        return self.__class__.__name__ + '(hid_size=%d)' % (self.hid_size)
