from torch import nn
import torch
from neuronlp2.nn import VarMaskedFastLSTM
import math
from graph_conv import FullGraphRel
from graphrel_add_gnn import GNNRel
# this is only p2 as the dep probs are already given
# based on GraphRel

class RelNetwork(nn.Module):
    def __init__(self, parser, base_encoder, graphrel, classifier, energy_temp=1., config=None, num_dep_rels=10):
        super(RelNetwork, self).__init__()
        self.parser = parser
        self.graphrel = graphrel
        self.classifier = classifier
        self.base_encoder = base_encoder
        self.energy_temp = energy_temp
        self.config = config
        self.num_dep_rels = num_dep_rels

    def forward(self, batch, batch_graph, instances, use_scores=False, one_best=False):
        word, char, pos, heads, types, masks, lengths, indices = batch
        word_graph, char_graph, pos_graph, _, _, masks_graph, lengths_graph, _ = batch_graph
        if one_best:
            with torch.no_grad():
                self.parser.eval()
                energy, one_best_tree = self.parser.get_probs(word, char, pos, mask=masks, length=lengths, energy_temp=self.energy_temp,
                                       use_scores=use_scores, get_mst_tree=one_best)
        elif self.config.parser_equal_probs:
            num_words = word.shape[1]
            average_prob = 1 / (num_words * self.num_dep_rels)
            energy = torch.full((word.shape[0], self.num_dep_rels, num_words, num_words), average_prob, device=word.device)
        else:
            self.parser.train()
            energy, _ = self.parser.get_probs(word, char, pos, mask=masks, length=lengths,
                                                          energy_temp=self.energy_temp,
                                                          use_scores=use_scores, get_mst_tree=one_best)
        word_h = self.base_encoder(word_graph, char_graph, pos_graph, masks_graph, lengths_graph)
        first_hiddens = []
        second_hiddens = []
        if one_best:
            heads, types = one_best_tree
            discrete_energy = torch.zeros_like(energy)
            for instance_index in range(heads.shape[0]):
                for dep in range(heads.shape[1]):
                    this_head = heads[instance_index][dep]
                    this_type = types[instance_index][dep]
                    discrete_energy[instance_index][this_type][this_head][dep] = 1.
            energy = discrete_energy


        if isinstance(self.graphrel, GCNRel):
            output = self.graphrel.forward(word_h, energy, mask=masks, length=lengths)
            for instance_index, instance in enumerate(instances):
                first_word_start, first_word_end, second_word_start, second_word_end = instance
                first_word_hidden = output[instance_index, first_word_start+1:first_word_end+1].mean(dim=-2)
                second_word_hidden = output[instance_index, second_word_start+1:second_word_end+1].mean(dim=-2)
                first_hiddens.append(first_word_hidden)
                second_hiddens.append(second_word_hidden)
            first_hiddens = torch.stack(first_hiddens, dim=0)
            second_hiddens = torch.stack(second_hiddens, dim=0)
        elif isinstance(self.graphrel, FullGraphRel):
            sent_len = word_h.shape[1]
            for instance_index, instance in enumerate(instances):
                first_word_start, first_word_end, second_word_start, second_word_end = instance
                first_word_hidden = word_h[instance_index, first_word_start+1:first_word_end+1].mean(dim=-2)
                second_word_hidden = word_h[instance_index, second_word_start+1:second_word_end+1].mean(dim=-2)
                first_hiddens.append(first_word_hidden)
                second_hiddens.append(second_word_hidden)
            first_hiddens = torch.stack(first_hiddens, dim=0)
            second_hiddens = torch.stack(second_hiddens, dim=0)
            first_hiddens = self.graphrel(energy, word_h, first_hiddens, second_hiddens, sent_len)
            second_hiddens = None
        else:
            sent_len = word_h.shape[1]
            for instance_index, instance in enumerate(instances):
                first_word_start, first_word_end, second_word_start, second_word_end = instance
                first_word_hidden = word_h[instance_index, first_word_start+1:first_word_end+1].mean(dim=-2)
                second_word_hidden = word_h[instance_index, second_word_start+1:second_word_end+1].mean(dim=-2)
                first_hiddens.append(first_word_hidden)
                second_hiddens.append(second_word_hidden)
            first_hiddens = torch.stack(first_hiddens, dim=0)
            second_hiddens = torch.stack(second_hiddens, dim=0)
            first_hiddens = self.graphrel(energy, word_h, first_hiddens, second_hiddens, sent_len)
            second_hiddens = None
        pred = self.classifier(first_hiddens, second_hiddens)
        return pred

class MeanPoolClassifier(nn.Module):
    def __init__(self, in_size=256, n_classes=2):
        super(MeanPoolClassifier, self).__init__()
        self.in_size = in_size
        self.n_classes = n_classes
        self.linear = nn.Linear(self.in_size, self.n_classes)

    def forward(self, tensor_0, tensor_1=None):
        if tensor_1 is not None: # graphrel
            features = torch.cat((tensor_0, tensor_1), dim=-1)
        else: # mem rel
            features = tensor_0
        logprobs = torch.log_softmax(self.linear(features), dim=-1)
        return logprobs

class GCN(nn.Module):
    def __init__(self, in_size=256, out_size=256//2):
        super(GCN, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.W = nn.Parameter(torch.rand((in_size, out_size)))
        self.b = nn.Parameter(torch.rand((out_size, )))

        self.init()

    def init(self):
        stdv = 1 / math.sqrt(self.out_size)

        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, inp, adj, is_relu=True):
        out = torch.matmul(inp, self.W) + self.b
        if adj is not None:
            out = torch.matmul(adj, out)
        if is_relu == True:
            out = nn.functional.relu(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(hid_size=%d)' % (self.hid_size)

class BaseEncoder(nn.Module):
    def __init__(self, hid_size=256, emb_pos=None, emb_word=None, emb_char=None, kernel_size=0, num_filters=0, num_rnn_encoder_layers=3
                 , p_rnn=(0.33, 0.33), dp=0.5):
        super(BaseEncoder, self).__init__()
        # base encoder structures
        self.emb_pos = emb_pos
        self.emb_word = emb_word
        self.emb_char = emb_char
        char_dim = self.emb_char.embedding_dim
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1)
        dim_enc = self.emb_word.embedding_dim + self.emb_pos.embedding_dim + num_filters
        self.base_rnn_encoder = VarMaskedFastLSTM(dim_enc, hid_size, num_layers=num_rnn_encoder_layers, batch_first=True, bidirectional=True, dropout=p_rnn)
        self.dp = nn.Dropout(dp)

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        # [batch, length, word_dim]
        # print(torch.max(input_word).cpu().item(), self.emb_word.weight.shape[0])

        word = self.emb_word(input_word)
        # apply dropout on input
        word = self.dp(word)

        input = word

        # if self.char:
        # [batch, length, char_length, char_dim]
        char = self.emb_char(input_char)
        char_size = char.size()
        # first transform to [batch *length, char_length, char_dim]
        # then transpose to [batch * length, char_dim, char_length]
        char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
        # put into cnn [batch*length, char_filters, char_length]
        # then put into maxpooling [batch * length, char_filters]
        char, _ = self.conv1d(char).max(dim=2)
        # reshape to [batch, length, char_filters]
        char = torch.tanh(char).view(char_size[0], char_size[1], -1)
        # apply dropout on input
        char = self.dp(char)
        # concatenate word and char [batch, length, word_dim+char_filter]
        input = torch.cat([input, char], dim=2)

        # if self.pos:
        # [batch, length, pos_dim]
        pos = self.emb_pos(input_pos)
        # apply dropout on input
        pos = self.dp(pos)
        input = torch.cat([input, pos], dim=2)

        # output from rnn [batch, length, hidden_size]
        output, hc = self.base_rnn_encoder(input, mask, hx=hx)

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dp(output.transpose(1, 2)).transpose(1, 2)

        return output

class GCNRel(nn.Module):
    def __init__(self, num_dep_rels=0, hid_size=256, gcn_layer=2, dp=0.5, dep_rel_emb_size=15, gnn_type='gnnrel',
                 energy_thres=0.):
        super(GCNRel, self).__init__()

        self.hid_size = hid_size
        self.gcn_layer = gcn_layer
        self.dp = dp

        self.emb_dep_rel = nn.Embedding(num_dep_rels, dep_rel_emb_size)
        self.after_gcn_linear = nn.ModuleList([nn.Linear(self.hid_size * 2, self.hid_size * 2) for i in range(self.gcn_layer)])

        if gnn_type == 'gcn':
            self.gcn_fw = nn.ModuleList([GCN(self.hid_size * 2, self.hid_size) if i == 0 else
                                         GCN(self.hid_size * 2, self.hid_size) for i in range(self.gcn_layer)])
            self.gcn_bw = nn.ModuleList([GCN(self.hid_size * 2, self.hid_size) if i == 0 else
                                         GCN(self.hid_size * 2, self.hid_size) for i in range(self.gcn_layer)])
        elif gnn_type == 'gnnrel':
            self.gcn_fw = nn.ModuleList([GNNRel(self.emb_dep_rel, in_size=self.hid_size * 2, out_size=self.hid_size) if i == 0 else
                                         GNNRel(self.emb_dep_rel, in_size=self.hid_size * 2, out_size=self.hid_size) for i in range(self.gcn_layer)])
            self.gcn_bw = nn.ModuleList([GNNRel(self.emb_dep_rel, in_size=self.hid_size * 2, out_size=self.hid_size) if i == 0 else
                                         GNNRel(self.emb_dep_rel, in_size=self.hid_size * 2, out_size=self.hid_size) for i in range(self.gcn_layer)])
        self.dp = nn.Dropout(dp)
        self.energy_thres = energy_thres
        self.gnn_type = gnn_type

    def forward(self, word_h, dep_probs, mask=None, length=None, hx=None):

        # sent_len = int(mask.sum().item())

        # dep_probs = dep_probs[:, :, :sent_len, :sent_len]
        # word_h = word_h[:, :sent_len]

        dep_probs = dep_probs.permute(0, 2, 3, 1)  # batch, head, dep, labels
        marginal_rel_probs = dep_probs.sum(dim=3)  # batch, head, dep

        out_fw, out_bw = 0, 0

        if self.gnn_type == 'gcn':
            dep_fw = marginal_rel_probs + torch.eye(marginal_rel_probs.shape[1]).to(marginal_rel_probs.device)
            dep_bw = marginal_rel_probs.transpose(1, 2) + torch.eye(marginal_rel_probs.shape[1]).to(marginal_rel_probs.device)

            for i in range(self.gcn_layer):
                out_fw = self.gcn_fw[i](word_h, dep_fw)
                out_bw = self.gcn_bw[i](word_h, dep_bw)
                word_h = torch.cat([out_fw, out_bw], dim=2)
            else:
                word_h = self.dp(word_h)
        elif self.gnn_type == 'gnnrel':
            dep_probs = dep_probs.permute(0, 2, 1, 3) # batch, dep, head, labels
            dep_probs = dep_probs.masked_fill(dep_probs < self.energy_thres, 0.0)
            backward_dep_probs = dep_probs.permute(0, 2, 1, 3).contiguous()
            for i in range(self.gcn_layer):
                out_fw = self.gcn_fw[i](word_h, dep_probs)
                out_bw = self.gcn_bw[i](word_h, backward_dep_probs)
                word_h = torch.cat([out_fw, out_bw], dim=2)
                word_h = self.dp(torch.relu(self.after_gcn_linear[i](word_h)))
        return word_h
