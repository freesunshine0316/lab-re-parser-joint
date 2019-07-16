from torch import nn
import torch
from neuronlp2.nn import VarMaskedFastLSTM
import math
from graph_conv import FullGraphRel
# this is only p2 as the dep probs are already given
# based on GraphRel

class RelNetwork(nn.Module):
    def __init__(self, parser, base_encoder, graphrel, classifier, energy_temp=1.):
        super(RelNetwork, self).__init__()
        self.parser = parser
        self.graphrel = graphrel
        self.classifier = classifier
        self.base_encoder = base_encoder
        self.energy_temp = energy_temp

    def forward(self, batch, batch_graph, instances):
        word, char, pos, heads, types, masks, lengths, indices = batch
        word_graph, char_graph, pos_graph, _, _, masks_graph, lengths_graph, _ = batch_graph
        first_word_start, first_word_end, second_word_start, second_word_end = instances[0]
        energy = self.parser.get_probs(word, char, pos, mask=masks, length=lengths, energy_temp=self.energy_temp)
        word_h = self.base_encoder(word_graph, char_graph, pos_graph, masks_graph, lengths_graph)
        if isinstance(self.graphrel, GCNRel):
            output = self.graphrel.forward(word_h, energy, mask=masks, length=lengths)
            # label = 1 if item['ref'] == 'True' else 0
            first_word_hidden = output.squeeze()[first_word_start+1:first_word_end+1]
            second_word_hidden = output.squeeze()[second_word_start+1:second_word_end+1]
        elif isinstance(self.graphrel, FullGraphRel):
            sent_len = int(masks.sum().item())
            first_word_hidden = (word_h.squeeze()[first_word_start+1:first_word_end+1]).mean(dim=-2)
            second_word_hidden = (word_h.squeeze()[second_word_start+1:second_word_end+1]).mean(dim=-2)
            first_word_hidden = self.graphrel(energy, word_h, first_word_hidden, second_word_hidden, sent_len)
            second_word_hidden = None
        else:
            sent_len = int(masks.sum().item())
            first_word_hidden = (word_h.squeeze()[first_word_start+1:first_word_end+1]).mean(dim=-2)
            second_word_hidden = (word_h.squeeze()[second_word_start+1:second_word_end+1]).mean(dim=-2)
            first_word_hidden = self.graphrel(energy, word_h, first_word_hidden, second_word_hidden, sent_len)
            second_word_hidden = None
        pred = self.classifier(first_word_hidden, second_word_hidden)
        return pred

class MeanPoolClassifier(nn.Module):
    def __init__(self, in_size=256, n_classes=2):
        super(MeanPoolClassifier, self).__init__()
        self.in_size = in_size
        self.n_classes = n_classes
        self.linear = nn.Linear(self.in_size, self.n_classes)

    def forward(self, tensor_0, tensor_1=None):
        if tensor_1 is not None: # graphrel
            tensor_mean_0 = torch.mean(tensor_0, dim=-2)
            tensor_mean_1 = torch.mean(tensor_1, dim=-2)
            features = torch.cat((tensor_mean_0, tensor_mean_1), dim=-1)
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
        # out = torch.matmul(adj, out)
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
    def __init__(self, mxl, num_rel, num_dep_rels=0, hid_size=256, rnn_layer=2, gcn_layer=2, dp=0.5,  dep_rel_emb_size=15):
        super(GCNRel, self).__init__()

        self.mxl = mxl
        self.num_rel = num_rel
        self.hid_size = hid_size
        self.rnn_layer = rnn_layer
        self.gcn_layer = gcn_layer
        self.dp = dp

        self.emb_dep_rel = nn.Embedding(num_dep_rels, dep_rel_emb_size)

        self.gcn_fw = nn.ModuleList([GCN(self.hid_size * 2+dep_rel_emb_size, self.hid_size) for i in range(self.gcn_layer)])
        self.gcn_bw = nn.ModuleList([GCN(self.hid_size * 2+dep_rel_emb_size, self.hid_size) for i in range(self.gcn_layer)])

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
        trs = torch.cat([trs0, trs1], dim=3)

        out_rel = self.fc_rel(trs)

        return out_ne, out_rel

    def forward(self, word_h, dep_probs, mask=None, length=None, hx=None):
        # pos = self.emb_pos(pos)
        # inp = torch.cat([inp, pos], dim=2)
        # inp = self.dp(inp)
        #drop the paddings
        sent_len = int(mask.sum().item())

        dep_probs = dep_probs[:, :, :sent_len, :sent_len]
        word_h = word_h[:, :sent_len]

        dep_probs = dep_probs.permute(0, 2, 3, 1)  # batch, head, dep, labels
        marginal_rel_probs = dep_probs.sum(dim=3)  # batch, head, dep

        for i in range(self.gcn_layer):

            word_h = word_h.transpose(1, 2)

            weighted_sum_of_labels = dep_probs @ self.emb_dep_rel.weight # batch, head, dep, dep rel embs

            weighted_sum_of_hiddens_deps = torch.bmm(word_h, marginal_rel_probs).transpose(1, 2) # batch, deps, hiddens
            weighted_sum_of_hiddens_heads = torch.bmm(word_h, marginal_rel_probs.permute(0, 2, 1)).transpose(1, 2) # batch, heads, hiddens

            weighted_sum_of_labels_deps = torch.sum(weighted_sum_of_labels, dim=1)
            weighted_sum_of_labels_heads = torch.sum(weighted_sum_of_labels, dim=2)

            weighted_sum_m_deps = torch.cat((weighted_sum_of_hiddens_deps, weighted_sum_of_labels_deps), dim=2) # batch, deps, hiddens
            weighted_sum_m_heads = torch.cat((weighted_sum_of_hiddens_heads, weighted_sum_of_labels_heads), dim=2) # batch, heads, hiddens
            # print(weighted_sum_m_deps.shape)
            length = weighted_sum_m_deps.shape[1]
            ave_prob = 1 / length

            dep_fw_bw = torch.full((length, length), ave_prob).to(weighted_sum_m_deps.device)

            out_fw = weighted_sum_m_heads
            out_bw = weighted_sum_m_deps

            out_fw = self.gcn_fw[i](out_fw, dep_fw_bw)
            out_bw = self.gcn_bw[i](out_bw, dep_fw_bw)

            out = torch.cat([out_fw, out_bw], dim=2)
            word_h = self.dp(out)
            # print(word_h.shape)
        return word_h
