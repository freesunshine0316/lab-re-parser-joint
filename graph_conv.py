import torch
from torch import nn

class FilterFactory(nn.Module):
    def __init__(self, in_size, hidden_size, num_filters, filter_width, filter_height, filter_depth, bias=True, dropout=0.5):
        super(FilterFactory, self).__init__()
        self.dp = nn.Dropout(dropout)
        self.input_transform = nn.Linear(in_size, hidden_size)
        factory_weight1 = torch.rand(num_filters, filter_depth, filter_height, filter_width, hidden_size)
        torch.nn.init.kaiming_normal_(factory_weight1)
        self.factory_weight1 = nn.Parameter(factory_weight1)
        if bias:
            factory_bias1 = torch.rand(num_filters, filter_depth, filter_height, filter_width)
            torch.nn.init.kaiming_normal_(factory_bias1)
            self.factory_bias1 = nn.Parameter(factory_bias1)
        else:
            self.factory_bias1 = 0
        self.actf = torch.nn.LeakyReLU()

    def forward(self, inputs):
        inputs = self.actf(self.input_transform(inputs))
        output1 = self.factory_weight1 @ inputs + self.factory_bias1
        output1 = self.dp(self.actf(output1))
        return output1

class FullGraphRel(nn.Module):
    def __init__(self, num_dep_rels, dep_rel_emb_size=15, in_size=256, dropout=0.5
                 , num_filters=200, filter_size=(3, 3), filter_factory_hidden=500):
        super(FullGraphRel, self).__init__()
        self.arc_composer = nn.Linear(in_size*4 + dep_rel_emb_size, in_size) # head rel dep
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filter_depth = in_size*2
        self.filter_factory = FilterFactory(in_size*4, filter_factory_hidden, num_filters, filter_size[0], filter_size[1], in_size)
        rel_embs = torch.rand(num_dep_rels, dep_rel_emb_size) # the weight, not the embedding module
        torch.nn.init.kaiming_normal_(rel_embs)
        self.rel_embs = torch.nn.Parameter(rel_embs)
        self.actf = nn.LeakyReLU()
        self.dp = nn.Dropout(dropout)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.input_transform = nn.Linear(in_size*4, num_filters)

    def forward(self, energy, word_h, e1, e2, sent_len):

        mem_bank = self.get_arc_representations(word_h, energy, sent_len)
        output = self.conv_graph(e1, e2, mem_bank)
        return output

    def get_arc_representations(self, word_representations, energy, sent_len):
        # only has values, can be augmented with keys. no query needed
        word_representations = word_representations.squeeze(0)[:sent_len]
        energy = energy.squeeze(0)
        energy = energy[:, :sent_len, :sent_len]
        marginal_energy = energy.sum(dim=0)
        rel_tensor = energy.permute(1, 2, 0) @ self.rel_embs # head dep vec

        head_tensor = word_representations.unsqueeze(0) * marginal_energy.unsqueeze(2) # head dep vec
        dep_tensor = (word_representations.unsqueeze(0) * marginal_energy.t().unsqueeze(2)).transpose(0, 1) # head dep vec
        all_tensors = torch.cat([head_tensor, rel_tensor, dep_tensor], dim=2)
        arc_reps = self.arc_composer(all_tensors)
        arc_reps = self.actf(arc_reps)
        arc_reps = self.dp(arc_reps).permute(2, 0, 1) # vec, dep head
        return arc_reps

    def conv_graph(self, entity1, entity2, mem_bank):
        # assume entity 1 and 2 have only 1 dim
        inputs = torch.cat([entity1, entity2], dim=0)
        filter = self.filter_factory(inputs)
        mem_bank = mem_bank.unsqueeze(0) # 1, vec, dep, head
        result = nn.functional.conv2d(mem_bank, filter)
        result = self.dp(self.actf(result))
        result = result.reshape(1, self.num_filters, -1)
        result = self.maxpool(result).squeeze()
        transformed_input = self.dp(self.actf(self.input_transform(inputs)))
        return result + transformed_input
