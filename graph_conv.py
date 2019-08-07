import torch
from torch import nn

class FilterFactory(nn.Module):
    def __init__(self, in_size, hidden_size, num_filters, filter_width, filter_height, filter_depth, bias=True, dp=0.5):
        super(FilterFactory, self).__init__()
        self.dp = nn.Dropout(dp)
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
        output1 = (self.factory_weight1 @ inputs.t()).permute(4, 0, 1, 2, 3) + self.factory_bias1
        output1 = self.dp(self.actf(output1))
        return output1

class FullGraphRel(nn.Module):
    def __init__(self, num_dep_rels, dep_rel_emb_size=15, in_size=256, dp=0.5, num_filters=200, filter_size=(3, 3),
                 filter_factory_hidden=300, shared_conv_flag=False, shared_conv_filters=0, energy_threshold=0.):
        super(FullGraphRel, self).__init__()
        self.arc_composer = nn.Linear(in_size*4 + dep_rel_emb_size, in_size) # head rel dep
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filter_factory = FilterFactory(in_size*4, filter_factory_hidden, num_filters, filter_size[0], filter_size[1],
                                            in_size, dp=dp)
        self.shared_conv_flag = shared_conv_flag
        self.num_shared_filters = shared_conv_filters
        if self.shared_conv_flag:
            self.shared_conv_filters = torch.rand(self.num_shared_filters, in_size,  filter_size[0], filter_size[1])
            torch.nn.init.kaiming_normal_(self.shared_conv_filters)
            self.shared_conv_filters = nn.Parameter(self.shared_conv_filters)
            self.conv_output_transform = nn.Linear(self.num_filters + self.num_shared_filters,
                                                   self.num_filters + self.num_shared_filters)
        rel_embs = torch.rand(num_dep_rels, dep_rel_emb_size) # the weight, not the embedding module
        torch.nn.init.kaiming_normal_(rel_embs)
        self.rel_embs = torch.nn.Parameter(rel_embs)
        self.actf = nn.LeakyReLU()
        self.dp = nn.Dropout(dp)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.input_transform = nn.Linear(in_size*4, self.num_filters+self.num_shared_filters)
        self.energy_thres = energy_threshold

    def forward(self, energy, word_h, e1, e2, sent_len):
        energy = energy.masked_fill(energy < self.energy_thres, 0.0)
        mem_bank = self.get_arc_representations(word_h, energy, sent_len)
        output = self.conv_graph(e1, e2, mem_bank)
        return output

    def get_arc_representations(self, word_representations, energy, sent_len):
        # only has values, can be augmented with keys. no query needed
        # word_representations = word_representations.squeeze(0)[:sent_len]
        # energy = energy.squeeze(0) # batch, rel, head, dep
        # energy = energy[:, :sent_len, :sent_len]
        marginal_energy = energy.sum(dim=1) # batch, head, dep
        rel_tensor = energy.permute(0, 2, 3, 1) @ self.rel_embs # head dep vec
        # word rep: batch word emb * batch, head, dep
        dep_head_tensors = combine_two_tensors(word_representations, word_representations)
        dep_head_tensors = dep_head_tensors * marginal_energy.unsqueeze(-1)
        all_tensors = torch.cat([dep_head_tensors, rel_tensor], dim=-1)
        arc_reps = self.arc_composer(all_tensors)
        arc_reps = self.actf(arc_reps)
        arc_reps = self.dp(arc_reps).permute(0, 3, 1, 2) # batch, vec, dep, head
        return arc_reps

    def conv_graph(self, entity1, entity2, mem_bank):
        # assume entity 1 and 2 have only 1 dim
        inputs = torch.cat([entity1, entity2], dim=-1)
        filter = self.filter_factory(inputs)
        results = []
        if self.shared_conv_flag:
            shared_results = nn.functional.conv2d(mem_bank, self.shared_conv_filters)
        for i in range(filter.shape[0]):
            results.append( nn.functional.conv2d(mem_bank[i].unsqueeze(0), filter[i]))
        results = torch.stack(results, dim=0)
        result = self.dp(self.actf(results))

        result = result.reshape(inputs.shape[0], self.num_filters, -1)
        result = self.maxpool(result).squeeze(-1)

        if self.shared_conv_flag:
            shared_results = self.dp(self.actf(shared_results))
            shared_results = shared_results.reshape(inputs.shape[0], self.num_shared_filters, -1)
            shared_results = self.maxpool(shared_results).squeeze(-1)
            result = torch.cat([shared_results, result], dim=-1)
        # transformed_input = self.dp(self.actf(self.input_transform(inputs)))
            result = self.conv_output_transform(result)
            result = self.dp(self.actf(result))
        transformed_input = self.dp(self.actf(self.input_transform(inputs)))
        transformed_input = transformed_input + result
        return transformed_input

def combine_two_tensors(X, Y):
    X1 = X.unsqueeze(1)
    Y1 = Y.unsqueeze(2)
    X2 = X1.repeat(1, Y.shape[1], 1, 1)
    Y2 = Y1.repeat(1, 1, X.shape[1], 1)
    Z = torch.cat([X2, Y2], -1)
    Z = Z.view(X.shape[0], X.shape[1], X.shape[1], Z.shape[-1])
    return Z