import torch
from torch import nn

class MemoryRel(nn.Module):
    def __init__(self, num_dep_rels, dep_rel_emb_size=15, hops=3, in_size=256, energy_threshold=1e-6, dropout=0.5
                 , tune_energy_threshold=False):
        super(MemoryRel, self).__init__()
        self.arc_composer = nn.Linear(in_size*4 + dep_rel_emb_size, in_size*2) # head rel dep
        self.hops = hops
        self.key_value_generators = nn.ModuleList()
        self.hidden_generators = nn.ModuleList()
        for i in range(self.hops):
            self.key_value_generators.append(nn.Linear(in_size*4, in_size*4))
            self.hidden_generators.append(nn.Linear(in_size*4, in_size*4))
        self.energy_threshold = energy_threshold
        rel_embs = torch.rand(num_dep_rels, dep_rel_emb_size) # the weight, not the embedding module
        torch.nn.init.kaiming_normal_(rel_embs)
        self.rel_embs = torch.nn.Parameter(rel_embs)
        self.actf = nn.LeakyReLU()
        self.dp = nn.Dropout(dropout)

    def forward(self, energy, word_h, e1, e2, sent_len):
        this_threshold = self.energy_threshold
        for i in range(3):
            filtering_status = self.filter_low_prob_pairs(energy, sent_len, this_threshold)
            if filtering_status.sum() != 0:
                break
            else:
                this_threshold = this_threshold * 0.1
        else:
            raise Exception('cannot find a low enough threshold')
        mem_bank = self.get_arc_representations(word_h, filtering_status, energy, sent_len)
        output = self.hopping(e1, e2, mem_bank)
        return output

    def filter_low_prob_pairs(self, energy, sent_len, this_threshold):
        # energy is batch, label, head, dep
        marginal_energy = energy.sum(dim=1).squeeze()[:sent_len, :sent_len]
        filtering_status = torch.zeros_like(marginal_energy)
        for head in range(marginal_energy.shape[0]):
            for dep in range(marginal_energy.shape[1]):
                if marginal_energy[head, dep] > this_threshold:
                    filtering_status[head, dep] = 1
        return filtering_status

    def get_arc_representations(self, word_representations, filtering_status, energy, sent_len):
        # only has values, can be augmented with keys. no query needed
        word_representations = word_representations.squeeze(0)
        energy = energy.squeeze(0)
        marginal_energy = energy.sum(dim=0)[:sent_len, :sent_len]
        head_indices = []
        dep_indices = []
        rels = []
        for head in range(sent_len):
            for dep in range(sent_len):
                if filtering_status[head, dep] == 1:
                    rels.append(energy[:, head, dep] @ self.rel_embs)
                    head_indices.append(head)
                    dep_indices.append(dep)
        marginal_energy_weights = marginal_energy[head_indices, dep_indices]
        print('number of filtered arcs: {}'.format(len(head_indices)))
        head_tensor = word_representations.index_select(0, torch.tensor(head_indices).to(word_representations.device))* marginal_energy_weights.unsqueeze(1)
        dep_tensor = word_representations.index_select(0, torch.tensor(dep_indices).to(word_representations.device)) * marginal_energy_weights.unsqueeze(1)
        rel_tensor = torch.stack(rels, dim=0)
        all_tensors = torch.cat([head_tensor, rel_tensor, dep_tensor], dim=1)
        arc_reps = self.arc_composer(all_tensors)
        arc_reps = self.dp(self.actf(arc_reps))
        return arc_reps

    def hopping(self, entity1, entity2, mem_bank):
        # assume entity 1 and 2 have only 1 dim
        inputs = torch.cat([entity1, entity2], dim=0)
        for i in range(self.hops):
            key_value = self.key_value_generators[i](inputs)
            # key_value = torch.tanh(key_value)
            key, value = torch.chunk(key_value, 2, dim=0)
            key = torch.tanh(key)
            value = self.actf(value)
            mem_dist = torch.softmax(mem_bank @ key, dim=0)
            mem_representation = mem_dist @ mem_bank
            inputs = self.hidden_generators[i](torch.cat([value, mem_representation], dim=0))
            inputs = self.dp(self.actf(inputs))
        return inputs
