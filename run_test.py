import json, os, copy
import torch
from torch import nn
import sys
import argparse
from argparse import Namespace
import shutil
sys.path.append('./biaffine')
from neuronlp2.io import Alphabet
from neuronlp2.models import BiRecurrentConvBiAffine
from graphrel_p2 import GCNRel, MeanPoolClassifier, RelNetwork, BaseEncoder
from memoryrel import MemoryRel
from graph_conv import FullGraphRel
import datasets
from neuronlp2.io import conllx_data
from sklearn.metrics import precision_recall_fscore_support
from eval_helper import get_eval_metrics
import tacred_scorer


parser = argparse.ArgumentParser()

parser.add_argument('--config-file', required=True)
parser.add_argument('--model-param-file', required=True)
parser.add_argument('--get-parses', action='store_true', default=False)
parser.add_argument('--need-before',  action='store_true', default=False)

main_args = parser.parse_args()

for arg in main_args.__dict__:
    print('{} = {}'.format(arg, main_args.__dict__[arg]))

with open(main_args.config_file, 'r', encoding='utf8') as cfh:
    custom_args = eval(cfh.readline().strip())
custom_args.memory_energy_threshold = 0.
print('='*30)
for arg in custom_args.__dict__:
    print('{} = {}'.format(arg, custom_args.__dict__[arg]))

trained_state_dict = torch.load(main_args.model_param_file)

pretrained_dependency_parser_fn = 'biaffine/models/biaffine/network.pt'
pretrained_dependency_parser_params_fn = 'biaffine/models/biaffine/network.pt.arg.json'
pretrained_dependency_parser_vocab_fn = 'biaffine/models/biaffine/alphabets'

with open(pretrained_dependency_parser_params_fn) as parser_param_fh:
    params = json.load(parser_param_fh)
args= params['args']
kwargs = params['kwargs']

# print(len(tr), len(ts))

char_alphabet = Alphabet('character', defualt_value=True)
pos_alphabet = Alphabet('pos')
type_alphabet = Alphabet('type')
word_alphabet = Alphabet('word', defualt_value=True, singleton=True)

parser = torch.load(pretrained_dependency_parser_fn, map_location=custom_args.device)

network = BiRecurrentConvBiAffine(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
                                  args[8], args[9], args[10], args[11], args[12], args[13],
                                  embedd_word=None, embedd_char=None,
                                  p_in=kwargs['p_in'], p_out=kwargs['p_out'], p_rnn=kwargs['p_rnn'],
                                  biaffine=kwargs['biaffine'], pos=kwargs['pos'], char=kwargs['char'])

char_alphabet.load(pretrained_dependency_parser_vocab_fn)
pos_alphabet.load(pretrained_dependency_parser_vocab_fn)
type_alphabet.load(pretrained_dependency_parser_vocab_fn)
word_alphabet.load(pretrained_dependency_parser_vocab_fn)

# char_alphabet.keep_growing = True
# word_alphabet.keep_growing = True

network.load_state_dict(parser)

#------------------------create data specific alphabets

print("Creating graph classifier Alphabets")
alphabet_path = os.path.join(custom_args.graph_alphabet_folder, custom_args.dataset_name)
train_path, dev_path, test_path, _, _, _ = datasets.DATASET_FILES[custom_args.dataset_name]
if custom_args.bio_embeddings != 'none':
    if 'bio' in custom_args.bio_embeddings:
        embedding_vocab_dict = datasets.load_bio_word_embedding_vocab(custom_args.bio_embeddings)
    elif 'glove' in custom_args.bio_embeddings:
        embedding_vocab_dict = datasets.load_glove_word_embedding_vocab(custom_args.bio_embeddings)
else:
    embedding_vocab_dict = None
graph_word_alphabet, graph_char_alphabet, _, _, graph_ner_alphabet = conllx_data.create_alphabets(alphabet_path, train_path,
                                                                                         data_paths=[dev_path, test_path],
                                                                                         max_vocabulary_size=100000,
                                                                                         embedd_dict=embedding_vocab_dict,
                                                                                         normalize_digits=False, suffix='graph_')

parser_data, graph_data, mentions, labels = datasets.load_data((word_alphabet, graph_word_alphabet),
                                                               (char_alphabet, graph_char_alphabet), pos_alphabet, type_alphabet,
                                                                graph_ner_alphabet, custom_args)
data_train, data_dev, data_test = parser_data
graph_data_train, graph_data_dev, graph_data_test = graph_data
mentions_train, mentions_dev, mentions_test = mentions
labels_train, labels_dev, labels_test = labels

all_labels = list(set(x.lower() for x in set(labels_train + labels_dev + labels_test)))
all_labels.sort()

if custom_args.dataset_name == 'cpr':
    negative_label = 'none'
elif custom_args.dataset_name == 'pgr':
    negative_label = 'false'
elif custom_args.dataset_name == 'tacred':
    negative_label = 'no_relation'
elif custom_args.dataset_name == 'semeval':
    negative_label = 'other'

if negative_label != all_labels[0]:
    all_labels.remove(negative_label)
    all_labels.insert(0, negative_label)

label_mapper = {}
for l_index, l in enumerate(all_labels):
    label_mapper[l] = l_index
labels_train = [label_mapper[x.lower()] for x in labels_train]
labels_dev = [label_mapper[x.lower()] for x in labels_dev]
labels_test = [label_mapper[x.lower()] for x in labels_test]

if custom_args.bio_embeddings != 'none':
    if 'bio' in custom_args.bio_embeddings:
        word_embs, grad_mask = datasets.load_bio_word_embeddings(custom_args.bio_embeddings,
                                                             embedding_vocab_dict, graph_word_alphabet)
    elif 'glove' in custom_args.bio_embeddings:
        word_embs, grad_mask = datasets.load_glove_word_embeddings(custom_args.bio_embeddings,
                                                            embedding_vocab_dict, graph_word_alphabet)
    if custom_args.base_encoder_train_random_embs:
        grad_mask = grad_mask.to(custom_args.device)
        def masking_grad(gr):
            return gr*grad_mask
        word_embs.weight.register_hook(masking_grad)
else:
    word_embs = torch.nn.Embedding(graph_word_alphabet.size(), 200)

if custom_args.base_encoder_no_word_emb_tuning:
    for param in word_embs.parameters():
        param.requires_grad = False

pos_embs = copy.deepcopy(network.pos_embedd)
num_dep_rels = type_alphabet.size()
# word_embs = network.word_embedd
char_embs = nn.Embedding(num_embeddings=graph_char_alphabet.size(), embedding_dim=custom_args.base_encoder_char_emb_size)
if graph_ner_alphabet.size() > 3:
    ner_embs = nn.Embedding(num_embeddings=graph_ner_alphabet.size(), embedding_dim=custom_args.base_encoder_ner_emb_size)
else:
    ner_embs = None

base_encoder = BaseEncoder(hid_size=custom_args.base_encoder_hidden_size, emb_pos=pos_embs, emb_word=word_embs, emb_char=char_embs, emb_ner=ner_embs,
                           kernel_size=custom_args.base_encoder_kernel_size, num_filters=custom_args.base_encoder_num_filters,
                           num_rnn_encoder_layers=3, p_rnn=(0.33, 0.33), dp=custom_args.base_encoder_dp)

if custom_args.rel_model == 'memory':
    graphrel_net = MemoryRel(num_dep_rels, dep_rel_emb_size=custom_args.dep_rel_emb_size, in_size=custom_args.base_encoder_hidden_size,
                             energy_threshold=custom_args.memory_energy_threshold)
elif custom_args.rel_model == 'graph_conv':
    graphrel_net = FullGraphRel(num_dep_rels, dep_rel_emb_size=custom_args.dep_rel_emb_size, in_size=custom_args.base_encoder_hidden_size,
                                num_filters=custom_args.private_conv_filters,
                                filter_factory_hidden=custom_args.filter_factory_hidden, dp=custom_args.rel_model_dp
                                , shared_conv_flag=custom_args.shared_conv_flag, shared_conv_filters=custom_args.shared_conv_filters,
                                energy_threshold=custom_args.memory_energy_threshold)
elif custom_args.rel_model == 'gcn':
    graphrel_net = GCNRel(num_dep_rels, hid_size=custom_args.base_encoder_hidden_size, dp=custom_args.rel_model_dp,
                          energy_thres=custom_args.memory_energy_threshold)
else:
    raise Exception

if custom_args.rel_model != 'graph_conv':
    classifier = MeanPoolClassifier(custom_args.base_encoder_hidden_size*4, n_classes=len(all_labels))
else:
    classifier = MeanPoolClassifier(custom_args.shared_conv_filters+custom_args.private_conv_filters, n_classes=len(all_labels))

total_net = RelNetwork(network, base_encoder, graphrel_net, classifier, energy_temp=custom_args.energy_temp, config=custom_args,
                       num_dep_rels=num_dep_rels)

if main_args.need_before:
    no_train_parses_file = main_args.config_file.replace('log.txt', 'before_finetune.parses')
    no_train_parses = {}
    with torch.no_grad():
        network = network.to(custom_args.device)
        network.eval()
        for batch, graph_batch in conllx_data.doubly_iterate_batch_tensors_dicts(data_test, graph_data_test,
                                                                                 custom_args.batch_size):
            word, char, pos, ners, heads, types, masks, lengths, indices = batch
            _, one_best_tree = network.get_probs(word, char, pos, mask=masks, length=lengths,
                                                          use_scores=False, get_mst_tree=True)
            parse_heads, parse_types = one_best_tree
            for instance_index in range(parse_heads.shape[0]):
                single_parse = []
                index = indices[instance_index].item()
                for dep in range(parse_heads.shape[1]):
                    this_head = parse_heads[instance_index][dep]
                    this_type = parse_types[instance_index][dep]
                    single_parse.append((dep, this_head, this_type))
                no_train_parses[index] = single_parse
        with open(no_train_parses_file, 'w') as f:
            for i in range(len(no_train_parses)):
                for dep, head, type in no_train_parses[i]:
                    print("{} {} {}".format(dep, head, type_alphabet.get_instance(type)), file=f)
                print('', file=f)

total_net.load_state_dict(trained_state_dict)
total_net = total_net.to(custom_args.device)

batch_size = custom_args.batch_size

train_total, dev_total, test_total = 0, 0, 0
train_corr, dev_corr, test_corr = 0, 0, 0
predicted_labels = []
gold_labels = []
predicted_labels_with_none = []
gold_labels_with_none = []
i = 0
train_loss = 0
ref_num_no_none = 0
out_num_no_none = 0
corr_num_no_none = 0

with torch.no_grad():
    total_net.eval()

    if main_args.get_parses:
        with_train_parses_file = main_args.config_file.replace('log.txt', 'after_finetune.parses')
        with_train_parses = {}
        with torch.no_grad():
            network.eval()
            for batch, graph_batch in conllx_data.doubly_iterate_batch_tensors_dicts(data_test, graph_data_test,
                                                                                     custom_args.batch_size):
                word, char, pos, ners, heads, types, masks, lengths, indices = batch
                energy, one_best_tree = network.get_probs(word, char, pos, mask=masks, length=lengths,
                                                     use_scores=False, get_mst_tree=True)
                for dep in range(energy[0].shape[-1]):
                    print(energy[0, ...,dep].tolist())
                exit()
                parse_heads, parse_types = one_best_tree
                for instance_index in range(parse_heads.shape[0]):
                    single_parse = []
                    index = indices[instance_index].item()
                    for dep in range(parse_heads.shape[1]):
                        this_head = parse_heads[instance_index][dep]
                        this_type = parse_types[instance_index][dep]
                        single_parse.append((dep, this_head, this_type))
                    with_train_parses[index] = single_parse
            with open(with_train_parses_file, 'w') as f:
                for i in range(len(with_train_parses)):
                    for dep, head, type in with_train_parses[i]:
                        print("{} {} {}".format(dep, head, type_alphabet.get_instance(type)), file=f)
                    print('', file=f)

    for batch, graph_batch in conllx_data.doubly_iterate_batch_tensors_dicts(data_test, graph_data_test, batch_size):
        word, char, pos, ners, heads, types, masks, lengths, indices = batch
        instances = []
        this_labels = []
        for index in indices:
            instances.append(mentions_test[index.item()])
            this_labels.append(labels_test[index.item()])
        this_labels = torch.tensor(this_labels)
        pred = total_net.forward(batch, graph_batch, instances)

        test_total += len(instances)
        predicted_label = torch.argmax(pred, dim=1).cpu()

        predicted_labels_with_none.extend(predicted_label.tolist())
        gold_labels_with_none.extend(this_labels.tolist())

        this_true = (predicted_label == this_labels).sum().item()
        test_corr += this_true

        for label_index in range(len(this_labels)):
            if this_labels[label_index] != 0 or predicted_label[label_index] != 0:
                gold_labels.append(this_labels[label_index].item())
                predicted_labels.append(predicted_label[label_index].item())
            if this_labels[label_index] != 0:
                ref_num_no_none += 1
            if predicted_label[label_index] != 0:
                out_num_no_none += 1
                if this_labels[label_index] == predicted_label[label_index]:
                    corr_num_no_none += 1
        if custom_args.debug and test_total > 10: break
    prec, rec, f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    print('>> test acc {:.4f} | prec {:.4f} rec {:.4f} f1 {:.4f}'.format(test_corr / test_total, prec, rec, f1))
    print('>> test acc {:.4f} | prec {:.4f} rec {:.4f} f1 {:.4f}'.format(test_corr / test_total, prec, rec, f1))
    self_prec = 0 if out_num_no_none == 0 else corr_num_no_none / out_num_no_none
    self_rec = 0 if ref_num_no_none == 0 else corr_num_no_none / ref_num_no_none
    self_f1 = 2 * self_prec * self_rec / (self_prec + self_rec + 1e-8)
    print('>> test | SELF EVAL | prec {:.4f} rec {:.4f} f1 {:.4f}'.format(self_prec, self_rec, self_f1))
    print(get_eval_metrics(predicted_labels_with_none, gold_labels_with_none))
    if custom_args.dataset_name == 'tacred':
        self_prec, self_rec, self_f1, _ = tacred_scorer.score(gold_labels_with_none, predicted_labels_with_none)
    elif custom_args.dataset_name == 'semeval' or custom_args.dataset_name == 'semeval_order':
        self_f1 = tacred_scorer.semeval_scorer(gold_labels_with_none, predicted_labels_with_none, all_labels,
                                               custom_args, sys.stdout, test=True)