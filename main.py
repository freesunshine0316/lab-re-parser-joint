import json, os, copy
import torch
from torch import nn
import sys
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--joint-training', default=False, action='store_true')
parser.add_argument('--base-lr', default=1e-6, type=float)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--training-epochs', default=20, type=int)
parser.add_argument('--unk-p', default=.2, type=float)
parser.add_argument('--rel-model', choices=['memory', 'graph_conv']) # memory rel network

parser.add_argument('--graph-conv-num-filters', default=200, type=int)
parser.add_argument('--filter-factory-hidden', default=300, type=int)

parser.add_argument('--rel-model-dp', default=0.3, type=float)
parser.add_argument('--memory-energy-threshold', default=1e-6, type=float)

parser.add_argument('--base-encoder-char-emb-size', default=50, type=int)
parser.add_argument('--base-encoder-hidden-size', default=300, type=int)
parser.add_argument('--base-encoder-no-word-emb-tuning', default=False, action='store_true')
parser.add_argument('--energy-temp', default=1., type=float) # smaller for peakier distribution
parser.add_argument('--base-encoder-num-filters', default=300, type=int)
parser.add_argument('--base-encoder-kernel-size', default=3, type=int)
parser.add_argument('--base-encoder-dp', default=0.3, type=float)
parser.add_argument('--dep-rel-emb-size', default=15, type=int)
parser.add_argument('--positive-alpha', default=1, type=float)
parser.add_argument('--eval-test', default=False, action='store_true')
parser.add_argument('--saved-folder', default='default')

parser.add_argument('--device', default='cuda')
parser.add_argument('--dataset_name', default='cpr')
parser.add_argument('--bio-embeddings', default='bioasq.zip')
parser.add_argument('--graph-alphabet-folder', default='graph_alphabets')

parser.add_argument('--debug', default=False, action='store_true')

custom_args = parser.parse_args()

saved_folder = os.path.join('saved_models', custom_args.saved_folder)

if os.path.exists(saved_folder):
    shutil.rmtree(saved_folder)
os.mkdir(saved_folder)

logging_file = os.path.join(saved_folder, 'log.txt')
logging_file_detail = os.path.join(logging_file+'.detail')

logger = open(logging_file, 'w')
logger_detail = open(logging_file_detail, 'w')

print(custom_args, file=logger)
print(' '.join(sys.argv), file=logger_detail)

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
alphabet_path = custom_args.graph_alphabet_folder
train_path, dev_path, test_path, _, _, _ = datasets.DATASET_FILES[custom_args.dataset_name]
if custom_args.bio_embeddings != 'none':
    embedding_vocab_dict = datasets.load_word_embedding_vocab(custom_args.bio_embeddings)
else:
    embedding_vocab_dict = None
graph_word_alphabet, graph_char_alphabet, _, _ = conllx_data.create_alphabets(alphabet_path, train_path,
                                                                                         data_paths=[dev_path, test_path],
                                                                                         max_vocabulary_size=100000,
                                                                                         embedd_dict=embedding_vocab_dict,
                                                                                         normalize_digits=False, suffix='graph_')

parser_data, graph_data, mentions, labels = datasets.load_data((word_alphabet, graph_word_alphabet),
                                                               (char_alphabet, graph_char_alphabet), pos_alphabet, type_alphabet,
                                                                     custom_args)
data_train, data_dev, data_test = parser_data
graph_data_train, graph_data_dev, graph_data_test = graph_data
mentions_train, mentions_dev, mentions_test = mentions
labels_train, labels_dev, labels_test = labels

all_labels = list(set(x.lower() for x in set(labels_train + labels_dev + labels_test)))
all_labels.sort()

if custom_args.dataset_name == 'cpr':
    negative_label = 'none'
else:
    negative_label = 'false'

if negative_label != all_labels[0]:
    all_labels.remove(negative_label)
    all_labels.insert(0, negative_label)
print(all_labels, file=logger)
label_mapper = {}
for l_index, l in enumerate(all_labels):
    label_mapper[l] = l_index
labels_train = [label_mapper[x.lower()] for x in labels_train]
labels_dev = [label_mapper[x.lower()] for x in labels_dev]
labels_test = [label_mapper[x.lower()] for x in labels_test]

if custom_args.bio_embeddings != 'none':
    word_embs = datasets.load_word_embeddings(custom_args.bio_embeddings, embedding_vocab_dict, graph_word_alphabet)
else:
    word_embs = torch.nn.Embedding(graph_word_alphabet.size(), 200)

if custom_args.base_encoder_no_word_emb_tuning:
    for param in word_embs.parameters():
        param.requires_grad = False

pos_embs = copy.deepcopy(network.pos_embedd)
num_dep_rels = type_alphabet.size()
# word_embs = network.word_embedd
char_embs = nn.Embedding(num_embeddings=graph_char_alphabet.size(), embedding_dim=custom_args.base_encoder_char_emb_size)

mxl = 100
base_encoder = BaseEncoder(hid_size=custom_args.base_encoder_hidden_size, emb_pos=pos_embs, emb_word=word_embs, emb_char=char_embs,
                           kernel_size=custom_args.base_encoder_kernel_size, num_filters=custom_args.base_encoder_num_filters,
                           num_rnn_encoder_layers=3, p_rnn=(0.33, 0.33), dp=custom_args.base_encoder_dp)
if custom_args.rel_model == 'memory':
    graphrel_net = MemoryRel(num_dep_rels, dep_rel_emb_size=custom_args.dep_rel_emb_size, in_size=custom_args.base_encoder_hidden_size,
                             energy_threshold=custom_args.memory_energy_threshold)
elif custom_args.rel_model == 'graph_conv':
    graphrel_net = FullGraphRel(num_dep_rels, dep_rel_emb_size=custom_args.dep_rel_emb_size, in_size=custom_args.base_encoder_hidden_size,
                                num_filters=custom_args.graph_conv_num_filters,
                                filter_factory_hidden=custom_args.filter_factory_hidden, dp=custom_args.rel_model_dp)
else:
    graphrel_net = GCNRel(mxl, 2, num_dep_rels, hid_size=custom_args.base_encoder_hidden_size, dp=custom_args.rel_model_dp)

if custom_args.rel_model != 'graph_conv':
    classifier = MeanPoolClassifier(custom_args.base_encoder_hidden_size*4, n_classes=len(all_labels))
else:
    classifier = MeanPoolClassifier(custom_args.graph_conv_num_filters, n_classes=len(all_labels))

total_net = RelNetwork(network, base_encoder, graphrel_net, classifier, energy_temp=custom_args.energy_temp)
total_net = total_net.to(custom_args.device)

optimizers = []
if not custom_args.joint_training:
    optimizers.append(torch.optim.SGD(network.parameters(), lr=custom_args.base_lr))
    optimizers.append( torch.optim.Adam([x for x in list(classifier.parameters())+list(graphrel_net.parameters())
                                     +list(base_encoder.parameters()) if x.requires_grad ]))
else:
    optimizers.append( torch.optim.Adam([x for x in list(classifier.parameters())+list(graphrel_net.parameters())
                                     +list(base_encoder.parameters())+list(network.parameters()) if x.requires_grad]))

# network.eval()
training_epochs = custom_args.training_epochs

batch_size = custom_args.batch_size

label_weights = torch.full((len(all_labels),), custom_args.positive_alpha).to(custom_args.device)
label_weights[0] = 1
loss_func = torch.nn.NLLLoss(weight=label_weights)
# training loop
best_dev_f1 = 0
for tepoch in range(training_epochs):
    total_net.train()

    print('training epoch {}'.format(tepoch), file=logger_detail)
    print('training epoch {}'.format(tepoch), file=logger)

    # train_true_pos, dev_true_pos, test_true_pos = 0, 0, 0
    # train_pred_pos, dev_pred_pos, test_pred_pos = 0, 0, 0
    # train_match_pos, dev_match_pos, test_match_pos = 0, 0, 0
    train_total, dev_total, test_total = 0, 0, 0
    train_corr, dev_corr, test_corr = 0, 0, 0
    predicted_labels = []
    gold_labels = []
    predicted_labels_with_none = []
    gold_labels_with_none = []
    i = 0
    train_loss = 0
    for batch, graph_batch in conllx_data.doubly_iterate_batch_tensors_dicts(data_train, graph_data_train, batch_size, shuffle=True):
        word, char, pos, heads, types, masks, lengths, indices = batch
        instances = []
        this_labels = []
        for index in indices:
            instances.append(mentions_train[index.item()])
            this_labels.append(labels_train[index.item()])
        this_labels = torch.tensor(this_labels)

        graph_words = graph_batch[0]
        graph_words = datasets.unk_single_mentions(graph_words, instances, custom_args.unk_p)
        graph_batch = (graph_words, *(graph_batch[1:]))

        pred = total_net.forward(batch, graph_batch, instances)


        train_total += len(instances)
        predicted_label = torch.argmax(pred, dim=1).cpu()

        predicted_labels_with_none.extend(predicted_label.tolist())
        gold_labels_with_none.extend(this_labels.tolist())

        this_true = (predicted_label == this_labels).sum().item()
        train_corr += this_true

        for label_index in range(len(this_labels)):
            if this_labels[label_index] != 0 or predicted_label[label_index] != 0:
                gold_labels.append(this_labels[label_index].item())
                predicted_labels.append(predicted_label[label_index].item())

        loss = loss_func(pred, this_labels.to(pred.device))

        loss.backward()
        train_loss += loss.item()
        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad()
        print('epoch {} iter {} | loss {} | corr? {}'.format(tepoch, i, loss.item(), this_true), file=logger_detail)
        i += 1
        if custom_args.debug and i > 10: break
    prec, rec, f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
    print('>> train | loss {:.4f} | acc {:.4f} | prec {:.4f} rec {:.4f} f1 {:.4f}'.format(train_loss, train_corr / train_total, prec,
                                                                          rec, f1), file=logger)
    print('>> train | loss {:.4f} | acc {:.4f} | prec {:.4f} rec {:.4f} f1 {:.4f}'.format(train_loss, train_corr / train_total, prec,
                                                                          rec, f1), file=logger_detail)
    print(get_eval_metrics(predicted_labels_with_none, gold_labels_with_none), file=logger)

    with torch.no_grad():
        # dev
        gold_labels = []
        predicted_labels = []
        dev_loss = 0
        total_net.eval()
        predicted_labels_with_none = []
        gold_labels_with_none = []
        for batch, graph_batch in conllx_data.doubly_iterate_batch_tensors_dicts(data_dev, graph_data_dev, batch_size):
            word, char, pos, heads, types, masks, lengths, indices = batch
            # else:
            instances = []
            this_labels = []
            for index in indices:
                instances.append(mentions_dev[index.item()])
                this_labels.append(labels_dev[index.item()])
            this_labels = torch.tensor(this_labels)
            pred = total_net.forward(batch, graph_batch, instances)

            dev_total += len(instances)
            predicted_label = torch.argmax(pred, dim=1).cpu()

            predicted_labels_with_none.extend(predicted_label.tolist())
            gold_labels_with_none.extend(this_labels.tolist())

            this_true = (predicted_label == this_labels).sum().item()
            dev_corr += this_true

            for label_index in range(len(this_labels)):
                if this_labels[label_index] != 0 or predicted_label[label_index] != 0:
                    gold_labels.append(this_labels[label_index].item())
                    predicted_labels.append(predicted_label[label_index].item())

            loss = loss_func(pred, this_labels.to(pred.device))
            dev_loss += loss.item()
            if custom_args.debug and dev_total > 10: break
        prec, rec, f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
        print('>> dev | loss {:.4f} | acc {:.4f} | prec {:.4f} rec {:.4f} f1 {:.4f}'.format(dev_loss, dev_corr / dev_total, prec, rec, f1), file=logger_detail)
        print('>> dev | loss {:.4f} | acc {:.4f} | prec {:.4f} rec {:.4f} f1 {:.4f}'.format(dev_loss, dev_corr / dev_total, prec, rec, f1), file=logger)
        print(get_eval_metrics(predicted_labels_with_none, gold_labels_with_none), file=logger)
        if f1 > best_dev_f1:
            fn = 'e{}.tch'.format(tepoch)
            full_fn = os.path.join(saved_folder, fn)
            if tepoch > 10:
                torch.save(total_net.state_dict(), full_fn)
            best_dev_f1 = f1

        if custom_args.eval_test:
            gold_labels = []
            predicted_labels = []
            # test_preds = ['X'] * sum(data_test[1])
            predicted_labels_with_none = []
            gold_labels_with_none = []
            for batch, graph_batch in conllx_data.doubly_iterate_batch_tensors_dicts(data_test, graph_data_test, batch_size):
                word, char, pos, heads, types, masks, lengths, indices = batch
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
                if custom_args.debug and test_total > 10: break
            prec, rec, f1, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
            print('>> test acc {:.4f} | prec {:.4f} rec {:.4f} f1 {:.4f}'.format(test_corr / test_total, prec, rec, f1), file=logger)
            print('>> test acc {:.4f} | prec {:.4f} rec {:.4f} f1 {:.4f}'.format(test_corr / test_total, prec, rec, f1), file=logger_detail)
            print(get_eval_metrics(predicted_labels_with_none, gold_labels_with_none), file=logger)
            # print(' '.join([str(x.cpu().item()) if not isinstance(x, str) else x for x in test_preds]), file=logger)
            # print(' '.join([str(x.cpu().item()) if not isinstance(x, str) else x for x in test_preds]), file=logger_detail)

    logger.flush()
    logger_detail.flush()
logger.close()
logger_detail.flush()