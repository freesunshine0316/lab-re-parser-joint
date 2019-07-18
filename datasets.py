import json, zipfile
from neuronlp2.io import conllx_data
import torch
import bidict
import numpy
def read_mention_ids_and_gold_labels(mention_fn):
    mention_ids = []
    gold_labels = []
    with open(mention_fn) as mfh:
        for line in mfh:
            items = line.strip().split(' ')
            mention_ids.append([int(x) for x in items[:4]])
            gold_labels.append(items[4].lower())
    return mention_ids, gold_labels

def load_word_embedding_vocab(zipped_word_embedding_file):
    words = bidict.bidict()
    with zipfile.ZipFile(zipped_word_embedding_file) as myzip:
        with myzip.open('bioasq.pubmed.vocab', 'r') as vocabfh:
            for index, line in enumerate(vocabfh):
                word = line.strip().decode('utf8')
                words[word] = index
    return words

def load_word_embeddings(zipped_word_embedding_file, embedding_vocab, data_vocab):
    useful_indices = [-1]
    for word in data_vocab.instances:
        if word in embedding_vocab:
            useful_indices.append(embedding_vocab[word])
        elif word.lower() in embedding_vocab:
            useful_indices.append(embedding_vocab[word.lower()])
        else:
            useful_indices.append(-1)
    vals = numpy.array(useful_indices)
    embeddings = torch.nn.Embedding(len(useful_indices), 200) # 200 is the size of the bio embeddings
    with zipfile.ZipFile(zipped_word_embedding_file) as myzip:
        with myzip.open('bioasq.pubmed.200d.txt', 'r') as embfh:
            for index, line in enumerate(embfh):
                ii = numpy.where(vals == index)[0]
                if len(ii) > 0:
                    this_embedding = torch.tensor([float(x) for x in line.decode('utf8').strip().split(' ')])
                    for i_index in ii:
                        embeddings.weight.data[i_index].copy_(this_embedding)
    return embeddings


DATASET_FILES = {
    'pgr':['pgr/train.conllx', 'pgr/dev.conllx','pgr/test.conllx', 'pgr/train.mention.and.gold', 'pgr/dev.mention.and.gold'
           , 'pgr/test.mention.and.gold'],
    'cpr': ['cpr/train.conllx', 'cpr/dev.conllx', 'cpr/test.conllx', 'cpr/train.mention.and.gold', 'cpr/dev.mention.and.gold',
            'cpr/test.mention.and.gold']
}


def load_data(word_alphabets, char_alphabets, pos_alphabet, type_alphabet, custom_args):
    parser_training_fn, parser_dev_fn, parser_test_fn, training_mention_id_and_gold, dev_mention_id_and_gold,\
    test_mention_id_and_gold = DATASET_FILES[custom_args.dataset_name]

    word_alphabet, graph_word_alphabet = word_alphabets
    char_alphabet, graph_char_alphabet = char_alphabets

    data_train = conllx_data.read_data_to_tensor_dicts(parser_training_fn, word_alphabet, char_alphabet, pos_alphabet,
                                                 type_alphabet,
                                                 symbolic_root=True, device=custom_args.device)
    data_dev = conllx_data.read_data_to_tensor_dicts(parser_dev_fn, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                               symbolic_root=True, device=custom_args.device)
    data_test = conllx_data.read_data_to_tensor_dicts(parser_test_fn, word_alphabet, char_alphabet, pos_alphabet,
                                                type_alphabet,
                                                symbolic_root=True, device=custom_args.device)

    graph_data_train = conllx_data.read_data_to_tensor_dicts(parser_training_fn, graph_word_alphabet, graph_char_alphabet, pos_alphabet,
                                                 type_alphabet,
                                                 symbolic_root=True, device=custom_args.device, normalize_digits=False)
    graph_data_dev = conllx_data.read_data_to_tensor_dicts(parser_dev_fn, graph_word_alphabet, graph_char_alphabet, pos_alphabet, type_alphabet,
                                               symbolic_root=True, device=custom_args.device, normalize_digits=False)
    graph_data_test = conllx_data.read_data_to_tensor_dicts(parser_test_fn, graph_word_alphabet, graph_char_alphabet, pos_alphabet,
                                                type_alphabet,
                                                symbolic_root=True, device=custom_args.device, normalize_digits=False)

    train_mentions, train_labels = read_mention_ids_and_gold_labels(training_mention_id_and_gold)
    dev_mentions, dev_labels = read_mention_ids_and_gold_labels(dev_mention_id_and_gold)
    test_mentions, test_labels = read_mention_ids_and_gold_labels(test_mention_id_and_gold)

    return (data_train, data_dev, data_test), \
           (graph_data_train, graph_data_dev, graph_data_test), \
           (train_mentions, dev_mentions, test_mentions), \
           (train_labels, dev_labels, test_labels)
