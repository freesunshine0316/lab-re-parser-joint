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

def load_bio_word_embedding_vocab(zipped_word_embedding_file):
    words = bidict.bidict()
    with zipfile.ZipFile(zipped_word_embedding_file) as myzip:
        with myzip.open('bioasq.pubmed.vocab', 'r') as vocabfh:
            for index, line in enumerate(vocabfh):
                word = line.strip().decode('utf8')
                words[word] = index
    return words

def load_glove_word_embedding_vocab(zipped_word_embedding_file):
    words = bidict.bidict()
    with zipfile.ZipFile(zipped_word_embedding_file) as myzip:
        with myzip.open('glove.42B.300d.txt', 'r') as vocabfh:
            for index, line in enumerate(vocabfh):
                word = line.strip().decode('utf8')
                word = word.split(' ')[0]
                words[word] = index
    return words

def load_bio_word_embeddings(zipped_word_embedding_file, embedding_vocab, data_vocab):
    embedding_size = 200
    useful_indices = [-1]
    pretrained_mask = [torch.ones((embedding_size,))]
    for word in data_vocab.instances:
        if word in embedding_vocab:
            useful_indices.append(embedding_vocab[word])
            pretrained_mask.append(torch.zeros((embedding_size,)))
        elif word.lower() in embedding_vocab:
            useful_indices.append(embedding_vocab[word.lower()])
            pretrained_mask.append(torch.zeros((embedding_size,)))
        else:
            useful_indices.append(-1)
            pretrained_mask.append(torch.ones((embedding_size,)))
    vals = numpy.array(useful_indices)
    embeddings = torch.nn.Embedding(len(useful_indices), embedding_size) # 200 is the size of the bio embeddings
    with zipfile.ZipFile(zipped_word_embedding_file) as myzip:
        with myzip.open('bioasq.pubmed.200d.txt', 'r') as embfh:
            for index, line in enumerate(embfh):
                ii = numpy.where(vals == index)[0]
                if len(ii) > 0:
                    this_embedding = torch.tensor([float(x) for x in line.decode('utf8').strip().split(' ')])
                    for i_index in ii:
                        embeddings.weight.data[i_index].copy_(this_embedding)
    pretrained_mask = torch.stack(pretrained_mask, dim=0)
    return embeddings, pretrained_mask

def load_glove_word_embeddings(zipped_word_embedding_file, embedding_vocab, data_vocab):
    embedding_size = 300
    useful_indices = [-1]
    pretrained_mask = [torch.ones((embedding_size,))]
    for word in data_vocab.instances:
        if word in embedding_vocab:
            useful_indices.append(embedding_vocab[word])
            pretrained_mask.append(torch.zeros((embedding_size,)))
        elif word.lower() in embedding_vocab:
            useful_indices.append(embedding_vocab[word.lower()])
            pretrained_mask.append(torch.zeros((embedding_size,)))
        else:
            useful_indices.append(-1)
            pretrained_mask.append(torch.ones((embedding_size,)))
    vals = numpy.array(useful_indices)
    embeddings = torch.nn.Embedding(len(useful_indices), embedding_size) # 200 is the size of the bio embeddings
    with zipfile.ZipFile(zipped_word_embedding_file) as myzip:
        with myzip.open('glove.42B.300d.txt', 'r') as embfh:
            for index, line in enumerate(embfh):
                ii = numpy.where(vals == index)[0]
                if len(ii) > 0:
                    this_embedding = torch.tensor([float(x) for x in line.decode('utf8').strip().split(' ')[1:]])
                    for i_index in ii:
                        embeddings.weight.data[i_index].copy_(this_embedding)
    pretrained_mask = torch.stack(pretrained_mask, dim=0)
    return embeddings, pretrained_mask

DATASET_FILES = {
    'pgr':['pgr/train.conllx', 'pgr/dev.conllx','pgr/test.conllx', 'pgr/train.mention.and.gold', 'pgr/dev.mention.and.gold'
           , 'pgr/test.mention.and.gold'],
    'cpr': ['cpr/train.conllx', 'cpr/dev.conllx', 'cpr/test.conllx', 'cpr/train.mention.and.gold', 'cpr/dev.mention.and.gold',
            'cpr/test.mention.and.gold'],
    'tacred': ['tacred/train.conllx', 'tacred/dev.conllx', 'tacred/test.conllx', 'tacred/train.mention.and.gold',
            'tacred/dev.mention.and.gold', 'tacred/test.mention.and.gold'],
    'semeval': ['semeval/train.fixed_ordering.conllx', 'semeval/dev.fixed_ordering.conllx', 'semeval/test.fixed_ordering.conllx',
                'semeval/train.fixed_ordering.mention.and.gold', 'semeval/dev.fixed_ordering.mention.and.gold',
                'semeval/test.fixed_ordering.mention.and.gold'],
    'semeval_order': ['semeval/train.conllx', 'semeval/dev.fixed_ordering.conllx',
                'semeval/test.fixed_ordering.conllx',
                'semeval/train.mention.and.gold', 'semeval/dev.fixed_ordering.mention.and.gold',
                'semeval/test.fixed_ordering.mention.and.gold']
}

def load_data(word_alphabets, char_alphabets, pos_alphabet, type_alphabet, ner_alphabet, custom_args):
    parser_training_fn, parser_dev_fn, parser_test_fn, training_mention_id_and_gold, dev_mention_id_and_gold,\
    test_mention_id_and_gold = DATASET_FILES[custom_args.dataset_name]

    word_alphabet, graph_word_alphabet = word_alphabets
    char_alphabet, graph_char_alphabet = char_alphabets

    data_train = conllx_data.read_data_to_tensor_dicts(parser_training_fn, word_alphabet, char_alphabet, pos_alphabet,
                                                 type_alphabet, ner_alphabet,
                                                 symbolic_root=True, device=custom_args.device)
    data_dev = conllx_data.read_data_to_tensor_dicts(parser_dev_fn, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, ner_alphabet,
                                               symbolic_root=True, device=custom_args.device)
    data_test = conllx_data.read_data_to_tensor_dicts(parser_test_fn, word_alphabet, char_alphabet, pos_alphabet,
                                                type_alphabet, ner_alphabet,
                                                symbolic_root=True, device=custom_args.device)
    graph_data_train = conllx_data.read_data_to_tensor_dicts(parser_training_fn, graph_word_alphabet, graph_char_alphabet, pos_alphabet,
                                                 type_alphabet, ner_alphabet,
                                                 symbolic_root=True, device=custom_args.device, normalize_digits=False)
    graph_data_dev = conllx_data.read_data_to_tensor_dicts(parser_dev_fn, graph_word_alphabet, graph_char_alphabet, pos_alphabet, type_alphabet, ner_alphabet,
                                               symbolic_root=True, device=custom_args.device, normalize_digits=False)
    graph_data_test = conllx_data.read_data_to_tensor_dicts(parser_test_fn, graph_word_alphabet, graph_char_alphabet, pos_alphabet,
                                                type_alphabet, ner_alphabet,
                                                symbolic_root=True, device=custom_args.device, normalize_digits=False)

    train_mentions, train_labels = read_mention_ids_and_gold_labels(training_mention_id_and_gold)
    dev_mentions, dev_labels = read_mention_ids_and_gold_labels(dev_mention_id_and_gold)
    test_mentions, test_labels = read_mention_ids_and_gold_labels(test_mention_id_and_gold)

    return (data_train, data_dev, data_test), \
           (graph_data_train, graph_data_dev, graph_data_test), \
           (train_mentions, dev_mentions, test_mentions), \
           (train_labels, dev_labels, test_labels)

import random

def unk_single_mentions(graph_words, mentions, p=0.2):
    for index, (b1, e1, b2, e2) in enumerate(mentions):
        if e1 - b1 == 1:
            if random.random() < p: graph_words[index, b1+1:e1+1] = 0
        if e2 - b2 == 1:
            if random.random() < p: graph_words[index, b2+1:e2+1] = 0
    return graph_words

def unk_single_words(graph_words, p=0.2):
    bernoulli_mask = torch.zeros_like(graph_words).bernoulli_(1-p)
    graph_words = graph_words * bernoulli_mask
    return graph_words