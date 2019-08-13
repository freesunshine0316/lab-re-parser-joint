__author__ = 'max'

import os.path
import random
import numpy as np
from .alphabet import Alphabet
from .logger import get_logger
from . import utils
import torch

# Special vocabulary symbols - we always put them at the start.
PAD = "_PAD"
PAD_POS = "_PAD_POS"
PAD_TYPE = "_<PAD>"
PAD_CHAR = "_PAD_CHAR"
PAD_NER = "_"
ROOT = "_ROOT"
ROOT_POS = "_ROOT_POS"
ROOT_TYPE = "_<ROOT>"
ROOT_CHAR = "_ROOT_CHAR"
ROOT_NER = "ROOT_NER"
END = "_END"
END_POS = "_END_POS"
END_TYPE = "_<END>"
END_CHAR = "_END_CHAR"

_START_VOCAB = [PAD, ROOT, END]

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 140, 300]

from .reader import CoNLLXReader


def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=50000, embedd_dict=None,
                     min_occurence=1, normalize_digits=True, suffix=''):
    def expand_vocab():
        vocab_set = set(vocab_list)
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r', encoding='utf8') as file:
                for line in file:
                    # line = line.decode('utf-8')
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = utils.DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    pos = tokens[4]
                    type = tokens[7]

                    pos_alphabet.add(pos)
                    type_alphabet.add(type)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet(suffix+'word', defualt_value=True, singleton=True)
    char_alphabet = Alphabet(suffix+'character', defualt_value=True)
    pos_alphabet = Alphabet(suffix+'pos')
    type_alphabet = Alphabet(suffix+'type')
    ner_alphabet = Alphabet(suffix+'ner')
    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)
        ner_alphabet.add(PAD_NER)

        char_alphabet.add(ROOT_CHAR)
        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)
        ner_alphabet.add(ROOT_NER)

        char_alphabet.add(END_CHAR)
        pos_alphabet.add(END_POS)
        type_alphabet.add(END_TYPE)

        vocab = dict()
        with open(train_path, 'r', encoding='utf8') as file:
            for line in file:
                # line = line.decode('utf-8')
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split('\t')
                for char in tokens[1]:
                    char_alphabet.add(char)

                word = utils.DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                pos = tokens[4]
                type = tokens[7]
                ner = tokens[-1]

                pos_alphabet.add(pos)
                type_alphabet.add(type)
                ner_alphabet.add(ner)

                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)
        ner_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)
        ner_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()
    ner_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())
    logger.info("NER Alphabet Size: {}".format(ner_alphabet.size()))
    return word_alphabet, char_alphabet, pos_alphabet, type_alphabet, ner_alphabet


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, ner_alphabet, max_size=None,
              normalize_digits=True, symbolic_root=False, symbolic_end=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, ner_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    while inst is not None and (not max_size or counter < max_size):
        inst.index = counter
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.ner_ids, inst.index])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length

def read_unbucketed_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, ner_alphabet, max_size=None,
              normalize_digits=True, symbolic_root=False, symbolic_end=False):
    data = {}
    max_char_length = {}
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, ner_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    while inst is not None and (not max_size or counter < max_size):
        inst.index = counter
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        if inst_size not in data:
            data[inst_size] = []
            max_char_length[inst_size] = 0

        data[inst_size].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, inst.ner_ids, inst.index])
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length[inst_size] < max_len:
            max_char_length[inst_size] = max_len

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length


def get_batch(data, batch_size, word_alphabet=None, unk_replace=0.):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    bucket_length = _buckets[bucket_id]
    char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([batch_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    hid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    tid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)
    single = np.zeros([batch_size, bucket_length], dtype=np.int64)

    for b in range(batch_size):
        wids, cid_seqs, pids, hids, tids = random.choice(data[bucket_id])

        inst_size = len(wids)
        # word ids
        wid_inputs[b, :inst_size] = wids
        wid_inputs[b, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[b, c, :len(cids)] = cids
            cid_inputs[b, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[b, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[b, :inst_size] = pids
        pid_inputs[b, inst_size:] = PAD_ID_TAG
        # type ids
        tid_inputs[b, :inst_size] = tids
        tid_inputs[b, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[b, :inst_size] = hids
        hid_inputs[b, inst_size:] = PAD_ID_TAG
        # masks
        masks[b, :inst_size] = 1.0

        if unk_replace:
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[b, j] = 1

    if unk_replace:
        noise = np.random.binomial(1, unk_replace, size=[batch_size, bucket_length])
        wid_inputs = wid_inputs * (1 - noise * single)

    return wid_inputs, cid_inputs, pid_inputs, hid_inputs, tid_inputs, masks


def iterate_batch(data, batch_size, word_alphabet=None, unk_replace=0., shuffle=False):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids = inst
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            if unk_replace:
                for j, wid in enumerate(wids):
                    if word_alphabet.is_singleton(wid):
                        single[i, j] = 1

        if unk_replace:
            noise = np.random.binomial(1, unk_replace, size=[bucket_size, bucket_length])
            wid_inputs = wid_inputs * (1 - noise * single)

        indices = None
        if shuffle:
            indices = np.arange(bucket_size)
            np.random.shuffle(indices)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield wid_inputs[excerpt], cid_inputs[excerpt], pid_inputs[excerpt], hid_inputs[excerpt], \
                  tid_inputs[excerpt], masks[excerpt]


def read_data_to_tensor(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None,
                        normalize_digits=True, symbolic_root=False, symbolic_end=False, device=torch.device('cpu')):
    data, max_char_length = read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                      max_size=max_size, normalize_digits=normalize_digits,
                                      symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_tensor = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensor.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        indices = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids, index = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            indices[i] = index
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        pos = torch.from_numpy(pid_inputs).to(device)
        heads = torch.from_numpy(hid_inputs).to(device)
        types = torch.from_numpy(tid_inputs).to(device)
        masks = torch.from_numpy(masks).to(device)
        single = torch.from_numpy(single).to(device)
        lengths = torch.from_numpy(lengths).to(device)
        indices = torch.from_numpy(indices).to(device)

        data_tensor.append((words, chars, pos, heads, types, masks, single, lengths, indices))

    return data_tensor, bucket_sizes


def read_data_to_tensor_dicts(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, ner_alphabet, max_size=None,
                        normalize_digits=True, symbolic_root=False, symbolic_end=False, device=torch.device('cpu')):
    data, max_char_length = read_unbucketed_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, ner_alphabet,
                                      max_size=max_size, normalize_digits=normalize_digits,
                                      symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    bucket_sizes = {x:len(data[x]) for x in data}
    data_tensor = {}

    for bucket_id in data:
        bucket_size = bucket_sizes[bucket_id]

        bucket_length = bucket_id
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        nerid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        indices = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids, nerids, index = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            indices[i] = index
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # ner ids
            nerid_inputs[i, :inst_size] = nerids
            nerid_inputs[i, inst_size:] = PAD_ID_TAG
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :inst_size] = tids
            tid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        ners = torch.from_numpy(nerid_inputs).to(device)
        pos = torch.from_numpy(pid_inputs).to(device)
        heads = torch.from_numpy(hid_inputs).to(device)
        types = torch.from_numpy(tid_inputs).to(device)
        masks = torch.from_numpy(masks).to(device)
        single = torch.from_numpy(single).to(device)
        lengths = torch.from_numpy(lengths).to(device)
        indices = torch.from_numpy(indices).to(device)

        data_tensor[bucket_id] = (words, chars, pos, ners, heads, types, masks, single, lengths, indices)

    return data_tensor, bucket_sizes

def get_batch_tensor(data, batch_size, unk_replace=0.):
    data_tensor, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    words, chars, pos, heads, types, masks, single, lengths = data_tensor[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    index = index.to(words.device)

    words = words[index]
    if unk_replace:
        ones = single.new_ones(batch_size, bucket_length)
        noise = masks.new_empty(batch_size, bucket_length).bernoulli_(unk_replace).long()
        words = words * (ones - single[index] * noise)

    return words, chars[index], pos[index], heads[index], types[index], masks[index], lengths[index]


def iterate_batch_tensor(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue

        words, chars, pos, heads, types, masks, single, lengths, indices = data_tensor[bucket_id]
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = masks.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        # indices = None
        # if shuffle:
        #     indices = torch.randperm(bucket_size).long()
        #     indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            # if shuffle:
            #     excerpt = indices[start_idx:start_idx + batch_size]
            # else:
            excerpt = slice(start_idx, start_idx + batch_size)
            yield words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], \
                  masks[excerpt], lengths[excerpt], indices[excerpt]


def doubly_iterate_batch_tensors(batch, graph_batch, batch_size=1, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = batch
    graph_data_tensor, graph_bucket_sizes = graph_batch

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))
        for bucket_id in bucket_indices:
            bucket_size = bucket_sizes[bucket_id]
            if bucket_size == 0:
                continue
            words, chars, pos, heads, types, masks, single, lengths, indices = data_tensor[bucket_id]
            graph_words, graph_chars, graph_pos, graph_heads, graph_types, graph_masks, graph_single, graph_lengths\
            , graph_indices = graph_data_tensor[bucket_id]
            data_indices = torch.randperm(bucket_size).long()
            for start_idx in range(0, bucket_size, batch_size):
                excerpt = slice(data_indices[start_idx].item(), data_indices[start_idx].item()+batch_size)
                yield (words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt],
                      masks[excerpt], lengths[excerpt], indices[excerpt]), (graph_words[excerpt], graph_chars[excerpt], graph_pos[excerpt],
                        graph_heads[excerpt], graph_types[excerpt], graph_masks[excerpt],
                                                         graph_lengths[excerpt], graph_indices[excerpt])

    else:
        for bucket_id in bucket_indices:
            bucket_size = bucket_sizes[bucket_id]
            bucket_length = _buckets[bucket_id]
            if bucket_size == 0:
                continue

            words, chars, pos, heads, types, masks, single, lengths, indices = data_tensor[bucket_id]
            graph_words, graph_chars, graph_pos, graph_heads, graph_types, graph_masks, graph_single, graph_lengths\
            , graph_indices = graph_data_tensor[bucket_id]
            if unk_replace:
                ones = single.new_ones(bucket_size, bucket_length)
                noise = masks.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
                words = words * (ones - single * noise)

            # indices = None
            # if shuffle:
            #     indices = torch.randperm(bucket_size).long()
            #     indices = indices.to(words.device)
            for start_idx in range(0, bucket_size, batch_size):

                excerpt = slice(start_idx, start_idx + batch_size)
                yield (words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt],
                      masks[excerpt], lengths[excerpt], indices[excerpt]), (graph_words[excerpt], graph_chars[excerpt], graph_pos[excerpt],
                        graph_heads[excerpt], graph_types[excerpt], graph_masks[excerpt],
                                                         graph_lengths[excerpt], graph_indices[excerpt])

def doubly_iterate_batch_tensors_dicts(parser_data, graph_data, batch_size=1, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = parser_data
    graph_data_tensor, graph_bucket_sizes = graph_data

    bucket_indices = list(data_tensor.keys())
    if shuffle:
        np.random.shuffle(bucket_indices)
        for bucket_id in bucket_indices:
            bucket_size = bucket_sizes[bucket_id]
            if bucket_size == 0:
                continue
            words, chars, pos, _, heads, types, masks, single, lengths, indices = data_tensor[bucket_id]
            graph_words, graph_chars, graph_pos, graph_ners, graph_heads, graph_types, graph_masks, graph_single, graph_lengths\
            , graph_indices = graph_data_tensor[bucket_id]
            data_indices = torch.randperm(bucket_size).long()
            for start_idx in range(0, bucket_size, batch_size):
                excerpt = data_indices[start_idx:start_idx+batch_size]
                yield (words[excerpt], chars[excerpt], pos[excerpt], None, heads[excerpt], types[excerpt],
                      masks[excerpt], lengths[excerpt], indices[excerpt]), (graph_words[excerpt], graph_chars[excerpt], graph_pos[excerpt], graph_ners[excerpt],
                        graph_heads[excerpt], graph_types[excerpt], graph_masks[excerpt],
                                                         graph_lengths[excerpt], graph_indices[excerpt])

    else:
        for bucket_id in bucket_indices:
            bucket_size = bucket_sizes[bucket_id]
            if bucket_size == 0:
                continue

            words, chars, pos, _, heads, types, masks, single, lengths, indices = data_tensor[bucket_id]
            graph_words, graph_chars, graph_pos, graph_ners, graph_heads, graph_types, graph_masks, graph_single, graph_lengths\
            , graph_indices = graph_data_tensor[bucket_id]

            for start_idx in range(0, bucket_size, batch_size):

                excerpt = slice(start_idx, start_idx + batch_size)
                yield (words[excerpt], chars[excerpt], pos[excerpt], None, heads[excerpt], types[excerpt],
                      masks[excerpt], lengths[excerpt], indices[excerpt]), (graph_words[excerpt], graph_chars[excerpt], graph_pos[excerpt],
                                graph_ners[excerpt], graph_heads[excerpt], graph_types[excerpt], graph_masks[excerpt], graph_lengths[excerpt], graph_indices[excerpt])
