import json

STR_FORMAT = '{}\t{}\t{}\t_\t{}\t_\t0\troot\t_\t_'

def load_data(fns):
    jsons = []
    for fn in fns:
        with open(fn) as trfh:
            jsons.append(json.load(trfh))
    return jsons
#
# training_fn = 'pgr/train.json'
# # dev_fn = 'pgr/dev.json'
# test_fn = 'pgr/test.json'
#
# training_outfn = 'pgr/train.conllx'
# dev_outfn = 'pgr/dev.conllx'
# test_outfn = 'pgr/test.conllx'
#
# training_mention_id_and_gold = 'pgr/train.mention.and.gold'
# dev_mention_id_and_gold = 'pgr/dev.mention.and.gold'
# test_mention_id_and_gold = 'pgr/test.mention.and.gold'
# # data_list = load_data([training_fn, test_fn])
training_fn = 'cpr/train.json'
dev_fn = 'cpr/dev.json'
test_fn = 'cpr/test.json'

training_outfn = 'cpr/train.conllx'
dev_outfn = 'cpr/dev.conllx'
test_outfn = 'cpr/test.conllx'

training_mention_id_and_gold = 'cpr/train.mention.and.gold'
dev_mention_id_and_gold = 'cpr/dev.mention.and.gold'
test_mention_id_and_gold = 'cpr/test.mention.and.gold'

data_list = load_data([training_fn, dev_fn, test_fn])

def assemble_conllx_entry(index, word, pos):
    #process the words and pos tags, converting them to ptb standards
    if word == '(' or word == '<':
        word = '-LRB-'
        pos = '-LRB-'
    elif word == ')' or word == '>':
        word = '-RRB-'
        pos = '-RRB-'
    elif word == '{':
        word = '-LCB-'
        pos = '-LRB-'
    elif word == '}':
        word = '-RCB-'
        pos = '-RRB-'
    else: word = word
    if pos == 'HYPH': pos = ','
    elif pos == 'ADD' or pos == 'XX' or pos == 'NFP': pos = 'FW'
    elif pos == 'AFX': pos = 'JJ'
    return STR_FORMAT.format(index, word, word.lower(), pos)

def print_out_conllx(dataset, outfn, mentionfn, train=False, dev_fn=None, dev_mention_fn=None):
    header = '# text = '
    ofh = open(outfn, 'w', encoding='utf8')
    menfh = open(mentionfn, 'w', encoding='utf8')
    if train:
        dev_size = int(0.15 * len(dataset))
        devfh = open(dev_fn, 'w', encoding='utf8')
        dev_mention_fh = open(dev_mention_fn, 'w', encoding='utf8')
    for index, instance in enumerate(dataset):
        if train and index < dev_size:
            handle = devfh
            mention_handle = dev_mention_fh
        else:
            handle = ofh
            mention_handle = menfh
        if 'file_data' in instance:
            for subinstance in instance['file_data']:
                get_conllx_string(subinstance, handle, mention_handle)
        else:
            get_conllx_string(instance, handle, mention_handle)

def get_conllx_string(instance, conllx_handle, mention_handle):
    if 'tokens' in instance:
        sent = instance['tokens']
    elif 'toks' in instance:
        sent = instance['toks']
    else:
        raise Exception
    poses = instance['poses']
    # print(header + ' '.join(sent), file=hanlde)
    for index, (word, pos) in enumerate(zip(sent, poses)):
        string = assemble_conllx_entry(index, word, pos)
        print(string, file=conllx_handle)
    print('', file=conllx_handle)
    # pgr must have +1, cpr must not have +1
    mention_str = [str(x) for x in [instance['subj_start'], instance['subj_end'], instance['obj_start'], instance['obj_end'],
                                    instance['ref']]]
    print(' '.join(mention_str), file=mention_handle)

for data, data_fn, mention_fn in zip(data_list, [training_outfn, dev_outfn, test_outfn], [training_mention_id_and_gold, dev_mention_id_and_gold,
                                                                              test_mention_id_and_gold]):
    print_out_conllx(data, data_fn, mention_fn)

# for data, data_fn, mention_fn in zip(data_list, [[training_outfn, dev_outfn], test_outfn], [[training_mention_id_and_gold, dev_mention_id_and_gold],
#                                                                               test_mention_id_and_gold]):
#     if len(data_fn) == 2:
#         print_out_conllx(data, data_fn[0], mention_fn[0], train=True, dev_fn=data_fn[1], dev_mention_fn=mention_fn[1])
#     else:
#         print_out_conllx(data, data_fn, mention_fn)