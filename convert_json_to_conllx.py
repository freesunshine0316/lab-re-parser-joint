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

#
# training_fn = 'cpr/train.json'
# dev_fn = 'cpr/dev.json'
# test_fn = 'cpr/test.json'
#
# training_outfn = 'cpr/train.conllx'
# dev_outfn = 'cpr/dev.conllx'
# test_outfn = 'cpr/test.conllx'
#
# training_mention_id_and_gold = 'cpr/train.mention.and.gold'
# dev_mention_id_and_gold = 'cpr/dev.mention.and.gold'
# test_mention_id_and_gold = 'cpr/test.mention.and.gold'

training_fn = 'tacred/train.json'
dev_fn = 'tacred/dev.json'
test_fn = 'tacred/test.json'

training_outfn = 'tacred/train.conllx'
dev_outfn = 'tacred/dev.conllx'
test_outfn = 'tacred/test.conllx'

training_mention_id_and_gold = 'tacred/train.mention.and.gold'
dev_mention_id_and_gold = 'tacred/dev.mention.and.gold'
test_mention_id_and_gold = 'tacred/test.mention.and.gold'

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

def tacred_preprocess(instance):
    subj_type = instance['subj_type']+'-SUBJ'
    obj_type = instance['obj_type']+'-OBJ'
    subj_len = instance['subj_end'] - instance['subj_start'] + 1
    obj_len = instance['obj_end'] - instance['obj_start'] + 1
    subj_start, subj_end, obj_start, obj_end = instance['subj_start'], instance['subj_end'], instance['obj_start'], instance['obj_end']
    if subj_start < obj_start:
        tokens = instance['token']
        poses = instance['stanford_pos']
        del tokens[subj_start:subj_end+1]
        del poses[subj_start:subj_end+1]
        tokens.insert(subj_start, subj_type)
        poses.insert(subj_start, 'NNP')
        forward_moving_gap = subj_len - 1
        del tokens[obj_start-forward_moving_gap:obj_end+1-forward_moving_gap]
        del poses[obj_start-forward_moving_gap:obj_end+1-forward_moving_gap]
        tokens.insert(obj_start-forward_moving_gap, obj_type)
        poses.insert(obj_start-forward_moving_gap, 'NNP')
        instance['subj_end'] = subj_start + 1
        instance['obj_start'] = obj_start-forward_moving_gap
        instance['obj_end'] = instance['obj_start'] + 1
    else:
        tokens = instance['token']
        poses = instance['stanford_pos']
        del tokens[obj_start:obj_end+1]
        del poses[obj_start:obj_end+1]
        tokens.insert(obj_start, obj_type)
        poses.insert(obj_start, 'NNP')
        forward_moving_gap = obj_len - 1
        del tokens[subj_start-forward_moving_gap:subj_end+1-forward_moving_gap]
        del poses[subj_start-forward_moving_gap:subj_end+1-forward_moving_gap]
        tokens.insert(subj_start-forward_moving_gap, subj_type)
        poses.insert(subj_start-forward_moving_gap, 'NNP')
        instance['obj_end'] = obj_start + 1
        instance['subj_start'] = subj_start-forward_moving_gap
        instance['subj_end'] = instance['subj_start'] + 1

TACRED = True

def get_conllx_string(instance, conllx_handle, mention_handle):

    if TACRED:
        tacred_preprocess(instance)

    if 'tokens' in instance:
        sent = instance['tokens']
    elif 'toks' in instance:
        sent = instance['toks']
    elif 'token' in instance:
        sent = instance['token']
    else:
        raise Exception

    if 'poses' in instance:
        poses = instance['poses']
    elif 'stanford_pos' in instance:
        poses = instance['stanford_pos']

    # print(header + ' '.join(sent), file=hanlde)
    for index, (word, pos) in enumerate(zip(sent, poses)):
        string = assemble_conllx_entry(index, word, pos)
        print(string, file=conllx_handle)
    print('', file=conllx_handle)
    # pgr must have +1, cpr and tacred must not have +1
    if 'ref' in instance:
        mention_str = [str(x) for x in [instance['subj_start'], instance['subj_end'], instance['obj_start'], instance['obj_end'],
                                    instance['ref']]]
    elif 'relation' in instance:
        mention_str = [str(x) for x in
                       [instance['subj_start'], instance['subj_end'], instance['obj_start'], instance['obj_end'],
                        instance['relation']]]
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