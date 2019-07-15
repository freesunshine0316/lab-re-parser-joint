import json

STR_FORMAT = '{}\t{}\t{}\t_\t{}\t_\t0\troot\t_\t_'

def load_data(training_fn, test_fn):
    with open(training_fn) as trfh, open(test_fn) as tsfh:
        training_json = json.load(trfh)
        test_json = json.load(tsfh)
    return training_json, test_json

training_fn = 'pgr/train.json'
test_fn = 'pgr/test.json'

training_outfn = 'pgr/train.conllx'
dev_outfn = 'pgr/dev.conllx'
test_outfn = 'pgr/test.conllx'

training, test = load_data(training_fn, test_fn)

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
        pos = '-LCB-'
    elif word == '}':
        word = '-RCB-'
        pos = '-RCB-'
    else: word = word
    if pos == 'HYPH': pos = ','
    elif pos == 'ADD' or pos == 'XX' or pos == 'NFP': pos = 'FW'
    return STR_FORMAT.format(index, word, word.lower(), pos)

def print_out_conllx(dataset, outfn, train=False, dev_fn=None):
    header = '# text = '
    ofh = open(outfn, 'w', encoding='utf8')
    if train:
        dev_size = int(0.15 * len(dataset))
        devfh = open(dev_fn, 'w', encoding='utf8')
    for index, instance in enumerate(dataset):
        if train and index < dev_size:
            handle = devfh
        else:
            handle = ofh
        sent = instance['tokens']
        poses = instance['poses']
        # print(header + ' '.join(sent), file=hanlde)
        for index, (word, pos) in enumerate(zip(sent, poses)):
            string = assemble_conllx_entry(index, word, pos)
            print(string, file=handle)
        print('', file=handle)


print_out_conllx(training, training_outfn, train=True, dev_fn=dev_outfn)
print_out_conllx(test, test_outfn)