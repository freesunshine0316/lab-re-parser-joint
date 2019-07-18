
import os, sys, json, codecs, random

def read_ref(inpath):
    all_refs = {}
    all_ids = []
    with codecs.open(inpath, 'rU', 'utf-8') as f:
        for g, inst in enumerate(json.load(f)):
            id = inst['id']
            ref = (inst['ref'].upper() == "TRUE")
            all_refs[id] = ref
            all_ids.append(id)
    return all_refs, all_ids

def cal_f1(out_num, ref_num, both_num):
    precision = both_num / out_num
    recall = both_num / ref_num
    fscore = 2*precision*recall/(precision+recall)
    return fscore

all_refs, all_ids = read_ref('data/test.json')
all_a1 = json.load(codecs.open('logs/res_test_%s.json'%sys.argv[1], 'rU', 'utf-8'))
all_a2 = json.load(codecs.open('logs/res_test_%s.json'%sys.argv[2], 'rU', 'utf-8'))

p_value = 0.0

for k in range(1000):
    both1_num, both2_num, a1_num, a2_num, ref_num = 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(len(all_ids)):
        n = random.randint(0, len(all_ids)-1)
        a1 = all_a1[all_ids[n]]
        a2 = all_a2[all_ids[n]]
        ref = all_refs[all_ids[n]]
        if ref:
            ref_num += 1.0
        if a1:
            a1_num += 1.0
            if ref:
                both1_num += 1.0
        if a2:
            a2_num += 1.0
            if ref:
                both2_num += 1.0
    if cal_f1(a1_num, ref_num, both1_num) < cal_f1(a2_num, ref_num, both2_num):
        p_value += 1.0
print p_value/1000.0
