
import os, sys, codecs, json

def gen_ref(inpath, refmap):
    with codecs.open(inpath, 'rU', 'utf-8') as f:
        for inst in json.load(f):
            ref = inst['relation']
            if ref not in refmap:
                refmap[ref] = len(refmap)

refmap = {}
gen_ref('dev.json', refmap)
gen_ref('test.json', refmap)
gen_ref('train.json', refmap)
print len(refmap)
json.dump(refmap, codecs.open('refmap.json', 'w', 'utf-8'))

