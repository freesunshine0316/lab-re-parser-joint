
import os, sys, codecs, json

data = []
data += json.load(codecs.open('dev.json', 'rU', 'utf-8'))
data += json.load(codecs.open('train.json', 'rU', 'utf-8'))
json.dump(data, codecs.open('dev_train.json', 'w', 'utf-8'))

