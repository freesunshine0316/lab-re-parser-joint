import json
import io
import zipfile
import spacy
from copy import deepcopy
import re

model = spacy.load('en_core_web_md')

entry = {'subj_end':0, 'subj_start':0, 'obj_end':0, 'obj_start':0, 'tokens':[], 'poses':[], 'ref':False, 'ner':[]}

SENT_PATTERN = re.compile('\d+\t"(.*<e1>(.+)</e1>.*<e2>(.+)</e2>.*)"')
BOTH_ORDERING = True

def get_sent_and_entities(line):
    match_obj = re.match(SENT_PATTERN, line)
    sent = match_obj.group(1)
    entity1 = match_obj.group(2)
    entity2 = match_obj.group(3)
    return sent, entity1, entity2

def clean_sent_and_get_indices(line, e1, e2):
    e1 = [x.text for x in model(e1)]
    e2 = [x.text for x in model(e2)]
    line = line.replace('<e1>', ' ')
    line = line.replace('</e1>', ' ')
    line = line.replace('<e2>', ' ')
    line = line.replace('</e2>', ' ')
    line = re.sub('  +', ' ', line).strip()
    words = model(line)
    sent_text = [x.text for x in words]
    e1_len = len(e1)
    e2_len = len(e2)
    # print(e2, e2_len, [x.text for x in words])
    for i in range(0, len(sent_text)-e1_len, 1):
        seg = sent_text[i:i+e1_len]
        if seg == e1:
            e1_start = i
            e1_end = i+e1_len
            break
    else:
        raise Exception
    for i in range(0, len(sent_text)-e2_len, 1):
        seg = sent_text[i:i+e2_len]
        if seg == e2:
            e2_start = i
            e2_end = i+e2_len
            break
    else:
        raise Exception
    return words, e1, e2, e1_start, e1_end, e2_start, e2_end

with zipfile.ZipFile('SemEval2010_task8_all_data.zip') as zipf:
    training_objs = []
    dev_objs = []
    test_objs = []
    with zipf.open('SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT', 'r') as train, \
            zipf.open('SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', 'r') as test:
        for index, file in enumerate([train, test]):
            if index == 0:
                cur_list = training_objs
            else:
                cur_list = test_objs
            file = io.TextIOWrapper(file, encoding='utf8')
            alllines = file.readlines()
            num_instances = 0
            for first_line_index in range(0, len(alllines), 4):
                num_instances += 1
                if num_instances > 7000 and cur_list is training_objs:
                    cur_list = dev_objs

                lines = alllines[first_line_index:first_line_index+4]
                sent, e1, e2 = get_sent_and_entities(lines[0])
                processed_tokens, e1, e2, e1_start, e1_end, e2_start, e2_end = clean_sent_and_get_indices(sent, e1, e2)

                tokens = [x.text for x in processed_tokens]
                poses = [x.tag_ for x in processed_tokens]
                ners = [x.ent_type_ if x.ent_type_ else 'O' for x in processed_tokens]

                item = deepcopy(entry)
                item['tokens'] = tokens
                item['poses'] = poses
                item['ner'] = ners

                label = lines[1].strip()
                if BOTH_ORDERING:
                    if 'e1' in label:
                        label_name = label.split('(')[0]
                        if "e1,e2" in label:
                            subj = e1
                            obj = e2
                            item['subj_start'] = e1_start
                            item['subj_end'] = e1_end
                            item['obj_start'] = e2_start
                            item['obj_end'] = e2_end
                        elif "e2,e1" in label:
                            subj = e2
                            obj = e1
                            item['subj_start'] = e2_start
                            item['subj_end'] = e2_end
                            item['obj_start'] = e1_start
                            item['obj_end'] = e1_end
                    else:
                        label_name = label
                        item['subj_start'] = e1_start
                        item['subj_end'] = e1_end
                        item['obj_start'] = e2_start
                        item['obj_end'] = e2_end
                    item['ref'] = label_name
                    cur_list.append(item)
                    if cur_list is training_objs:
                        item = deepcopy(entry)
                        item['tokens'] = tokens
                        item['poses'] = poses
                        item['ner'] = ners

                        label = lines[1].strip()
                        if 'e1' in label:
                            label_name = label.split('(')[0]+'-1'
                            if "e2,e1" in label:
                                subj = e1
                                obj = e2
                                item['subj_start'] = e1_start
                                item['subj_end'] = e1_end
                                item['obj_start'] = e2_start
                                item['obj_end'] = e2_end
                            elif "e1,e2" in label:
                                subj = e2
                                obj = e1
                                item['subj_start'] = e2_start
                                item['subj_end'] = e2_end
                                item['obj_start'] = e1_start
                                item['obj_end'] = e1_end
                        else:
                            label_name = label
                            item['subj_start'] = e2_start
                            item['subj_end'] = e2_end
                            item['obj_start'] = e1_start
                            item['obj_end'] = e1_end

                        item['ref'] = label_name
                        cur_list.append(item)
                else:
                    if 'e1' in label:
                        label_name = label.split('(')[0]
                        subj = e1
                        obj = e2
                        item['subj_start'] = e1_start
                        item['subj_end'] = e1_end
                        item['obj_start'] = e2_start
                        item['obj_end'] = e2_end

                        if "e2,e1" in label:
                            label_name += "-1"
                    else:
                        label_name = label
                        item['subj_start'] = e1_start
                        item['subj_end'] = e1_end
                        item['obj_start'] = e2_start
                        item['obj_end'] = e2_end
                    item['ref'] = label_name
                    cur_list.append(item)
    if BOTH_ORDERING:
        json.dump(training_objs, open('train.json','w',encoding='utf8'))
        json.dump(dev_objs, open('dev.json', 'w', encoding='utf8'))
        json.dump(test_objs, open('test.json', 'w', encoding='utf8'))
    else:
        json.dump(training_objs, open('train.fixed_ordering.json','w',encoding='utf8'))
        json.dump(dev_objs, open('dev.fixed_ordering.json', 'w', encoding='utf8'))
        json.dump(test_objs, open('test.fixed_ordering.json', 'w', encoding='utf8'))