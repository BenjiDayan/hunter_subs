from transformers import (
  BertTokenizerFast,
  AutoModel,
)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model_bert_ner = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ner')

from transformers import pipeline
nlp_bert_ner = pipeline('ner', model='ckiplab/bert-base-chinese-ner', tokenizer=tokenizer, device='cuda')


import pysubs2
import re
fps = 29.97002997002997
subs = pysubs2.load('../video_and_data/ep2_v2.sub', fps=fps)


# frame 3300 to 40000 is roughly where the subs are
subs_subset = [s for s in subs if s.start >= 1000 * 3300/fps and s.end <= 1000 * 40000/fps]

# just the text itself, i.e. ['很好', '你们三个就由本大爷我', '亲自送到距离审查会场\\n最近的港口', ...]
subs_texts = [s.text for s in subs_subset]


def process_text(t):
    t = t.replace('\\n', '\n')  # idk why but this is how the subs are
    t = re.sub(r'\s', '', t)  # all whitespace is deleted - fine for chinese
    return t
subs_texts = [process_text(t) for t in subs_texts]

# each element is a list of extracted entities, e.g. ners[7]:
# [{'entity': 'B-PERSON',
#   'score': 0.9999995,
#   'index': 1,
#   'word': '小',
#   'start': 0,
#   'end': 1},
#  {'entity': 'E-PERSON',
#   'score': 0.9999995,
#   'index': 2,
#   'word': '杰',
#   'start': 1,
#   'end': 2},
#  {'entity': 'B-PERSON',
#   'score': 0.9999995,
#   'index': 4,
#   'word': '酷',
#   'start': 3,
#   'end': 4}, ...]
ners = [nlp_bert_ner(x) for x in subs_texts]

def group_person_ners(ner_list):
    """Group together all the person entities in a [dict1, dict2, ...] list"""
    person_ners = [n for n in ner_list if 'PERSON' in n['entity']]
    # group together if they are adjacently indexed
    grouped = []
    if len(person_ners) > 0:
        person = [person_ners[0]]
        for ner_dict in person_ners[1:]:
            if ner_dict['start'] == person[-1]['end']:
                person.append(ner_dict)
            else:
                grouped.append(person)
                person = [ner_dict]
        grouped.append(person)

    # so now [[{... '小'}, {... '杰'}], [{... '酷'}], ...]
    return grouped



people_grouped_ners = [group_person_ners(ner_list) for ner_list in ners]

def get_people_from_ners(people_grouped_ners):
    """Get the people names from the grouped ners"""
    people = []
    for person_sub_list in people_grouped_ners:
        name = ''.join([p['word'] for p in person_sub_list])
        people.append(name)
    return people

people = [get_people_from_ners(p) for p in people_grouped_ners]
from functools import reduce
people = reduce(lambda x,y: x+y, people)  # flatten list of lists


from collections import Counter
# Counter({'酷拉皮卡': 10, '雷欧力': 9, '小杰': 6, '杰': 2, '确宾': 1, '歙妤': 1, '童新': 1})
people_counts = Counter(people)
