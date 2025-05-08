from dataset.POS.process import POS_MAP, read_common_pos
import os

inverse_POS = {v: k for k, v in POS_MAP.items()}

mapper = read_common_pos(os.environ['COMMON_POS_FILENAME'])

marks = ['.', ',', ':', ';', '?', '!', "'", '-', '"', '{', '}', '(', ')', '[', ']' '|', '&', '*', '/', '~']

sentence = "The man, who has not been identified by police, was taken to a hospital for treatment."

for word in sentence.split():
    for m in marks:
        word = word.replace(m, '')

    if word in mapper:
        tags = [inverse_POS[x] for x in mapper[word]]
    else:
        tags = ['None']
    print(word, tags)