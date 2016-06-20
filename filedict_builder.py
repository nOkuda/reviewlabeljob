import pickle
import re

FILENAME = '/aml/data/amazon/amazon.txt'
LABELS = '/aml/data/amazon/amazon.response'
PROFANITY = '/aml/home/okuda/data/profanity.txt'

def clean_text(data, blacklist):
    """Labels the document with its untokenized text, but with bad words filtered"""
    for word in blacklist:
        replace = '*' * len(word)
        regex = re.compile(r'\b({})\b'.format(word), re.M | re.I)
        data = regex.sub(replace, data)
    return data

if __name__ == '__main__':
    filedict = {}
    # Here we populate filedict with docnumber as the key and document as the value
    with open(FILENAME, 'r') as f:
        with open(PROFANITY, 'r') as blackf:
            blacklist = blackf.readlines()
        for line in f:
            filedict[line.split('\t')[0]] = {}
            filedict[line.split('\t')[0]]['text'] = clean_text(
                line.split('\t')[1], blacklist)
    with open(LABELS, 'r') as ifh:
        for line in ifh:
            (doc_id, label) = line.strip().split('\t')
            filedict[doc_id]['label'] = float(label)
    with open('filedict.pickle', 'wb') as ofh:
        pickle.dump(filedict, ofh)

