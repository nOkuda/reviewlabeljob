"""Sort documents by top topic"""
import os
import pickle


OUTPUTDIR = 'sorted_output'


def _run():
    """Dump out sorted documents"""
    os.makedirs(OUTPUTDIR, exist_ok=True)
    with open('toptopic.pickle', 'rb') as ifh:
        toptopic = pickle.load(ifh)
    with open('filedict.pickle', 'rb') as ifh:
        filedict = pickle.load(ifh)
    for i, topic in enumerate(toptopic):
        with open(os.path.join(OUTPUTDIR, str(i)+'.txt'), 'w') as ofh:
            for perc, title in topic:
                ofh.write(str(perc)+'\t'+title+'\t'+filedict[title]['text'])
                ofh.write('\n')


if __name__ == '__main__':
    _run()
