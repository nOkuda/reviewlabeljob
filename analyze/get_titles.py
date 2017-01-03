"""Get titles from corpus pickle"""
import argparse
import pickle


def _parse_args():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        description='Extract titles from pickled corpus')
    parser.add_argument(
        'corpus',
        help='path to pickled corpus')
    return parser.parse_args()


def _run():
    """Save out titles from corpus pickle as `titles.pickle`"""
    args = _parse_args()
    with open(args.corpus, 'rb') as ifh:
        corpus = pickle.load(ifh)
    with open('titles.pickle', 'wb') as ofh:
        pickle.dump(corpus.titles, ofh)


if __name__ == '__main__':
    _run()
