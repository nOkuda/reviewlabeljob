"""Plots post survey data"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np


def _parse_args():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(description='Plot post survey data')
    parser.add_argument(
        '-o',
        default='.',
        help='directory where output is to be placed; default is current ' +
             'working directory')
    parser.add_argument(
        'postsurvey',
        help='file path to post survey data')
    return parser.parse_args()


def _parse_postsurvey(filepath):
    """Parse post survey data"""
    result = []
    with open(filepath) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                if line.startswith('#'):
                    continue
                result.append([int(a) for a in line.split('\t')])
    return np.array(result)


def _plot_postsurvey_bar(filename, data):
    """Plot and save bar chart"""
    fig, axis = plt.subplots(1, 1)
    axis.bar(
        np.arange(5),
        data,
        width=0.8,
        alpha=0.9,
        align='center',
        tick_label=[
            'Strongly\ndisagree',
            'Disagree',
            'Neither\nagree nor disagree',
            'Agree\n',
            'Strongly\nagree'])
    axis.set_xlim([-0.5, 4.5])
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _plot_postsurvey(filepath, outdir):
    """Plot post survey data"""
    data = _parse_postsurvey(filepath)
    print(
        'Number of females, number of males:',
        np.sum(data[:, 0] == 0),
        ',',
        np.sum(data[:, 0] == 1))
    harder_counts = [np.sum(data[:, 1] == a) for a in range(5)]
    moretime_counts = [np.sum(data[:, 2] == a) for a in range(5)]
    _plot_postsurvey_bar(
        os.path.join(outdir, 'harder.pdf'),
        harder_counts)
    _plot_postsurvey_bar(
        os.path.join(outdir, 'moretime.pdf'),
        moretime_counts)


def _main():
    args = _parse_args()
    _plot_postsurvey(args.postsurvey, args.o)


if __name__ == '__main__':
    _main()
