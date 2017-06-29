"""For analyzing user study data"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import scipy.stats


def _parse_args():
    """Parses commandline arguments"""
    parser = argparse.ArgumentParser(description='Analyze user study data')
    parser.add_argument(
        'userdata',
        help='directory where data is stored; assuming the data files end ' +
             'in ".data"')
    parser.add_argument(
        'filedict',
        help='file path to pickle containing corpus information')
    return parser.parse_args()


def _parse_data(dirpath):
    """Parses participant data, ignoring first five labeled document per
    treatment

    For each data file, there is a matrix.
        * the 0th column is the start time (in milliseconds)
        * the 1st column is the end time (in milliseconds)
        * the 2nd column is the topic number
        * the 3rd column is the document id
        * the 4th column is the user's label
    All values in the matrix are integers.
    """
    result = {}
    for filename in os.listdir(dirpath):
        filename_name, filename_ext = os.path.splitext(filename)
        if filename_ext == '.data':
            record = {}
            rawdata = []
            with open(os.path.join(dirpath, filename)) as ifh:
                for line in ifh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        parsed = line.split(', ')
                        record['group'] = parsed[0]
                        record['orderedfirst'] = parsed[1] == 'Ordered First'
                    else:
                        rawdata.append([int(a) for a in line.split()])
            # each user should have labeled 90 documents, 45 per treatment,
            # where the first 5 of each treatment is ignored
            if record['orderedfirst']:
                record['ordered'] = np.array(rawdata[5:45])
                record['random'] = np.array(rawdata[50:90])
            else:
                record['random'] = np.array(rawdata[4:45])
                record['ordered'] = np.array(rawdata[50:90])
            result[filename_name] = record
    return result


def _get_accuracies_helper(data, filedict, treatment):
    """Computes accuracy for given treatment"""
    correct = 0
    total = 0
    for i, title in enumerate(data[treatment][:, 3]):
        if data[treatment][i, 4] == filedict[str(title)]['label']:
            correct += 1
        total += 1
    return correct / total


def _get_accuracies(userdata, filedict):
    """Computes accuracies of random and ordered treatments"""
    randoms = []
    ordereds = []
    for _, data in userdata.items():
        randoms.append(_get_accuracies_helper(data, filedict, 'random'))
        ordereds.append(_get_accuracies_helper(data, filedict, 'ordered'))
    return randoms, ordereds


def _get_times(data):
    """Gets labeling time in seconds"""
    return (data[:, 1] - data[:, 0]) / 1000


def _get_sums(userdata):
    """Gets sum of labeling times per treatment

    Returns two arrays, where index represents participant
    """
    randoms = []
    ordereds = []
    for _, data in userdata.items():
        randoms.append(_get_times(data['random']).sum())
        ordereds.append(_get_times(data['ordered']).sum())
    return np.array(randoms), np.array(ordereds)


def _get_time_values(userdata):
    """Sorts labeling times into treatments"""
    randoms = []
    ordereds = []
    for _, data in userdata.items():
        randoms.extend(_get_times(data['random']))
        ordereds.extend(_get_times(data['ordered']))
    return np.array(randoms), np.array(ordereds)


def _is_normal(diffs):
    """Tests for normality"""
    tstat, pval = scipy.stats.shapiro(diffs)
    print('Shapiro-Wilk Test:', tstat, pval)
    if pval < 0.05:
        return False
    return True


def _is_greater(diffs):
    """Checks if differences support hypothesis that minuend treatment has a
    significantly greater distribution via paired t test"""
    tstat, pval = scipy.stats.ttest_1samp(diffs, 0, nan_policy='raise')
    print('Paired t Test:', tstat, pval)
    if pval / 2 < 0.05 and tstat > 0:
        return True
    return False


def _try_t_test(randoms, ordereds):
    """Performs paired t test, with randoms as minuend treatment"""
    randoms_normal = _is_normal(randoms)
    ordereds_normal = _is_normal(ordereds)
    if not randoms_normal:
        print('Distribution of randoms is not normally distributed')
    if not ordereds_normal:
        print('Distribution of ordereds is not normally distributed')
    if not randoms_normal or not ordereds_normal:
        print('Cannot perform paired t test: normality assumption violated')
        return
    diffs = np.array(randoms) - np.array(ordereds)
    if _is_greater(diffs):
        print('Significantly greater')
    else:
        print('Not significantly greater')


def _try_wilcoxon(randoms, ordereds):
    """Performs Wilcoxon-Mann-Whitney test"""
    tstat, pval = scipy.stats.mannwhitneyu(
        randoms,
        ordereds,
        alternative='greater')
    print('Wilcoxon-Mann-Whitney Test:', tstat, pval)
    if pval < 0.05:
        print('Significantly greater')
    else:
        print('Not significantly greater')


def _plot_treatments(randoms, ordereds, xlabel, ylabel, filename):
    """Makes box plots of labeling times"""
    fig, axis = plt.subplots(1, 1)
    axis.boxplot([randoms, ordereds], labels=['random', 'ordered'])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    # axis.set_ylim([0, 45])
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    print('Randoms median, mean:', np.median(randoms), np.mean(randoms))
    print('Ordereds median, mean:', np.median(ordereds), np.mean(ordereds))


def _main():
    """Performs paired t test on user study data"""
    args = _parse_args()
    userdata = _parse_data(args.userdata)
    with open(args.filedict, 'rb') as ifh:
        truelabels = pickle.load(ifh)
    random_sums, ordered_sums = _get_sums(userdata)
    print('# Summed times t test')
    _try_t_test(random_sums, ordered_sums)
    random_times, ordered_times = _get_time_values(userdata)
    _try_wilcoxon(random_times, ordered_times)
    _plot_treatments(
        random_times,
        ordered_times,
        'Treatment',
        'Time (seconds)',
        'treatment_times.pdf')
    random_accs, ordered_accs = _get_accuracies(userdata, truelabels)
    print('# Accuracies')
    _try_t_test(random_accs, ordered_accs)
    _try_wilcoxon(random_accs, ordered_accs)
    _plot_treatments(
        random_accs,
        ordered_accs,
        'Treatment',
        'Accuracy',
        'treatment_accs.pdf')


if __name__ == '__main__':
    _main()
