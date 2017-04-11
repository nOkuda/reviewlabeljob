"""Playground for making sure that I'm doing things right"""
import scipy.stats


def _greater_onetail():
    """One-tailed t-test, greater than"""
    # http://stackoverflow.com/questions/15984221
    d1 = scipy.stats.norm.rvs(loc=1, size=50)
    d2 = scipy.stats.norm.rvs(loc=10, size=50)
    diff = d2 - d1
    tstat, pval = scipy.stats.ttest_1samp(diff, 0, nan_policy='raise')
    print(tstat, pval)
    if pval / 2 < 0.05 and tstat > 0:
        print('Significantly greater')
    else:
        print('Not significantly greater')


if __name__ == '__main__':
    _greater_onetail()
