"""Script to analyze user study data"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
from scipy.stats import kstest, ks_2samp, gamma, mannwhitneyu
from sklearn.linear_model import LinearRegression

import GPy


#pylint:disable-msg=too-few-public-methods
class DivergenceChecker():
    """Class for looking up document divergences"""

    def __init__(self, divergence, titles):
        """

            * divergence :: 2D numpy matrix of floats
                DxD, where D is number of documents; should line up in order as
                given by titles; each cell contains of JS divergence of topic
                mixtures of the documents compared
            * titles :: 1D numpy matrix of strings
                There should be D entries
        """
        self.divergence = divergence
        self.indexfinder = {}
        for i, title in enumerate(titles):
            self.indexfinder[title] = i

    def find_div(self, title1, title2):
        """Get divergence score between title1 and title2"""
        return self.divergence[
            self.indexfinder[title1],
            self.indexfinder[title2]]


def _parse_args():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(description='Analyze user study data')
    parser.add_argument(
        '-o',
        default='.',
        help='directory where output is to be placed; default is current ' +\
            'working directory')
    parser.add_argument(
        'userdata',
        help='directory where data is stored; assuming the data files end ' +\
            'in ".data"')
    parser.add_argument(
        'corpus',
        help='file path to pickle containing corpus information')
    parser.add_argument(
        'divergence',
        help='file path to pickle containing divergence matrix of documents')
    parser.add_argument(
        'titles',
        help='file path to pickle containing ordered titles')
    return parser.parse_args()


def _get_data(datadir):
    """Parse data files

    For each data file, there is a matrix.
        * the 0th column is the start time (in milliseconds)
        * the 1st column is the end time (in milliseconds)
        * the 2nd column is the topic number
        * the 3rd column is the document id
        * the 4th column is the user's label
    All values in the matrix are integers.
    """
    data = {}
    for filename in os.listdir(datadir):
        filename_name, filename_ext = os.path.splitext(filename)
        if filename_ext == '.data':
            parsed_data = []
            with open(os.path.join(datadir, filename)) as ifh:
                for line in ifh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parsed_data.append([int(a) for a in line.split()])
            data[filename_name] = np.array(parsed_data)
    return data


def grab_pickle(filename):
    """Load pickle"""
    with open(filename, 'rb') as ifh:
        return pickle.load(ifh)


def _get_true_labels(corpus, titles):
    """Get true labels of documents"""
    result = []
    for title in titles:
        result.append(corpus[title]['label'])
    return np.array(result)


def _get_true_labels_by_user(userdata, corpus):
    """Get true labels by user"""
    true_labels_by_user = {}
    for user in userdata:
        true_labels_by_user[user] = _get_true_labels(
            corpus,
            [str(a) for a in userdata[user][:, 3]])
    return true_labels_by_user


#pylint:disable-msg=no-member
def _plot2d_linear(xdata, ydata, _, axis):
    """Fits a linear regression to the data and plots regression line"""
    linreg = LinearRegression()
    regx = xdata.reshape((len(xdata), 1))
    regy = ydata.reshape((len(ydata), 1))
    linreg.fit(regx, regy)
    corr = linreg.score(regx, regy)
    plotx = np.linspace(xdata.min(), xdata.max())
    axis.plot(
        plotx,
        linreg.predict(plotx.reshape(len(plotx), 1)))
    axis.set_title('Linear Regression ($r^2$ = ' + str(corr) + ')')


#pylint:disable-msg=no-member
def _plot2d_gaussian(xdata, ydata, _, axis):
    """Fit a Gaussian process to the data and plots the regression"""
    sortedx = np.sort(xdata)
    meandist = np.mean(sortedx[1:] - sortedx[:-1])
    # TODO lengthscale needs to be tuned...not sure how to do that yet
    # lengthscale is set to 3*maxdist because points that far away along the x
    # axis seem like they should affect a given point
    kernel = GPy.kern.RBF(input_dim=1, lengthscale=meandist)
    gpx = xdata.reshape((xdata.size, 1))
    gpy = ydata.reshape((ydata.size, 1))
    gpr = GPy.models.GPRegression(gpx, gpy, kernel)
    gpr.optimize()
    plotx = np.linspace(xdata.min(), xdata.max(), 200)
    plotx = plotx.reshape((plotx.size, 1))
    pred_mean, _ = gpr.predict(plotx)
    pred_line = axis.plot(
        plotx,
        pred_mean,
        alpha=0.8,
        linewidth=3)[0]
    pred_quants = gpr.predict_quantiles(plotx, quantiles=(25., 75.))
    axis.plot(
        plotx,
        pred_quants[0],
        color=pred_line.get_color(),
        alpha=0.8,
        linestyle='dashed',
        linewidth=1.5)
    axis.plot(
        plotx,
        pred_quants[1],
        color=pred_line.get_color(),
        alpha=0.8,
        linestyle='dashed',
        linewidth=1.5)
    axis.set_title('Gaussian Process')


def _plot2d(filename, xdata, ydata, plot_helper):
    """Plot data as scatter with regression

    Opacity settings are dealt with in this function, so don't include alpha in
    kwargs.
    """
    fig, axis = plt.subplots(1, 1)
    axis.scatter(xdata, ydata, alpha=0.5)
    plot_helper(xdata, ydata, fig, axis)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _accuracy(guesses, true_labels):
    """Return accuracy scores"""
    return float(np.sum(guesses == true_labels)) / len(guesses)


def _mean_absolute_error(guesses, true_labels):
    """Return error scores"""
    return float(np.sum(abs(guesses - true_labels))) / len(guesses)


def _score_eval_helper(scorer):
    """Return a function that evaluates score according to scorer"""
    def _inner(guesseses, true_labelses):
        """Return scores"""
        result = []
        for guesses, true_labels in zip(guesseses, true_labelses):
            result.append(scorer(guesses, true_labels))
        return np.array(result)
    return _inner


def _totaltime_vs_finalscore(
        userdata,
        true_labels_by_user,
        filename,
        score_eval):
    """Analyze data and plot total time vs. final score, as per score_eval"""
    users = sorted(userdata.keys())
    # note that there are 60000 milliseconds per minute
    xdata = np.array(
        [
            float(userdata[user][-1, 1] - userdata[user][0, 0]) / 60000 \
            for user in users])
    ydata = score_eval(
        [userdata[user][:, -1] for user in users],
        [true_labels_by_user[user] for user in users])
    _plot2d(filename, xdata, ydata, _plot2d_linear)


def _doclength_vs_time(userdata, corpus, filename):
    """Analyze data and plot document length vs. time spent labeling document"""
    xdata = []
    ydata = []
    for user in userdata:
        docids = userdata[user][:, 3]
        doclengths = [len(corpus[str(a)]['text'].split()) for a in docids]
        xdata.extend(doclengths)
        # 1000 milliseconds per second
        times = [
            float(a) / 1000 \
            for a in (userdata[user][:, 1] - userdata[user][:, 0])]
        ydata.extend(times)
    _plot2d(filename, np.array(xdata), np.array(ydata), _plot2d_linear)


def _extract_time(_, usermatrix):
    """Extract time information for all labels except first"""
    # there are 1000 milliseconds per second
    return (usermatrix[1:, 1] - usermatrix[1:, 0]) / 1000


def _extract_reltime(_, usermatrix):
    """Extract relative time information for all labels except first"""
    times = usermatrix[1:, 1] - usermatrix[1:, 0]
    return times / times.max()


def _extract_runningaccdiff(true_labels_by_user):
    """Return a function that extracts running accuracy data"""
    def _inner(user, usermatrix):
        """Extract running accuracy data"""
        truelabels = true_labels_by_user[user]
        accuracies = truelabels == usermatrix[:, 4]
        runningacc = np.array([
            float(np.sum(accuracies[:a])) / float(a + 1) \
            for a in range(len(accuracies))])
        return runningacc[1:] - runningacc[:-1]
    return _inner


def _extract_runningtotalaccdiff(true_labels_by_user):
    """Return a function that extracts running total accuracy data"""
    def _inner(user, usermatrix):
        """Extract running total accuracy data"""
        truelabels = true_labels_by_user[user]
        accuracies = truelabels == usermatrix[:, 4]
        runningtotalacc = np.array([
            float(np.sum(accuracies[:a])) / float(len(accuracies)) \
            for a in range(len(accuracies))])
        return runningtotalacc[1:] - runningtotalacc[:-1]
    return _inner


def _extract_runningmeanerrdiff(true_labels_by_user):
    """Return a function that extracts running mean error data"""
    def _inner(user, usermatrix):
        """Extract running mean error data"""
        truelabels = true_labels_by_user[user]
        errors = np.abs(truelabels - usermatrix[:, 4])
        runningmeanerr = np.array([
            float(np.sum(errors[:a])) / float(a + 1) \
            for a in range(len(errors))])
        return runningmeanerr[1:] - runningmeanerr[:-1]
    return _inner


def _extract_runningtotalerrdiff(true_labels_by_user):
    """Return a function that extracts running total error data"""
    def _inner(user, usermatrix):
        """Extract running total error data"""
        truelabels = true_labels_by_user[user]
        errors = np.abs(truelabels - usermatrix[:, 4])
        result = [errors[0]]
        for error in errors[1:]:
            result.append(result[-1] + error)
        runningtotalerr = np.array(result)
        return runningtotalerr[1:] - runningtotalerr[:-1]
    return _inner


def _other_eval_helper(extractor):
    """Return a function that extracts data"""
    def _inner(user, usermatrix):
        """Return data"""
        return extractor(user, usermatrix)
    return _inner


def _docdiv_vs_other(userdata, s_checker, filename, other_eval):
    """Analyze data and plot document divergence vs. time spent labeling"""
    xdata = []
    ydata = []
    for user in userdata:
        docids = userdata[user][:, 3]
        divs = [
            s_checker.find_div(str(a), str(b)) \
            for a, b in zip(docids[:-1], docids[1:])]
        xdata.extend(divs)
        others = other_eval(user, userdata[user])
        ydata.extend(others)
    _plot2d(filename, np.array(xdata), np.array(ydata), _plot2d_linear)


def _get_relative_times(userdata):
    """Calculate relative times by user

    Relative time here means the proportion to the max time spent labeling for a
    given user.
    """
    result = {}
    for user in userdata:
        times = userdata[user][:, 1] - userdata[user][:, 0]
        maxtime = float(np.max(times))
        portions = []
        for time in times:
            portions.append(float(time) / maxtime)
        result[user] = np.array(portions)
    return result


def _make_boxplot(data, labels, filename):
    """Plot boxplots"""
    fig, axis = plt.subplots(1, 1)
    axis.boxplot(data, labels=labels)
    if len(data) == 2:
        test_stat, pval = ks_2samp(data[0], data[1])
        axis.set_title(
            'KS: test_stat=' + str(test_stat) + '; pval=' + str(pval))
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _switch_vs_not(userdata, filename):
    """Analyze data and make box plots of times"""
    switch = []
    notswitch = []
    for user in userdata:
        times = (userdata[user][:, 1] - userdata[user][:, 0]) / 1000
        not_switch_indices = userdata[user][:-1, 2] == userdata[user][1:, 2]
        # the first item is considered a switch
        not_switch_indices = np.insert(not_switch_indices, 0, False)
        notswitch.extend(times[not_switch_indices])
        switch.extend(times[np.logical_not(not_switch_indices)])
    _make_boxplot([switch, notswitch], ['switch', 'not switch'], filename)


def _switch_vs_not_relative(userdata, relative_times_by_user, filename):
    """Analyze data and make box plots of relative times"""
    switch = []
    notswitch = []
    for user in userdata:
        reltimes = relative_times_by_user[user]
        not_switch_indices = userdata[user][:-1, 2] == userdata[user][1:, 2]
        # the first item is considered a switch
        not_switch_indices = np.insert(not_switch_indices, 0, False)
        notswitch.extend(reltimes[not_switch_indices])
        switch.extend(reltimes[np.logical_not(not_switch_indices)])
    _make_boxplot([switch, notswitch], ['switch', 'not switch'], filename)


def _get_not_switch_indices(userdata):
    """Get array of booleans indicating whether there was a topic switch"""
    not_switch_indices = np.ones(1)
    for _, data in userdata.items():
        cur_not_switch_indices = \
            data[:-1, 2] == \
            data[1:, 2]
        if len(not_switch_indices) < len(cur_not_switch_indices):
            not_switch_indices = cur_not_switch_indices
    # the first item is considered a switch
    not_switch_indices = np.insert(not_switch_indices, 0, False)
    return not_switch_indices


def _numlabeled_vs_time(userdata, filename):
    """Analyze data and plot number of documents labeled vs. time"""
    bins = []
    for _, data in userdata.items():
        bins.append((data[:, 1] - data[:, 0]) / 1000)
    # every row is now the ith time for each user
    bins = np.array(bins).T

    not_switch_indices = _get_not_switch_indices(userdata)
    switch_indices = np.nonzero(np.logical_not(not_switch_indices))[0]
    major_locator = FixedLocator(switch_indices+1)
    major_formatter = FixedFormatter([str(i+1) for i in switch_indices])

    fig, axis = plt.subplots(1, 1)
    axis.boxplot([b for b in bins])
    axis.xaxis.set_major_locator(major_locator)
    axis.xaxis.set_major_formatter(major_formatter)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _numlabeled_vs_reltime(userdata, relative_times_by_user, filename):
    """Analyze data and plot number of documents labeled vs. relative time"""
    bins = []
    for user in userdata:
        bins.append(relative_times_by_user[user])
    # every row is now the ith relative time for each user
    bins = np.array(bins).T

    not_switch_indices = _get_not_switch_indices(userdata)
    switch_indices = np.nonzero(np.logical_not(not_switch_indices))[0]
    major_locator = FixedLocator(switch_indices+1)
    major_formatter = FixedFormatter([str(i+1) for i in switch_indices])

    fig, axis = plt.subplots(1, 1)
    axis.boxplot([b for b in bins])
    axis.xaxis.set_major_locator(major_locator)
    axis.xaxis.set_major_formatter(major_formatter)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _multiregression_helper(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user):
    """Get data necessary for multiple regression cases"""
    doclengths = []
    reltimes = []
    final_doclengths = []
    final_reltimes = []
    doc_divs = []
    for user, data in userdata.items():
        docids = data[:, 3]
        curdoclengths = [len(corpus[str(a)]['text'].split()) for a in docids]
        doclengths.extend(curdoclengths)
        final_doclengths.extend(curdoclengths[1:])
        reltimes.extend(relative_times_by_user[user])
        final_reltimes.extend(relative_times_by_user[user][1:])
        divs = [
            s_checker.find_div(str(a), str(b)) \
            for a, b in zip(docids[:-1], docids[1:])]
        doc_divs.extend(divs)
    doclengths = np.array(doclengths).reshape((len(doclengths), 1))
    reltimes = np.array(reltimes)
    final_doclengths = np.array(final_doclengths).reshape((
        len(final_doclengths), 1))
    final_reltimes = np.array(final_reltimes)
    doc_divs = np.array(doc_divs)
    return doclengths, reltimes, final_doclengths, final_reltimes, doc_divs


#pylint:disable-msg=invalid-name
def _docdiv_vs_doclength_reltime_residuals(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user,
        filename):
    """Analyze data and plot document length residuals and document divergence
    vs. relative time

    Note that the first document's length is not considered because we don't
    know what its JS divergences is comparative to nothing"""
    doclengths, reltimes, final_doclengths, final_reltimes, doc_divs = \
        _multiregression_helper(
            userdata,
            s_checker,
            corpus,
            relative_times_by_user)

    doclength_vs_reltime = LinearRegression()
    doclength_vs_reltime.fit(doclengths, reltimes)
    predictions = doclength_vs_reltime.predict(final_doclengths)
    residuals = np.abs(final_reltimes - predictions)

    _plot2d(filename, doc_divs, residuals, _plot2d_linear)


#pylint:disable-msg=too-many-locals
def _doclength_docdiv_vs_reltime(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user,
        filename):
    """Analyze data and plot document length residuals and document divergence
    vs. relative time

    Note that the first document's length is not considered because we don't
    know what its JS divergences is comparative to nothing"""
    _, _, final_doclengths, final_reltimes, doc_divs = \
        _multiregression_helper(
            userdata,
            s_checker,
            corpus,
            relative_times_by_user)

    regr = LinearRegression()
    regr.fit(
        np.array(
            [[a, b] for a, b in zip(final_doclengths.ravel(), doc_divs)]),
        final_reltimes)
    xcoords = np.linspace(0, np.max(final_doclengths), 100)
    ycoords = np.linspace(0, np.max(doc_divs), 100)
    predictions = []
    for ycoord in ycoords:
        predictions.append(
            regr.predict(
                np.array(
                    [
                        xcoords,
                        np.array([ycoord] * len(ycoords))]).T))
    fig, axis = plt.subplots(1, 1)
    plot = axis.matshow(
        np.array(predictions),
        cmap=plt.cm.YlGn,
        vmin=0.0,
        vmax=1.0)
    axis.grid()
    plt.colorbar(plot)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _firsts_vs_lasts_reltimes(userdata, relative_times_by_user, filename):
    """Analyze data and make plots of relative times"""
    firsts = []
    lasts = []
    for user, data in userdata.items():
        reltimes = relative_times_by_user[user]
        switches = data[:-1, 2] != data[1:, 2]
        switches = np.insert(switches, 0, True)
        switch_indices = np.nonzero(switches)[0]
        for i in range(3):
            firsts_indices = switch_indices+i
            firsts.extend(reltimes[firsts_indices])
            lasts_indices = switch_indices-(i+1)
            lasts.extend(reltimes[lasts_indices])
    _make_boxplot([firsts, lasts], ['firsts', 'lasts'], filename)


def _plot_table(data, cmap, highlighter, filename):
    """Plot a table

     * data :: 2-D np.array
        the data to be tabulated
     * cmap :: matplotlib.colors.Colormap
        the colormap to use; this will be scaled according to the absolute value
        of the data
     * highlighter :: float -> boolean
        function that determines whether the numbers shown are highlighted or
        not
     * filename :: str
        ouptut file name
    """
    # http://stackoverflow.com/questions/10194482/custom-matplotlib-plot-chess-board-like-table-with-colored-cells
    fig, axis = plt.subplots(1, 1)
    axis.set_axis_off()
    table = matplotlib.table.Table(axis, bbox=[0, 0, 1, 1])
    dmin, dmax = np.min(np.abs(data)), np.max(np.abs(data))
    def colorer(val):
        """Returns the proper color depending on val

        It is assumed that dmin <= val <= dmax
        """
        vrange = dmax - dmin
        return cmap((abs(val) - dmin) / vrange)
    nrows, ncols = data.shape
    width, height = 1.0/ncols, 1.0/nrows
    for (i, j), datum in np.ndenumerate(data):
        text = str(round(datum, 3))
        table.add_cell(
            i,
            j,
            width,
            height,
            text=text,
            loc='center',
            facecolor=colorer(datum))
        color = (0.0, 0.0, 0.0, 0.75)
        weight = 'normal'
        if highlighter(datum):
            color = 'c'
            weight = 'bold'
        cell = table.get_celld()[(i, j)]
        cell.get_text().set_color(color)
        cell.get_text().set_weight(weight)
    for i in range(data.shape[0]):
        table.add_cell(
            i,
            -1,
            width,
            height,
            text=str(i),
            loc='right',
            edgecolor='none',
            facecolor='none')
    for j in range(data.shape[1]):
        table.add_cell(
            -1,
            j,
            width,
            height,
            text=str(j),
            loc='center',
            edgecolor='none',
            facecolor='none')
    axis.add_table(table)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _plot_matrix(data, cmap, colorbar, filename):
    """Plot a matrix

     * data :: 2-D np.array
        that data to be matricized
     * cmap :: matplotlib.colors.Colormap
        the colormap to use
     * colorbar :: boolean
        whether to display color bar
     * filename :: str
        output file name
    """
    fig, axis = plt.subplots(1, 1)
    plot = axis.matshow(data, cmap=cmap)
    if colorbar:
        plt.colorbar(plot)
    axis.grid(b=False)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def _compare_test(sampleses, comparer, filename):
    """Plots statistical test results

    Assuming that filename ends with ".pdf"
    """
    test_stats = []
    test_pvals = []
    for samples in sampleses[:16]:
        test_stats.append([])
        test_pvals.append([])
        for other in sampleses[:16]:
            stat, pval = comparer.stat_test(samples, other)
            test_stats[-1].append(stat)
            test_pvals[-1].append(pval)
    test_pvals = np.array(test_pvals)
    np.savetxt('test_pvals.txt', test_pvals)

    _plot_table(
        test_pvals,
        plt.cm.YlGn_r,
        comparer.pval_highlighter,
        filename[:-4]+'_'+comparer.test_name+'_pvals.pdf')

    sig_pvals = []
    for row in test_pvals:
        sig_pvals.append([])
        for val in row:
            sig_pvals[-1].append(1.0 if val < 0.05 else 0.0)
    _plot_table(
        np.array(test_stats),
        plt.cm.YlGn,
        comparer.stat_highlighter,
        filename[:-4]+'_'+comparer.test_name+'_stats.pdf')


def _pad_num(i):
    """Pad number to make two characters long

    Assuming that i is an integer never greater than 99
    """
    if i < 10:
        return '0' + str(i)
    return str(i)


def _fit_gamma(sampleses, filename):
    """Fits a gamma distribution to the first 16 samples and plots the results

    Assuming that filename ends with ".pdf"
    """
    for i, samples in enumerate(sampleses[:16]):
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)
        sample_median = np.median(samples)
        shape, loc, scale = gamma.fit(samples)
        stat, pval = kstest(
            samples,
            'gamma',
            args=(shape, loc, scale))
        fig, axis = plt.subplots(1, 1)
        axis.hist(samples, normed=True)
        if i == 15:
            fig.savefig('last.pdf')
        plotx = np.linspace(np.min(samples), np.max(samples))
        axis.plot(
            plotx,
            gamma.pdf(plotx, shape, loc=loc, scale=scale),
            linewidth=3)
        axis.set_title('shape='+str(shape)+'; loc='+str(loc) + \
            '; scale='+str(scale)+'\n' + \
            'stat='+str(stat)+'; pval='+str(pval)+'\n' + \
            'mean='+str(shape*scale)+'; var='+str(shape*scale*scale)+'\n' + \
            's_mean='+str(sample_mean)+'; s_var='+str(sample_var)+'\n' + \
            's_median='+str(sample_median))
        fig.savefig(
            filename[:-4]+'_fit_'+_pad_num(i+1)+'.pdf',
            bbox_inches='tight')
        plt.close()


def _plot_stats(sampleses, filename):
    """Plot sample means, medians, and variances for the first 16 samples

    Assuming that filename ends with ".pdf"
    Also spits out text file with data
    """
    means = []
    std_devs = []
    medians = []
    first_quartile = []
    third_quartile = []
    variances = []
    for samples in sampleses[:16]:
        means.append(np.mean(samples))
        std_devs.append(np.std(samples))
        medians.append(np.median(samples))
        first_quartile.append(np.percentile(samples, 25))
        third_quartile.append(np.percentile(samples, 75))
        variances.append(np.var(samples))
    ind = np.arange(16)
    first_quartile_distance = np.array(medians) - np.array(first_quartile)
    third_quartile_distance = np.array(third_quartile) - np.array(medians)

    fig, axis = plt.subplots(1, 1)
    axis.bar(ind, means, alpha=0.9, yerr=std_devs)
    fig.savefig(filename[:-4]+'_means.pdf', bbox_inches='tight')
    plt.close()

    fig, axis = plt.subplots(1, 1)
    axis.bar(
        ind,
        medians,
        alpha=0.9,
        yerr=[first_quartile_distance, third_quartile_distance])
    fig.savefig(filename[:-4]+'_medians.pdf', bbox_inches='tight')
    plt.close()

    fig, axis = plt.subplots(1, 1)
    ind = np.arange(16)
    axis.bar(ind, variances, alpha=0.9)
    fig.savefig(filename[:-4]+'_vars.pdf', bbox_inches='tight')
    plt.close()

    with open(filename[:-4]+'_stats.txt', 'w') as ofh:
        ofh.write('# mean median variance\n')
        for men, med, vari in zip(means, medians, variances):
            ofh.write(str(men)+' '+str(med)+' '+str(vari)+'\n')


def _order_vs_times(userdata, comparers, filename):
    """Analyze data and make plots of document number within topic vs. times

    Also plot statistical tests of first 16 in order against each other
    """
    max_same = 0
    for _, data in userdata.items():
        switches = data[:-1, 2] != data[1:, 2]
        switches = np.insert(switches, 0, True)
        switch_indices = np.nonzero(switches)[0]
        same_counts = switch_indices[1:] - switch_indices[:-1]
        same_counts = np.append(
            same_counts,
            data[:, 2].shape[0] - switch_indices[-1])
        for count in same_counts:
            if count > max_same:
                max_same = count
    result = [[] for _ in range(max_same)]
    for _, data in userdata.items():
        times = (data[:, 1] - data[:, 0]) / 1000
        switches = data[:-1, 2] != data[1:, 2]
        switches = np.insert(switches, 0, True)
        switch_indices = np.nonzero(switches)[0]
        same_counts = switch_indices[1:] - switch_indices[:-1]
        same_counts = np.append(
            same_counts,
            data[:, 2].shape[0] - switch_indices[-1])
        for i, switch in enumerate(switch_indices):
            for j in range(same_counts[i]):
                result[j].append(times[switch+j])
    # I want information for the first 16 only
    _make_boxplot(result[:16], [str(i+1) for i in range(16)], filename)
    for comparer in comparers:
        _compare_test(result, comparer, filename)
    _plot_stats(result, filename)


def _order_vs_reltimes(userdata, relative_times_by_user, comparers, filename):
    """Analyze data and make plots of document number within topic vs. relative
    times

    Also plot statistical tests of first 16 in order against each other
    """
    max_same = 0
    for user, data in userdata.items():
        switches = data[:-1, 2] != data[1:, 2]
        switches = np.insert(switches, 0, True)
        switch_indices = np.nonzero(switches)[0]
        same_counts = switch_indices[1:] - switch_indices[:-1]
        same_counts = np.append(
            same_counts,
            data[:, 2].shape[0] - switch_indices[-1])
        for count in same_counts:
            if count > max_same:
                max_same = count
    result = [[] for _ in range(max_same)]
    for user, data in userdata.items():
        reltimes = relative_times_by_user[user]
        switches = data[:-1, 2] != data[1:, 2]
        switches = np.insert(switches, 0, True)
        switch_indices = np.nonzero(switches)[0]
        same_counts = switch_indices[1:] - switch_indices[:-1]
        same_counts = np.append(
            same_counts,
            data[:, 2].shape[0] - switch_indices[-1])
        for i, switch in enumerate(switch_indices):
            for j in range(same_counts[i]):
                result[j].append(reltimes[switch+j])
    # I want information for the first 16 only
    _make_boxplot(result[:16], [str(i+1) for i in range(16)], filename)
    for comparer in comparers:
        _compare_test(result, comparer, filename)
    _plot_stats(result, filename)


class Comparer:
    """Class to encapsulate functions and data for statistical tests"""

    def __init__(
            self,
            test_name,
            stat_test,
            pval_highlighter,
            stat_highlighter):
        """Constructor"""
        self.test_name = test_name
        self.stat_test = stat_test
        self.pval_highlighter = pval_highlighter
        self.stat_highlighter = stat_highlighter


def _mannwhitneyu_helper(x, y):
    """Return common language effect size and p-value"""
    u, pval = mannwhitneyu(x, y, alternative='greater')
    count = 0
    for first in x:
        for second in y:
            if first > second:
                count += 1
    total = len(x) * len(y)
    return float(count) / float(total), pval


def _pval_five(val):
    """Check whether val < 0.05"""
    return val < 0.05


def _abs_stat(val):
    """Check whether abs(val) > 0.6"""
    return abs(val) > 0.6


def _analyze_data(userdata, corpus, divergence, titles, outdir):
    """Analyze data"""
    true_labels_by_user = _get_true_labels_by_user(userdata, corpus)
    s_checker = DivergenceChecker(divergence, titles)
    relative_times_by_user = _get_relative_times(userdata)
    """
    # total time spent vs. final accuracy
    _totaltime_vs_finalscore(
        userdata,
        true_labels_by_user,
        os.path.join(outdir, 'totaltime_finalaccuracy.pdf'),
        _score_eval_helper(_accuracy))
    # total time spent vs. final average absolute error
    _totaltime_vs_finalscore(
        userdata,
        true_labels_by_user,
        os.path.join(outdir, 'totaltime_finalmae.pdf'),
        _score_eval_helper(_mean_absolute_error))
    # document length vs. time spent
    _doclength_vs_time(
        userdata,
        corpus,
        os.path.join(outdir, 'doclength_time.pdf'))
    # JS divergence of switch topics vs. time spent
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_time.pdf'),
        _other_eval_helper(_extract_time))
    # JS divergence of switch topics vs. relative time spent
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_reltime.pdf'),
        _other_eval_helper(_extract_reltime))
    # JS divergence of switch topics vs. running accuracy
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningaccdiff.pdf'),
        _other_eval_helper(_extract_runningaccdiff(true_labels_by_user)))
    # JS divergence of switch topics vs. running total accuracy
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningtotalaccdiff.pdf'),
        _other_eval_helper(_extract_runningtotalaccdiff(true_labels_by_user)))
    # JS divergence of switch topics vs. running mean error
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningmeanerrdiff.pdf'),
        _other_eval_helper(_extract_runningmeanerrdiff(true_labels_by_user)))
    # JS divergence of switch topics vs. running total error
    _docdiv_vs_other(
        userdata,
        s_checker,
        os.path.join(outdir, 'docdiv_runningtotalerrdiff.pdf'),
        _other_eval_helper(_extract_runningtotalerrdiff(true_labels_by_user)))
    # box plot:  relative time spent on switch, relative time spent not on switch
    _switch_vs_not_relative(
        userdata,
        relative_times_by_user,
        os.path.join(outdir, 'switch_relativetimes.pdf'))
    # box plot:  time spent on switch/not on switch
    _switch_vs_not(
        userdata,
        os.path.join(outdir, 'switch_times.pdf'))
    # number of docs labeled vs. relative time spent
    _numlabeled_vs_reltime(
        userdata,
        relative_times_by_user,
        os.path.join(outdir, 'numlabeled_relativetimes.pdf'))
    # number of docs labeled vs. time spent
    _numlabeled_vs_time(
        userdata,
        os.path.join(outdir, 'numlabeled_times.pdf'))
    # residual on length of docs and JS divergences of doc vs. relative time
    # spent
    _docdiv_vs_doclength_reltime_residuals(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user,
        os.path.join(outdir, 'docdiv_doclengthreltimeresiduals.pdf'))
    # heatmap of doc length and divergence vs relative time
    _doclength_docdiv_vs_reltime(
        userdata,
        s_checker,
        corpus,
        relative_times_by_user,
        os.path.join(outdir, 'doclengthdiv_reltime.pdf'))
    # box plot:  relative time spent after switch, relative time spent before
    # switch
    _firsts_vs_lasts_reltimes(
        userdata,
        relative_times_by_user,
        os.path.join(outdir, 'firsts_lasts_relativetimes.pdf'))
    """
    stat_tests = [
        Comparer('ks', ks_2samp, _pval_five, _pval_five),
        Comparer('mwu', _mannwhitneyu_helper, _pval_five, _abs_stat)]
    # box plot:  relative times per document within group
    _order_vs_reltimes(
        userdata,
        relative_times_by_user,
        stat_tests,
        os.path.join(outdir, 'order_reltime.pdf'))
    # box plot:  times per document within group
    _order_vs_times(
        userdata,
        stat_tests,
        os.path.join(outdir, 'order_time.pdf'))
    # TODO bar plot:  change in time as number of topics labeled increases


def _run():
    """Run analysis"""
    args = _parse_args()
    userdata = _get_data(args.userdata)
    corpus = grab_pickle(args.corpus)
    divergence = np.load(args.divergence)
    titles = grab_pickle(args.titles)
    _analyze_data(userdata, corpus, divergence, titles, args.o)


if __name__ == '__main__':
    _run()
