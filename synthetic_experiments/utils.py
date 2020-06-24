import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns

'''
These functions help with plotting
'''

def plot_results(results, x, x_label, y_label, title):

    plt.errorbar(x, [np.mean(i) for i in results], yerr=[stats.sem(i) for i in results])
    plt.xlabel(x_label, fontsize = 15)
    plt.ylabel(y_label, fontsize = 15)
    plt.title(title, fontsize = 15)
    plt.show()

def plot_results_multiple(results, x, x_labels, y_labels, num_rows, num_columns, title, subfolder, file_name):
    fig = plt.figure(figsize=(12,4))

    for i in range(num_rows):
        for j in range(num_columns):
            plt.subplot(num_rows, num_columns, i + j + 1)
            plt.errorbar(x, [np.mean(k) for k in results[i*num_columns + j]], yerr=[stats.sem(k) for k in results[i*num_columns + j]])
            plt.xlabel(x_labels[i*num_columns + j], fontsize = 15)
            plt.ylabel(y_labels[i*num_columns + j], fontsize = 15)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
    plt.tight_layout()
    fig.suptitle(title, fontsize = 20)
    fig.subplots_adjust(top=0.8)
    plt.savefig('figs/{}/{}.pdf'.format(subfolder,file_name))
    plt.show()

def plot_increasing_samples(results, bound, try_thresholds, x):

    fig = plt.figure()
    ax = plt.subplot(111)

    for t in try_thresholds:
        label = 't = {}'.format(t)
        if t == try_thresholds[-1]:
            label = 'FBTL'
        plt.errorbar(x, [np.mean(np.log10(k)) for k in results[t]], yerr=[stats.sem(np.log10(k)) for k in results[t]], label = label)
    # plt.plot(x, [np.mean(np.log10(k)) for k in bound], label = 'theoretical bound')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 13)

    plt.xlabel('number of pairwise comparisons \n on $\log_2$ scale', fontsize = 15)
    plt.ylabel('$\log_{10} \parallel w^* - \\hat{w} \parallel_2$', fontsize = 15)
    plt.xticks( fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title('Estimation error on log-log scale', fontsize = 20)
    # plt.legend(fontsize = 11)
    # plt.tight_layout()
    plt.savefig('figs/paper/multiple_threshold_error.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_threshold_sweep(results, try_thresholds, x, y_label, x_label,  title, name, log = False):

    fig = plt.figure()
    ax = plt.subplot(111)

    for t in try_thresholds:
        label = 't = {}'.format(t)
        if t == try_thresholds[-1]:
            label = 'FBTL'
        if log:
            plt.errorbar(x, [np.mean(np.log10(k)) for k in results[t]], yerr=[stats.sem(np.log10(k)) for k in results[t]], label = label)
        else:
            plt.errorbar(x, [np.mean(k) for k in results[t]], yerr=[stats.sem(k) for k in results[t]], label = label)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 13)

    plt.xlabel(x_label, fontsize = 15)
    plt.ylabel(y_label, fontsize = 15)
    # plt.ylabel('$\log \parallel w^* - \\hat{w} \parallel_2$', fontsize = 15)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title(title, fontsize = 20)
    # plt.legend(fontsize = 11)
    # plt.tight_layout()
    plt.savefig('figs/paper/threshold_sweep_{}.pdf'.format(name), bbox_extra_artists=(lgd,), bbox_inches='tight')

# def plot_threshold_sweep(results, btl_results, x, y_label, x_label,  title, name, log = False):
#     if log:
#         plt.errorbar(x, [np.mean(np.log10(k)) for k in results], yerr=[stats.sem(np.log10(k)) for k in results], label = 'salient feature MLE')
#         plt.errorbar(x, [np.mean(np.log10(k)) for k in btl_results], yerr=[stats.sem(np.log10(k)) for k in btl_results], label = 'FBTL MLE')
#     else:
#         plt.errorbar(x, [np.mean(k) for k in results], yerr=[stats.sem(k) for k in results], label = 'salient feature MLE')
#         plt.errorbar(x, [np.mean(k) for k in btl_results], yerr=[stats.sem(k) for k in btl_results], label = 'FBTL MLE')
#
#     plt.xlabel(x_label, fontsize = 15)
#     plt.ylabel(y_label, fontsize = 15)
#     # plt.ylabel('$\log \parallel w^* - \\hat{w} \parallel_2$', fontsize = 15)
#     plt.xticks(fontsize = 12)
#     plt.yticks(fontsize = 12)
#     plt.title(title, fontsize = 20)
#     plt.legend(fontsize = 11)
#     plt.tight_layout()
#     plt.savefig('figs/paper/threshold_sweep_{}.pdf'.format(name))

def plot_threshold_sweep_multiple(results, btl_results, x, y_label, x_label,  title, name, log = False):
    if log:
        plt.errorbar(x, [np.mean(np.log10(k)) for k in results], yerr=[stats.sem(np.log10(k)) for k in results], label = 'salient feature MLE')
        plt.errorbar(x, [np.mean(np.log10(k)) for k in btl_results], yerr=[stats.sem(np.log10(k)) for k in btl_results], label = 'FBTL MLE')
    else:
        plt.errorbar(x, [np.mean(k) for k in results], yerr=[stats.sem(k) for k in results], label = 'salient feature MLE')
        plt.errorbar(x, [np.mean(k) for k in btl_results], yerr=[stats.sem(k) for k in btl_results], label = 'FBTL MLE')

    plt.xlabel(x_label, fontsize = 15)
    plt.ylabel(y_label, fontsize = 15)
    # plt.ylabel('$\log \parallel w^* - \\hat{w} \parallel_2$', fontsize = 15)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title(title, fontsize = 20)
    plt.legend(fontsize = 11)
    plt.tight_layout()
    plt.savefig('figs/paper/threshold_sweep_{}.pdf'.format(name))


def plot_sst_pairwise_inconsistencies(results, x, xlabel, ylabel, title, subfolder, file_name):
    fig = plt.figure()

    plt.errorbar(x, [np.mean(k) for k in results[0]], yerr=[stats.sem(k) for k in results[0]], label = 'moderate violations')
    plt.errorbar(x, [np.mean(k) for k in results[1]], yerr=[stats.sem(k) for k in results[1]], label = 'strong violations')
    plt.errorbar(x, [np.mean(k) for k in results[2]], yerr=[stats.sem(k) for k in results[2]], label = 'weak violations')
    plt.errorbar(x, [np.mean(k) for k in results[3]], yerr=[stats.sem(k) for k in results[3]], label = 'pairwise inconsistencies')
    plt.xlabel(xlabel, fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title(title, fontsize = 20)
    plt.legend(fontsize = 11)
    plt.tight_layout()
    plt.savefig('figs/{}/sst_{}.pdf'.format(subfolder, file_name))
    plt.show()
