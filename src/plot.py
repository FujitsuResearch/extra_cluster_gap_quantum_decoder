from typing import Literal, Optional
from itertools import count
from glob import glob
import re
import sys
import math

import matplotlib.pyplot as plt
import numpy as np

from cache import Cache



def get_color_idx_dict() -> dict[str, int]:
    return {'0.0001': 0, '0.0005': 1, '0.001': 2, '0.005': 3, '0.01': 4}

def get_marker_dict() -> dict[str, str]:
    return {'0.0001': 'o', '0.0005': 's', '0.001': '^', '0.005': 'D', '0.01': 'v'}


def plot_correlation(circuit_name: str, decoder_name: str, base: Literal['X', 'Z'] = 'X', key1: str = 'preskill_softoutput', key2: str = 'cluster_graph_softoutput', not_fitting: bool = True):
    cache = Cache(f'results/cache-{circuit_name}-{base}-{decoder_name}.json')
    cache.load_all_cachedivs()
    backup_font_size = plt.rcParams['font.size']
    plt.rcParams['font.size'] = 40
    fig, axs = plt.subplots(5, len(cache.cache), layout='constrained')
    less_than_20dB_ratio = {}
    ax_row = 0
    for d, cache_d in sorted(cache.cache.items(), key=lambda x:int(x[0])):
        ax_col = 0
        for p, cache_p in sorted(cache_d.items(), key=lambda x:float(x[0])):
            if p not in ('0.0001', '0.0005', '0.001', '0.005', '0.01'):
                continue

            if axs.ndim == 1:
                ax = axs[ax_row]
            else:
                ax = axs[ax_col, ax_row]
            ax_col += 1

            key1_values = cache_p[key1]
            key2_values = cache_p[key2]
            ax.scatter(key1_values, key2_values)
            ax.set_title(f'd={d}, p={p}')
            max_val = 20
            for val1, val2 in zip(key1_values, key2_values):
                if val1 is not None and val2 is not None:
                    max_val = max([max_val, val1, val2])

            ax.vlines(20, 0, max_val, colors='black', linestyles='--')
            ax.hlines(20, 0, max_val, colors='black', linestyles='--')
            ax.plot([0, max_val], [0, max_val], color='black', linestyle=':')

            key1_values = np.array(key1_values)
            key2_values = np.array(key2_values)
            dB = int(re.search('(\d+)dB', decoder_name).group(1))
            not_none_and_less = (key2_values[(key2_values != None)]<=dB).sum()
            none_or_more = (key2_values == None).sum() + (key2_values[(key2_values != None)] > dB).sum()
            less_preskill = len(key1_values[key1_values<=dB])
            more_preskill = len(key1_values[key1_values>dB])
            print(f'd{d}-p{p}; !None and <={dB}dB:{not_none_and_less},<={dB}dB:{less_preskill}; None or >{dB}dB:{none_or_more},>{dB}dB:{more_preskill}')
            copied_key2_values = key2_values.copy()
            copied_key2_values[copied_key2_values == None] = dB + 1
            print(f'{key1}<={dB}dB & {key2}<={dB}dB samples : {key1}<={dB}dB samples = {((key1_values <= dB) & (copied_key2_values <= dB)).sum()} : {less_preskill}')

            if not_none_and_less == 0:
                continue
            if p not in less_than_20dB_ratio:
                less_than_20dB_ratio[p] = {
                        'd': [],
                        'not_none_and_less': [],
                        'not_none_and_less-se': [],
                        'less_preskill': [],
                        'less_preskill-se': [],
                        }
            less_than_20dB_ratio[p]['d'].append(int(d))
            num = cache_p['num_experiments']
            less_than_20dB_ratio[p]['not_none_and_less'].append(not_none_and_less / num)
            less_than_20dB_ratio[p]['not_none_and_less-se'].append(math.sqrt(not_none_and_less) / num)
            less_than_20dB_ratio[p]['less_preskill'].append(less_preskill / num)
            less_than_20dB_ratio[p]['less_preskill-se'].append(math.sqrt(less_preskill) / num)

        ax_row += 1
    supxlabel = key1
    supylabel = key2
    if key1 == 'preskill_softoutput':
        supxlabel = 'cluster gap'
    if key2 == 'cluster_graph_softoutput':
        supylabel = 'extra-cluster gap w/ CG'
    elif key2 == 'bounded_dijkstra_softoutput':
        supylabel = 'bounded cluster gap'
    elif key2 == 'simple_softoutput':
        supylabel = 'extra-cluster gap w/o CG'
    fig.supxlabel(supxlabel)
    fig.supylabel(supylabel)
    if axs.ndim > 1:
        fig.set_figwidth(len(axs[0])*6)
    fig.set_figheight(len(axs)*6)
    plt.savefig(f'plots/{circuit_name}-{base}-corr-{decoder_name}-{key1}-{key2}.png', bbox_inches='tight')

    plt.rcParams['font.size'] = backup_font_size

    if key2 != 'simple_softoutput':
        return

    all_d_set = set()
    fig, ax = plt.subplots()
    color_idx_dict = get_color_idx_dict()#{p: idx for idx, p in enumerate(sorted(less_than_20dB_ratio, key=float))}
    markers = get_marker_dict()
    for i, (p, dict_) in enumerate(less_than_20dB_ratio.items()):
        mean = np.array(dict_['not_none_and_less'])
        se = np.array(dict_['not_none_and_less-se'])
        ax.plot(dict_['d'], mean, label=f'p={float(p):.2%}', color=f'C{color_idx_dict[p]}', marker=markers[p])
        ax.fill_between(dict_['d'], mean-se, mean+se, alpha=0.2, fc=f'C{color_idx_dict[p]}')
        all_d_set |= set(dict_['d'])

        if not_fitting:
            continue
        indices_not_zero = np.where(mean!=0)[0]
        logmean = np.log10(mean[indices_not_zero])
        d = np.array(dict_['d'])
        coef = np.polyfit(d[indices_not_zero], logmean, 1)
        ax.plot(d, 10 ** coef[1] * np.exp(coef[0] * np.log(10) * d), label=f'p={float(p):.2%}(fit: 10^{coef[1]:.2f} 10^({coef[0]:.2f}d))', color=f'C{color_idx_dict[p]}', linestyle=':')
    for i, (p, dict_) in enumerate(less_than_20dB_ratio.items()):
        mean = np.array(dict_['less_preskill'])
        se = np.array(dict_['less_preskill-se'])
        ax.plot(dict_['d'], mean, label=f'p={float(p):.2%}', color=f'C{color_idx_dict[p]}', linestyle=':', marker=markers[p])
        ax.fill_between(dict_['d'], mean-se, mean+se, alpha=0.2, fc=f'C{color_idx_dict[p]}')

        if not_fitting:
            continue
        indices_not_zero = np.where(mean!=0)[0]
        logmean = np.log10(mean[indices_not_zero])
        d = np.array(dict_['d'])
        if len(indices_not_zero) > 0:
            coef = np.polyfit(d[indices_not_zero], logmean, 1)
            ax.plot(d, 10 ** coef[1] * np.exp(coef[0] * np.log(10) * d), label=f'p={float(p):.2%}(preskill fit: 10^{coef[1]:.2f} 10^({coef[0]:.2f}d))', color=f'C{color_idx_dict[p]}', linestyle='-.')
    if not_fitting:
        ax.legend(ncol=2, loc='center right', bbox_to_anchor=(1.0, 0.65))
    else:
        ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(which='both')
    ax.set_xticks(sorted(all_d_set))
    ax.set_yscale('log')
    ax.set_xlabel('Code distance $d$', fontsize=16)
    ax.set_ylabel(f'Ratio of samples $\\leq${dB} dB', fontsize=16)
    plt.savefig(f'plots/{circuit_name}-{base}-simple_softoutput_too_rejecting_ratio-{decoder_name}{"" if not_fitting else "-withfitting"}.png', bbox_inches="tight")


def plot_actually_visited_nodes(circuit_name: str, decoder_name: str, base: Literal['X', 'Z'] = 'X', not_fitting: bool = True):
    cache = Cache(f'results/cache-{circuit_name}-{base}-{decoder_name}.json')
    cache.load_all_cachedivs()

    number_of_nodes_of_cluster_graph = cache.get_extra('bounded_dijkstra_actually_visited_nodes', ignore_0=True)
    all_d_set = set()
    fig, ax = plt.subplots()
    color_idx_dict = get_color_idx_dict()#{p: idx for idx, p in enumerate(sorted(number_of_nodes_of_cluster_graph, key=float))}
    markers = get_marker_dict()
    for i, (p, dict_) in enumerate(number_of_nodes_of_cluster_graph.items()):
        if p not in ('0.0001', '0.0005', '0.001', '0.005', '0.01'):
            continue
        mean = np.array(dict_['bounded_dijkstra_actually_visited_nodes(mean)'])
        std = np.array(dict_['bounded_dijkstra_actually_visited_nodes(std)'])
        ax.plot(dict_['d'], mean, label=f'p={float(p):.2%}', color=f'C{color_idx_dict[p]}', marker=markers[p])
        ax.fill_between(dict_['d'], mean-std, mean+std, alpha=0.2, fc=f'C{color_idx_dict[p]}')
        all_d_set |= set(dict_['d'])

        if not_fitting:
            continue
        logmean = np.log10(mean)
        logd = np.log10(dict_['d'])
        coef = np.polyfit(logd[2:], logmean[2:], 1) # fitting on d>=7
        ax.plot(dict_['d'], 10 ** coef[1] * np.array(dict_['d'])**coef[0], label=f'p={float(p):.2%}(bounded cluster gap fit: 10^{coef[1]:.2f} d^{coef[0]:.2f})', color=f'C{color_idx_dict[p]}', linestyle=':')
    preskill_num_nodes_of_decode_graph = cache.get_extra('preskill_actually_visited_nodes', ignore_0=True)
    for i, (p, dict_) in enumerate(preskill_num_nodes_of_decode_graph.items()):
        if p not in ('0.0001', '0.0005', '0.001', '0.005', '0.01'):
            continue
        mean = np.array(dict_['preskill_actually_visited_nodes(mean)'])
        std = np.array(dict_['preskill_actually_visited_nodes(std)'])
        ax.plot(dict_['d'], mean, label=f'p={float(p):.2%}', color=f'C{color_idx_dict[p]}', linestyle=':', marker=markers[p])
        ax.fill_between(dict_['d'], mean-std, mean+std, alpha=0.2, fc=f'C{color_idx_dict[p]}')

        if not_fitting:
            continue
        logmean = np.log10(mean)
        logd = np.log10(dict_['d'])
        coef = np.polyfit(logd[2:], logmean[2:], 1) # fitting on d>=7
        ax.plot(dict_['d'], 10 ** coef[1] * np.array(dict_['d'])**coef[0], label=f'p={float(p):.2%}(cluster gap fit: 10^{coef[1]:.2f} d^{coef[0]:.2f})', color=f'C{color_idx_dict[p]}', linestyle='-.')

    if not_fitting:
        ax.legend(ncol=2)
    else:
        ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(which='both')
    ax.set_xticks(sorted(all_d_set))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(ymax=10**4)
    ax.set_xlabel('Code distance $d$', fontsize=16)
    ax.set_ylabel('Number of nodes', fontsize=16)
    plt.savefig(f'plots/{circuit_name}-{base}-actually_visited_nodes-{decoder_name}{"" if not_fitting else "-withfitting"}.png', bbox_inches="tight")


def plot_cluster_nodes(circuit_name: str, decoder_name: str, base: Literal['X', 'Z'] = 'X'):
    cache = Cache(f'results/cache-{circuit_name}-{base}-{decoder_name}.json')
    cache.load_all_cachedivs()
    number_of_nodes_of_cluster_graph = cache.get_extra('uf_total_cluster_num_nodes', ignore_none=True)
    all_d_set = set()
    fig, ax = plt.subplots()
    color_idx_dict = get_color_idx_dict()
    markers = get_marker_dict()
    for i, (p, dict_) in enumerate(number_of_nodes_of_cluster_graph.items()):
        if p not in ('0.0001', '0.0005', '0.001', '0.005', '0.01'):
            continue

        mean = np.array(dict_['uf_total_cluster_num_nodes(mean)'])
        std = np.array(dict_['uf_total_cluster_num_nodes(std)'])

        logmean = np.log10(mean)
        logd = np.log10(dict_['d'])
        coef = np.polyfit(logd[2:], logmean[2:], 1)

        ax.plot(dict_['d'], mean, label=f'p={float(p):.2%}(fit:$B={coef[0]:.2f}$)', color=f'C{color_idx_dict[p]}', marker=markers[p])
        ax.fill_between(dict_['d'], mean-std, mean+std, alpha=0.2, fc=f'C{color_idx_dict[p]}')
        all_d_set |= set(dict_['d'])

        ax.plot(dict_['d'], 10 ** coef[1] * np.array(dict_['d'])**coef[0], color=f'C{color_idx_dict[p]}', linestyle=':')

    ax.legend()
    ax.grid(which='both')
    ax.set_xticks(sorted(all_d_set))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Code distance $d$', fontsize=16)
    ax.set_ylabel('Number of nodes', fontsize=16)
    plt.savefig(f'plots/{circuit_name}-{base}-uf_nodes-{decoder_name}.png', bbox_inches="tight")


    number_of_nodes_of_cluster_graph = cache.get_extra('extra_cluster_num_nodes', ignore_none=True)
    all_d_set = set()
    fig, ax = plt.subplots()
    for i, (p, dict_) in enumerate(number_of_nodes_of_cluster_graph.items()):
        if p not in ('0.0001', '0.0005', '0.001', '0.005', '0.01'):
            continue

        mean = np.array(dict_['extra_cluster_num_nodes(mean)'])
        std = np.array(dict_['extra_cluster_num_nodes(std)'])

        logmean = np.log10(mean)
        logd = np.log10(dict_['d'])
        coef = np.polyfit(logd[2:], logmean[2:], 1)

        ax.plot(dict_['d'], mean, label=f'p={float(p):.2%}(fit:$B={coef[0]:.2f}$)', color=f'C{color_idx_dict[p]}', marker=markers[p])
        ax.fill_between(dict_['d'], mean-std, mean+std, alpha=0.2, fc=f'C{color_idx_dict[p]}')
        all_d_set |= set(dict_['d'])

        if len(mean) <= 3:
            continue
        logmean = np.log10(mean)
        logd = np.log10(dict_['d'])
        coef = np.polyfit(logd[2:], logmean[2:], 1)
        ax.plot(dict_['d'], 10 ** coef[1] * np.array(dict_['d'])**coef[0], color=f'C{color_idx_dict[p]}', linestyle=':')
    ax.legend()
    ax.grid(which='both')
    ax.set_xticks(sorted(all_d_set))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Code distance $d$', fontsize=16)
    ax.set_ylabel('Number of nodes', fontsize=16)
    plt.savefig(f'plots/{circuit_name}-{base}-only_extra_nodes-{decoder_name}.png', bbox_inches="tight")


    number_of_nodes_of_cluster_graph = cache.get_extra('grown_weight_ufd', ignore_none=True, factor=1/8)
    all_d_set = set()
    fig, ax = plt.subplots()
    for i, (p, dict_) in enumerate(number_of_nodes_of_cluster_graph.items()):
        if p not in ('0.0001', '0.0005', '0.001', '0.005', '0.01'):
            continue

        mean = np.array(dict_['grown_weight_ufd(mean)'])
        std = np.array(dict_['grown_weight_ufd(std)'])
        ax.plot(dict_['d'], mean, label=f'p={float(p):.2%}', color=f'C{color_idx_dict[p]}', marker=markers[p], linestyle=':')
        ax.fill_between(dict_['d'], mean-std, mean+std, alpha=0.2, fc=f'C{color_idx_dict[p]}')
        all_d_set |= set(dict_['d'])

        continue
        if len(mean) <= 3:
            continue
        logmean = np.log10(mean)
        logd = np.log10(dict_['d'])
        coef = np.polyfit(logd[2:], logmean[2:], 1)
        ax.plot(dict_['d'], 10 ** coef[1] * np.array(dict_['d'])**coef[0], label=f'p={float(p):.2%}(ufd weight fit: 10^{coef[1]:.2f} d^{coef[0]:.2f})', color=f'C{color_idx_dict[p]}', linestyle=':')
    ax.plot([min(all_d_set), max(all_d_set)], [10, 10], color='black', label=r'$\epsilon_\mathrm{max}/2=10$ dB', linestyle='--')
    ax.legend(ncol=2, loc='lower right')
    ax.grid(which='both')
    ax.set_xticks(sorted(all_d_set))
    ax.set_ylim(ymin=0)
    ax.set_xlabel('Code distance $d$', fontsize=16)
    ax.set_ylabel('Grown UFD weight (dB)', fontsize=16)
    plt.savefig(f'plots/{circuit_name}-{base}-ufd_weight-{decoder_name}.png', bbox_inches="tight")


def plot_helios_lut():
    fig, ax = plt.subplots()
    d = np.array([3, 5, 9, 13, 17])
    mean = np.array([1557, 11128, 93797, 340084, 888854])
    ax.plot(d, mean, label='#LUTs', marker='o')
    logmean = np.log10(mean)
    logd = np.log10(d)
    coef = np.polyfit(logd, logmean, 1)
    ax.plot(d, 10 ** coef[1] * np.array(d)**coef[0], label=f'fit: 10^{coef[1]:.2f} d^{coef[0]:.2f}', linestyle=':', color='C0')
    ax.legend()
    ax.grid(which='both')
    ax.set_xticks(d)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Code distance $d$', fontsize=16)
    ax.set_ylabel('#LUTs', fontsize=16)
    plt.savefig(f'plots/lut-by-helios.png', bbox_inches="tight")




if __name__ == '__main__':
    circuit_name = 'rotated_surface_code'
    decoder_name = 'ufrust-grow20dB-v2'
    base = 'X'

    plot_correlation(circuit_name, decoder_name, base, 'preskill_softoutput', 'simple_softoutput', True)
    plot_correlation(circuit_name, decoder_name, base, 'preskill_softoutput', 'simple_softoutput', False)
    plot_correlation(circuit_name, decoder_name, base, 'preskill_softoutput', 'cluster_graph_softoutput')
    plot_correlation(circuit_name, decoder_name, base, 'preskill_softoutput', 'bounded_dijkstra_softoutput')
    plot_actually_visited_nodes(circuit_name, decoder_name, base, True)
    plot_actually_visited_nodes(circuit_name, decoder_name, base, False)
    plot_cluster_nodes(circuit_name, decoder_name, base)
    plot_helios_lut()
