from typing import List, Dict, Tuple, Optional
from glob import glob
import os
import sys
import re

import numpy as np
from tqdm import tqdm

from generate_rotated_surface_code import generate_rotated_surface_code
from uf_rust_cln import UFRust
from surface_code import SurfaceCode
from cache import Cache


def experiment(
        *,
        total_num_experiments: int,
        d_list: Optional[List[int]] = None,
        p_list: Optional[List[float]] = None,
        multiprocessing_lock: Optional["Lock"] = None,
        cachediv: Optional[int] = None,
        ):
    decoder_cls = UFRust
    decoder_cls_params = {}
    decoder_cls_params['use_preskill'] = True
    decoder_cls_params['use_bounded_dijkstra'] = True
    decoder_cls_params['additional_growth'] = 20

    decoder_name = decoder_cls.get_name()

    if multiprocessing_lock is not None:
        multiprocessing_lock.acquire()
    cache_basename = f'results/cache-rotated_surface_code-X-{decoder_name}'
    cache = Cache(f'{cache_basename}.json' if cachediv is None else f'{cache_basename}-cachedivs/{cachediv}.json')
    if multiprocessing_lock is not None:
        multiprocessing_lock.release()

    iterated_circuit_params_list = ({'d': d, 'p': p} for d in d_list for p in p_list)

    for iterated_circuit_params in tqdm(iterated_circuit_params_list):
        d = iterated_circuit_params['d']
        p = iterated_circuit_params['p']

        sc = SurfaceCode(circuit=generate_rotated_surface_code(**iterated_circuit_params), d=d)
        decoder = decoder_cls(sc, **decoder_cls_params)

        for num_experiments in tqdm(range(total_num_experiments), leave=False):
            det_list, obs_list = sc.sample()

            decode_result = decoder.decode(det_list)

            is_logical_errors = sc.is_logical_error(actual_obs=obs_list, estimated_obs=decode_result.result)

            cache.add_value(d, p, 'num_experiments', 1)
            for i, is_logical_error in enumerate(is_logical_errors):
                cache.add_value(d, p, f'logical_error_{i}_count', 1 if is_logical_error else 0)
            cache.add_value(d, p, f'logical_error_count', 1 if any(is_logical_errors) else 0)

            cache.append_value(d, p, 'logical_error', 1 if any(is_logical_errors) else 0)
            cache.append_value(d, p, 'preskill_softoutput', decode_result.extra['preskill_softoutput'])
            cache.append_value(d, p, 'preskill_num_nodes_of_decode_graph', decode_result.extra['preskill_num_nodes_of_decode_graph'])
            cache.append_value(d, p, 'preskill_actually_visited_nodes', decode_result.extra['preskill_actually_visited_nodes'])
            cache.append_value(d, p, 'bounded_dijkstra_softoutput', decode_result.extra['bounded_dijkstra_softoutput'])
            cache.append_value(d, p, 'bounded_dijkstra_num_nodes_of_decode_graph', decode_result.extra['bounded_dijkstra_num_nodes_of_decode_graph'])
            cache.append_value(d, p, 'bounded_dijkstra_actually_visited_nodes', decode_result.extra['bounded_dijkstra_actually_visited_nodes'])
            cache.append_value(d, p, 'simple_softoutput', decode_result.extra['simple_softoutput'])
            cache.append_value(d, p, 'cluster_graph_softoutput', decode_result.extra['cluster_graph_softoutput'])
            cache.append_value(d, p, 'number_of_nodes_of_cluster_graph', decode_result.extra['number_of_nodes_of_cluster_graph'])
            cache.append_value(d, p, 'actually_visited_nodes_of_cluster_graph', decode_result.extra['actually_visited_nodes_of_cluster_graph'])
            cache.append_value(d, p, 'uf_time', decode_result.extra['uf_time'])
            cache.append_value(d, p, 'preskill_time', decode_result.extra['preskill_time'])
            cache.append_value(d, p, 'bounded_dijkstra_time', decode_result.extra['bounded_dijkstra_time'])
            cache.append_value(d, p, 'additional_growth_time', decode_result.extra['additional_growth_time'])
            if 'uf_total_cluster_num_nodes' in decode_result.extra:
                cache.append_value(d, p, 'uf_total_cluster_num_nodes', decode_result.extra['uf_total_cluster_num_nodes'])
            if 'extra_cluster_num_nodes' in decode_result.extra:
                cache.append_value(d, p, 'extra_cluster_num_nodes', decode_result.extra['extra_cluster_num_nodes'])
            if 'extra_cluster_num_collisions' in decode_result.extra:
                cache.append_value(d, p, 'extra_cluster_num_collisions', decode_result.extra['extra_cluster_num_collisions'])
            if 'grown_weight_ufd' in decode_result.extra:
                cache.append_value(d, p, 'grown_weight_ufd', decode_result.extra['grown_weight_ufd'])

        if multiprocessing_lock is not None:
            multiprocessing_lock.acquire()
            cache.reload(remain_d=d, remain_p=p)
        cache.save()
        if multiprocessing_lock is not None:
            multiprocessing_lock.release()
