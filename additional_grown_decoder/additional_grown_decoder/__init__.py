import math
from copy import deepcopy

import numpy as np
import networkx as nx
import stim
import pymatching
from tqdm.contrib import tzip

from ._rust_lib import UFD as RustUFD, UFD2 as RustUFD2


def calc_signed_gap(
        circuit: stim.Circuit,
        num_shots: int,
        *,
        uf_growth_rate: float = 1,
        use_preskill: bool = False,
        use_bounded_dijkstra: bool = False,
        use_additional_growth: bool = False,
        additional_max_growth: float | None = None,
        additional_max_growth_and_returns_are_dB: bool = False,
        additional_growth_rate: float | None = None,
        use_v1: bool = False,
        debug: bool = False,
        ) -> (
                list[list[bool]],
                list[list[bool]],
                dict[str, list[list[float]]],
                ):
    assert not (use_additional_growth and additional_max_growth is None)
    assert not (use_bounded_dijkstra and additional_max_growth is None)
    if additional_growth_rate is None:
        additional_growth_rate = uf_growth_rate

    ufd = UFD(circuit, uf_growth_rate, additional_growth_rate, use_v1)
    signed_gap_dict = {
            'sign': [],
            'uf_time': [],
            'preskill_time': [],
            'bounded_dijkstra_time': [],
            'additional_growth_time': [],
            'trivial_length': ufd.trivial_length,
            'debug_time': [],
            'grown_weight_ufd': [],
            }
    if use_preskill:
        signed_gap_dict['preskill_signed_gaps'] = []
        signed_gap_dict['preskill_num_nodes_of_decode_graph'] = []
        signed_gap_dict['preskill_actually_visited_nodes'] = []
    if use_bounded_dijkstra:
        signed_gap_dict['bounded_dijkstra_signed_gaps'] = []
        signed_gap_dict['bounded_dijkstra_num_nodes_of_decode_graph'] = []
        signed_gap_dict['bounded_dijkstra_actually_visited_nodes'] = []
    if use_additional_growth:
        signed_gap_dict['additional_growth_simple_signed_gaps'] = []
        signed_gap_dict['additional_growth_cluster_graph_signed_gaps'] = []
        signed_gap_dict['additional_growth_num_nodes_of_cluster_graph'] = []
        signed_gap_dict['additional_growth_actually_visited_nodes'] = []
    if debug:
        signed_gap_dict['debug_grown_edges'] = []
        signed_gap_dict['debug_growths'] = []
        signed_gap_dict['debug_det_to_de'] = []
        signed_gap_dict['debug_additional_det'] = []
        signed_gap_dict['debug_additional_collisions'] = []

    shots, observables = ufd.sampler.sample(num_shots, separate_observables=True)
    for shot, observable in tzip(shots, observables):
        #shot = np.array([False, False, False, False, False, False, False, False, False, True, True, True, False, False, False, False, False, True, True, False, False, False, False, False])
        try:
            result, preskill_result, bounded_dijkstra_result, additional_growth_result, times, grown_edges, growths, det_to_de, additional_det, additional_collisions, grown_weight_ufd = ufd.decode(
                    shot,
                    use_preskill,
                    use_bounded_dijkstra,
                    additional_max_growth if use_additional_growth else 0,
                    this_arg_and_returns_are_dB=additional_max_growth_and_returns_are_dB,
                    debug=debug,
                    )
        except:
            print(shot.tolist())
            raise Exception()
        sign = +1 if result == observable[0] else -1
        signed_gap_dict['sign'].append([sign])
        signed_gap_dict['uf_time'].append(times[0])
        signed_gap_dict['preskill_time'].append(times[1])
        signed_gap_dict['bounded_dijkstra_time'].append(times[2])
        signed_gap_dict['additional_growth_time'].append(times[3])
        signed_gap_dict['grown_weight_ufd'].append([grown_weight_ufd])
        if len(times) > 3:
            signed_gap_dict['debug_time'].append(times[3])
        if use_preskill:
            signed_gap_dict['preskill_signed_gaps'].append([sign * preskill_result[0]])
            signed_gap_dict['preskill_num_nodes_of_decode_graph'].append(preskill_result[1])
            signed_gap_dict['preskill_actually_visited_nodes'].append(preskill_result[2])
        if use_bounded_dijkstra:
            if bounded_dijkstra_result[0] is None:
                signed_gap_dict['bounded_dijkstra_signed_gaps'].append([None])
            else:
                signed_gap_dict['bounded_dijkstra_signed_gaps'].append([sign * bounded_dijkstra_result[0]])
            signed_gap_dict['bounded_dijkstra_num_nodes_of_decode_graph'].append(bounded_dijkstra_result[1])
            signed_gap_dict['bounded_dijkstra_actually_visited_nodes'].append(bounded_dijkstra_result[2])
        if use_additional_growth:
            if additional_growth_result is None:
                signed_gap_dict['additional_growth_simple_signed_gaps'].append([None])
                signed_gap_dict['additional_growth_cluster_graph_signed_gaps'].append([None])
                signed_gap_dict['additional_growth_num_nodes_of_cluster_graph'].append(0)
                signed_gap_dict['additional_growth_actually_visited_nodes'].append(0)
            else:
                signed_gap_dict['additional_growth_simple_signed_gaps'].append([sign * additional_growth_result[0]])
                signed_gap_dict['additional_growth_cluster_graph_signed_gaps'].append([sign * additional_growth_result[1]])
                signed_gap_dict['additional_growth_num_nodes_of_cluster_graph'].append(additional_growth_result[2])
                signed_gap_dict['additional_growth_actually_visited_nodes'].append(additional_growth_result[3])
        if debug:
            signed_gap_dict['debug_grown_edges'].append([grown_edges])
            signed_gap_dict['debug_growths'].append([growths])
            signed_gap_dict['debug_det_to_de'].append([det_to_de])
            signed_gap_dict['debug_additional_det'].append([additional_det])
            signed_gap_dict['debug_additional_collisions'].append([additional_collisions])

    return shots, observables, signed_gap_dict


class UFD:
    def __init__(self,
            circuit: stim.Circuit,
            uf_growth_rate: float = 1,
            additional_growth_rate: float | None = None,
            use_v1: bool = False,
            ):
        if additional_growth_rate is None:
            additional_growth_rate = uf_growth_rate

        self.circuit = circuit
        self.dem = circuit.detector_error_model(decompose_errors=True)
        self.sampler = circuit.compile_detector_sampler()
        self.graph = pymatching.Matching.from_detector_error_model(self.dem).to_networkx()
        self.graph_with_two_boundary_nodes, self.boundary_node_with_obs, self.boundary_node_without_obs = self.convert_graph_to_two_boundary_graph(self.graph)
        self.trivial_length = nx.dijkstra_path_length(self.graph_with_two_boundary_nodes, self.boundary_node_with_obs, self.boundary_node_without_obs)

        self.link_list = nx.to_dict_of_lists(self.graph_with_two_boundary_nodes)
        self.weights = {((edge_a, edge_b) if edge_a < edge_b else (edge_b, edge_a)): data['weight'] for edge_a, edge_b, data in self.graph_with_two_boundary_nodes.edges(data=True)}
        self.growth = {((edge_a, edge_b) if edge_a < edge_b else (edge_b, edge_a)): 0. for edge_a, edge_b, data in self.graph_with_two_boundary_nodes.edges(data=True)}

        self.convert_coef = 2

        self.use_v1 = use_v1
        if use_v1:
            self.rust_ufd = RustUFD(
                    self.link_list,
                    self.weights,
                    self.boundary_node_with_obs,
                    self.boundary_node_without_obs,
                    uf_growth_rate,
                    additional_growth_rate,
                    )
        else:
            self.rust_ufd = RustUFD2(
                    self.link_list,
                    self.weights,
                    self.boundary_node_with_obs,
                    self.boundary_node_without_obs,
                    self.convert_coef,
                    )

    def decode(self,
            detection_events: list[bool],
            use_preskill: bool,
            use_bounded_dijkstra: bool,
            additional_max_growth: float,
            this_arg_and_returns_are_dB: bool = False,
            debug: bool = False,
            ) -> (
            bool,
            tuple[float, int, int] | None,
            tuple[float, float, int, int] | None,
            tuple[float, float, float],
            ):
        DB_FACTOR = 1
        if this_arg_and_returns_are_dB:
            DB_FACTOR = 10 * math.log10(math.e)
            additional_max_growth /= DB_FACTOR
        if not self.use_v1:
            additional_max_growth = int(round(additional_max_growth * self.convert_coef)) * 4
            factor_after = self.convert_coef * 4
        else:
            factor_after = 1

        result, preskill_result, bounded_dijkstra_result, additional_growth_result, times, grown_edges, growths, det_to_de, additional_det, additional_collisions, grown_weight_ufd = self.rust_ufd.decode(detection_events, use_preskill, use_bounded_dijkstra, additional_max_growth, debug)
        if this_arg_and_returns_are_dB:
            if use_preskill:
                if preskill_result is None:
                    preskill_result = (self.trivial_length * DB_FACTOR, 0, 0)
                else:
                    preskill_result = (preskill_result[0] * DB_FACTOR / factor_after, preskill_result[1], preskill_result[2])
                    #assert preskill_result[0] >= 0
            if use_bounded_dijkstra:
                if bounded_dijkstra_result is None:
                    bounded_dijkstra_result = (None, 0, 0)
                elif bounded_dijkstra_result[0] is not None:
                    bounded_dijkstra_result = (bounded_dijkstra_result[0] * DB_FACTOR / factor_after, bounded_dijkstra_result[1], bounded_dijkstra_result[2])
            if additional_growth_result is not None:
                additional_growth_result = (additional_growth_result[0] * DB_FACTOR / factor_after, additional_growth_result[1] * DB_FACTOR / factor_after, additional_growth_result[2], additional_growth_result[3])
            if grown_weight_ufd is not None:
                grown_weight_ufd *= DB_FACTOR
        else:
            if use_preskill and preskill_result is None:
                preskill_result = (self.trivial_length, 0, 0)
            if bounded_dijkstra_result is None:
                bounded_dijkstra_result = (None, 0, 0)
        return result, preskill_result, bounded_dijkstra_result, additional_growth_result, times, grown_edges, growths, det_to_de, additional_det, additional_collisions, grown_weight_ufd

    @staticmethod
    def convert_graph_to_two_boundary_graph(graph: nx.Graph) -> (nx.Graph, int, int):
        graph = deepcopy(graph)

        boundary_node_with_obs = graph.number_of_nodes() - 1
        assert graph.nodes[boundary_node_with_obs]['is_boundary']
        boundary_node_without_obs = boundary_node_with_obs + 1

        for edge_a, edge_b, edge_data in list(graph.edges(boundary_node_with_obs, data=True)):
            assert edge_a == boundary_node_with_obs
            if len(edge_data['fault_ids']) == 0:
                graph.add_edge(boundary_node_without_obs, edge_b, **edge_data)
                graph.remove_edge(edge_a, edge_b)

        return graph, boundary_node_with_obs, boundary_node_without_obs
