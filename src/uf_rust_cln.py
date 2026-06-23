from typing import List, Dict, Optional
from enum import Enum, auto
from dataclasses import dataclass

from additional_grown_decoder import UFD

from surface_code import SurfaceCode


@dataclass
class DecodeResult:
    result: List[bool]
    extra: Optional[Dict[str, object]] = None


class UFRust:
    GROW_WEIGHT = 1
    DEBUG = True

    def __init__(self, sc: SurfaceCode, use_preskill: bool = False, use_bounded_dijkstra: bool = False, additional_growth: float = 0):
        self.sc = sc
        self.use_preskill = use_preskill
        self.use_bounded_dijkstra = use_bounded_dijkstra
        self.additional_growth = additional_growth
        self.ufd = UFD(
                sc.circuit,
                uf_growth_rate=UFRust.GROW_WEIGHT,
                use_v1=False,
                )

    def decode(self, detection_events: list[int]) -> DecodeResult:
        result, preskill_result, bounded_dijkstra_result, additional_growth_result, times, grown_edges, growths, det_to_de, additional_det, additional_collisions, grown_weight_ufd = self.ufd.decode(
                detection_events,
                self.use_preskill,
                self.use_bounded_dijkstra,
                self.additional_growth,
                True,
                debug=UFRust.DEBUG,
                )
        extra = {}
        if self.use_preskill:
            extra['preskill_softoutput'] = preskill_result[0]
            extra['preskill_num_nodes_of_decode_graph'] = preskill_result[1]
            extra['preskill_actually_visited_nodes'] = preskill_result[2]
        if self.use_bounded_dijkstra:
            extra['bounded_dijkstra_softoutput'] = bounded_dijkstra_result[0]
            extra['bounded_dijkstra_num_nodes_of_decode_graph'] = bounded_dijkstra_result[1]
            extra['bounded_dijkstra_actually_visited_nodes'] = bounded_dijkstra_result[2]
        if self.additional_growth > 0:
            if additional_growth_result is None:
                extra['simple_softoutput'] = None
                extra['cluster_graph_softoutput'] = None
                extra['number_of_nodes_of_cluster_graph'] = 0
                extra['actually_visited_nodes_of_cluster_graph'] = 0
            else:
                extra['simple_softoutput'] = additional_growth_result[0]
                extra['cluster_graph_softoutput'] = additional_growth_result[1]
                extra['number_of_nodes_of_cluster_graph'] = additional_growth_result[2]
                extra['actually_visited_nodes_of_cluster_graph'] = additional_growth_result[3]
        extra['uf_time'] = times[0]
        extra['preskill_time'] = times[1]
        extra['bounded_dijkstra_time'] = times[2]
        extra['additional_growth_time'] = times[3]
        if UFRust.DEBUG:
            if det_to_de is None:
                extra['uf_total_cluster_num_nodes'] = None
            else:
                extra['uf_total_cluster_num_nodes'] = len(det_to_de)
            if additional_det is None:
                extra['extra_cluster_num_nodes'] = None
            else:
                extra['extra_cluster_num_nodes'] = len(additional_det)
            if additional_collisions is None:
                extra['extra_cluster_num_collisions'] = None
            else:
                extra['extra_cluster_num_collisions'] = additional_collisions
        extra['grown_weight_ufd'] = grown_weight_ufd
        return DecodeResult([result], extra)

    @classmethod
    def get_name(cls) -> str:
        return 'ufrust-grow20dB-v2'
