import stim
import numpy as np

from additional_grown_decoder import calc_signed_gap


d = 11
#p = 0.001
p = 0.001
NUM_SHOTS = 5
UF_GROWTH_RATE = 1
ADDITIONAL_MAX_GROWTH = 20 #dB
ADDITIONAL_GROWTH_RATE = 1


def test_softoutput():
    circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            rounds=d,
            distance=d,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
            after_clifford_depolarization=p
            )
    detection_events, observables, signed_gap_dict = calc_signed_gap(
            circuit,
            NUM_SHOTS,
            uf_growth_rate=UF_GROWTH_RATE,
            use_preskill=True,
            use_bounded_dijkstra=True,
            use_additional_growth=True,
            additional_max_growth=ADDITIONAL_MAX_GROWTH,
            additional_max_growth_and_returns_are_dB=True,
            additional_growth_rate=ADDITIONAL_GROWTH_RATE,
            debug=True,
            use_v1=False,
            )

    for key, val in signed_gap_dict.items():
        print(f'{key}:\n{val}')
