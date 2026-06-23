import stim
import numpy as np

from additional_grown_decoder import calc_signed_gap


NUM_SHOTS = 10000
d = 7
p = 0.005


def test_ufd():
    circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            rounds=d,
            distance=d,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
            after_clifford_depolarization=p
            )
    detection_events, observables, signed_gap_dict = calc_signed_gap(circuit, NUM_SHOTS, use_v1=False)
    ufd_logical_errors = ((1 - np.array(signed_gap_dict['sign'])[:, 0]) / 2).astype(bool) # the number of negative signs is the number of logical errors

    print(f'logical error rate: {ufd_logical_errors.sum()/NUM_SHOTS}')
