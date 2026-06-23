from typing import Optional, Tuple, Union, Dict, List, Literal
from itertools import product

import stim

from generate_circuit_common import Rec, CircuitDetail


def generate_rotated_surface_code(
                                  *,
                                  p: int,
                                  d: Optional[int] = None,
                                  d_height: Optional[int] = None,
                                  T: Optional[int] = None,
                                  start_qubit_index: int = 0,
                                  origin_coordinate: Tuple[int, int] = (0, 0),
                                  use_destructive_x_measurement: bool = True,
                                  use_destructive_z_measurement: bool = False,
                                  use_observable_x: bool = True,
                                  use_observable_z: bool = False,
                                  return_circuit_detail: bool = False,
                                  rec: Optional[Rec] = None,
                                  circuit: Optional[stim.Circuit] = None,
                                  pos_to_dataqubit_index_dict: Optional[Dict[Tuple[int, int], int]] = None,
                                  pos_to_xstabilizer_index_dict: Optional[Dict[Tuple[int, int], int]] = None,
                                  pos_to_zstabilizer_index_dict: Optional[Dict[Tuple[int, int], int]] = None,
                                  add_qubits_initialized_x: bool = True,
                                  add_qubits_initialized_z: bool = False,
                                  only_add_qubits: bool = False,
                                  initial_not_syndrome_measurement: Optional[List[Tuple[int, int]]] = None,
                                  initial_single_syndrome_measurement: Union[Literal['allX', 'allZ'], List[Tuple[int, int]], None] = 'allX',
                                  initial_double_syndrome_measurement: Union[Literal['all', 'allX', 'allZ'], List[Tuple[int, int]], None] = None,
                                  additional_first_detector: Optional[Dict[Tuple[int, int], List[Tuple[int, int]]]] = None,
                                  skip_reset_first_stabilizers: bool = False,
                                  ) -> Union[stim.Circuit, CircuitDetail]:
    if not T:
        T = d
    if not d_height:
        d_height = d

    if circuit is None:
        circuit = stim.Circuit()

    if pos_to_dataqubit_index_dict is None:
        pos_to_dataqubit_index_dict = {}
    if pos_to_xstabilizer_index_dict is None:
        pos_to_xstabilizer_index_dict = {}
    if pos_to_zstabilizer_index_dict is None:
        pos_to_zstabilizer_index_dict = {}

    qubit_index = start_qubit_index

    texts = []
    if rec is None:
        rec = Rec()

    # bind position to index
    if add_qubits_initialized_x or add_qubits_initialized_z:
        for y in range(0, d_height*2+1):
            if y == 0:
                for x in range(2, d*2, 4):
                    x += origin_coordinate[0]
                    y += origin_coordinate[1]

                    pos_to_zstabilizer_index_dict[(x, y)] = qubit_index
                    texts.append(f'QUBIT_COORDS({x}, {y}) {qubit_index}')
                    qubit_index += 1
            elif y == d_height*2:
                for x in range(4, d*2, 4):
                    x += origin_coordinate[0]
                    y += origin_coordinate[1]

                    pos_to_zstabilizer_index_dict[(x, y)] = qubit_index
                    texts.append(f'QUBIT_COORDS({x}, {y}) {qubit_index}')
                    qubit_index += 1
            elif y % 4 == 0:
                for x in range(0, d*2, 2):
                    if x % 4 == 0:
                        x += origin_coordinate[0]
                        y += origin_coordinate[1]
                        pos_to_xstabilizer_index_dict[(x, y)] = qubit_index
                    else:# x % 2 == 0:
                        x += origin_coordinate[0]
                        y += origin_coordinate[1]
                        pos_to_zstabilizer_index_dict[(x, y)] = qubit_index
                    texts.append(f'QUBIT_COORDS({x}, {y}) {qubit_index}')
                    qubit_index += 1
            elif y % 2 == 0:
                for x in range(2, d*2+1, 2):
                    if x % 4 == 0:
                        x += origin_coordinate[0]
                        y += origin_coordinate[1]
                        pos_to_zstabilizer_index_dict[(x, y)] = qubit_index
                    else:
                        x += origin_coordinate[0]
                        y += origin_coordinate[1]
                        pos_to_xstabilizer_index_dict[(x, y)] = qubit_index
                    texts.append(f'QUBIT_COORDS({x}, {y}) {qubit_index}')
                    qubit_index += 1
            else:
                for x in range(1, d*2, 2):
                    x += origin_coordinate[0]
                    y += origin_coordinate[1]
                    pos_to_dataqubit_index_dict[(x, y)] = qubit_index
                    texts.append(f'QUBIT_COORDS({x}, {y}) {qubit_index}')
                    qubit_index += 1

        if add_qubits_initialized_x:
            # initialize dataqubits |+>
            texts.append('RX ' + ' '.join(str(idx) for idx in pos_to_dataqubit_index_dict.values()))
        else: # add_qubits_initialized_z
            # initialize dataqubits |0>
            texts.append('R ' + ' '.join(str(idx) for idx in pos_to_dataqubit_index_dict.values()))

    if only_add_qubits:
        circuit.append_from_stim_program_text('\n'.join(texts))
        if return_circuit_detail:
            return CircuitDetail(
                                circuit=circuit,
                                pos_to_dataqubit_index_dict=pos_to_dataqubit_index_dict,
                                pos_to_xstabilizer_index_dict=pos_to_xstabilizer_index_dict,
                                pos_to_zstabilizer_index_dict=pos_to_zstabilizer_index_dict,
                                qubit_length=qubit_index,
                                rec=rec,
                                )
        return circuit

    #texts.append('TICK')

    def add_errors_after_CX():
        texts.append(f'DEPOLARIZE2({p}) {texts[-1][3:]}')
        texts.append(f'DEPOLARIZE1({p}) ' + ' '.join(str(idx) for idx in (set(pos_to_dataqubit_index_dict.values())|set(pos_to_xstabilizer_index_dict.values())|set(pos_to_zstabilizer_index_dict.values()))-{int(cx_idx) for cx_idx in texts[-2][3:].split(' ')}))

    def syndrome_measurements(skip_reset=False):
        if not skip_reset:
            texts.append('TICK')

            texts.append('RX ' + ' '.join(str(idx) for idx in pos_to_xstabilizer_index_dict.values()))
            texts.append('R ' + ' '.join(str(idx) for idx in pos_to_zstabilizer_index_dict.values()))
        if p > 0:
            texts.append(f'Z_ERROR({p}) ' + ' '.join(str(idx) for idx in pos_to_xstabilizer_index_dict.values()))
            texts.append(f'X_ERROR({p}) ' + ' '.join(str(idx) for idx in pos_to_zstabilizer_index_dict.values()))
            texts.append(f'DEPOLARIZE1({p}) ' + ' '.join(str(idx) for idx in pos_to_dataqubit_index_dict.values()))

        texts.append('TICK')

        #   ⊕   .
        # X/  Z/
        # .   ⊕
        texts.append('CX '
                            + ' '.join([f'{index_xstabilizer} {pos_to_dataqubit_index_dict[(x+1, y-1)]}' for (x, y), index_xstabilizer in pos_to_xstabilizer_index_dict.items() if (x+1, y-1) in pos_to_dataqubit_index_dict])
                            + ' '
                            + ' '.join([f'{pos_to_dataqubit_index_dict[(x+1, y-1)]} {index_zstabilizer}' for (x, y), index_zstabilizer in pos_to_zstabilizer_index_dict.items() if (x+1, y-1) in pos_to_dataqubit_index_dict]))
        if p > 0:
            add_errors_after_CX()
        texts.append('TICK')

        # X
        # .   .
        #  \   \Z
        #   ⊕   ⊕
        texts.append('CX '
                            + ' '.join([f'{index_xstabilizer} {pos_to_dataqubit_index_dict[(x+1, y+1)]}' for (x, y), index_xstabilizer in pos_to_xstabilizer_index_dict.items() if (x+1, y+1) in pos_to_dataqubit_index_dict])
                            + ' '
                            + ' '.join([f'{pos_to_dataqubit_index_dict[(x-1, y-1)]} {index_zstabilizer}' for (x, y), index_zstabilizer in pos_to_zstabilizer_index_dict.items() if (x-1, y-1) in pos_to_dataqubit_index_dict]))
        if p > 0:
            add_errors_after_CX()
        texts.append('TICK')

        #     Z
        # ⊕   ⊕
        #  \X  \
        #   .   .
        texts.append('CX '
                            + ' '.join([f'{index_xstabilizer} {pos_to_dataqubit_index_dict[(x-1, y-1)]}' for (x, y), index_xstabilizer in pos_to_xstabilizer_index_dict.items() if (x-1, y-1) in pos_to_dataqubit_index_dict])
                            + ' '
                            + ' '.join([f'{pos_to_dataqubit_index_dict[(x+1, y+1)]} {index_zstabilizer}' for (x, y), index_zstabilizer in pos_to_zstabilizer_index_dict.items() if (x+1, y+1) in pos_to_dataqubit_index_dict]))
        if p > 0:
            add_errors_after_CX()
        texts.append('TICK')

        #   X   Z
        #   .   ⊕
        #  /   /
        # ⊕   .
        texts.append('CX '
                            + ' '.join([f'{index_xstabilizer} {pos_to_dataqubit_index_dict[(x-1, y+1)]}' for (x, y), index_xstabilizer in pos_to_xstabilizer_index_dict.items() if (x-1, y+1) in pos_to_dataqubit_index_dict])
                            + ' '
                            + ' '.join([f'{pos_to_dataqubit_index_dict[(x-1, y+1)]} {index_zstabilizer}' for (x, y), index_zstabilizer in pos_to_zstabilizer_index_dict.items() if (x-1, y+1) in pos_to_dataqubit_index_dict]))
        if p > 0:
            add_errors_after_CX()
        texts.append('TICK')

        # measure ancilla qubits
        if p > 0:
            texts.append(f'Z_ERROR({p}) ' + ' '.join(str(idx) for idx in pos_to_xstabilizer_index_dict.values()))
            texts.append(f'X_ERROR({p}) ' + ' '.join(str(idx) for idx in pos_to_zstabilizer_index_dict.values()))
            texts.append(f'DEPOLARIZE1({p}) ' + ' '.join(str(idx) for idx in pos_to_dataqubit_index_dict.values()))
        texts.append('MX ' + ' '.join(str(idx) for idx in pos_to_xstabilizer_index_dict.values()))
        texts.append('M ' + ' '.join(str(idx) for idx in pos_to_zstabilizer_index_dict.values()))

        # register with rec
        for (x, y) in pos_to_xstabilizer_index_dict:
            rec.add((x, y))
        for (x, y) in pos_to_zstabilizer_index_dict:
            rec.add((x, y))

    syndrome_measurements(skip_reset=skip_reset_first_stabilizers)

    if initial_single_syndrome_measurement == 'allX':
        assert initial_double_syndrome_measurement not in ('all', 'allX')
        for (x, y), index_xstabilizer in pos_to_xstabilizer_index_dict.items():
            if initial_not_syndrome_measurement is not None and (x, y) in initial_not_syndrome_measurement:
                continue
            texts.append(f'DETECTOR({x}, {y}, 0) rec[{rec.get((x, y))}]')
            if additional_first_detector is not None and (x, y) in additional_first_detector:
                texts[-1] += ' ' + ' '.join([f'rec[{rec.get(rec_pos)}]' for rec_pos in additional_first_detector[(x, y)]])
    elif initial_single_syndrome_measurement == 'allZ':
        assert initial_double_syndrome_measurement not in ('all', 'allZ')
        for (x, y), index_zstabilizer in pos_to_zstabilizer_index_dict.items():
            if initial_not_syndrome_measurement is not None and (x, y) in initial_not_syndrome_measurement:
                continue
            texts.append(f'DETECTOR({x}, {y}, 0) rec[{rec.get((x, y))}]')
            if additional_first_detector is not None and (x, y) in additional_first_detector:
                texts[-1] += ' ' + ' '.join([f'rec[{rec.get(rec_pos)}]' for rec_pos in additional_first_detector[(x, y)]])
    elif initial_single_syndrome_measurement is None:
        pass
    else:
        for (x, y) in initial_single_syndrome_measurement:
            assert not (initial_not_syndrome_measurement is not None and (x, y) in initial_not_syndrome_measurement)
            texts.append(f'DETECTOR({x}, {y}, 0) rec[{rec.get((x, y))}]')
            if additional_first_detector is not None and (x, y) in additional_first_detector:
                texts[-1] += ' ' + ' '.join([f'rec[{rec.get(rec_pos)}]' for rec_pos in additional_first_detector[(x, y)]])

    if initial_double_syndrome_measurement == 'all':
        for (x, y), index_stabilizer in {**pos_to_xstabilizer_index_dict, **pos_to_zstabilizer_index_dict}.items():
            if initial_not_syndrome_measurement is not None and (x, y) in initial_not_syndrome_measurement:
                continue
            texts.append(f'DETECTOR({x}, {y}, 0) rec[{rec.get((x, y))}] rec[{rec.get((x, y), -2)}]')
            if additional_first_detector is not None and (x, y) in additional_first_detector:
                texts[-1] += ' ' + ' '.join([f'rec[{rec.get(rec_pos)}]' for rec_pos in additional_first_detector[(x, y)]])
    elif initial_double_syndrome_measurement == 'allX':
        for (x, y), index_xstabilizer in pos_to_xstabilizer_index_dict.items():
            if initial_not_syndrome_measurement is not None and (x, y) in initial_not_syndrome_measurement:
                continue
            texts.append(f'DETECTOR({x}, {y}, 0) rec[{rec.get((x, y))}] rec[{rec.get((x, y), -2)}]')
            if additional_first_detector is not None and (x, y) in additional_first_detector:
                texts[-1] += ' ' + ' '.join([f'rec[{rec.get(rec_pos)}]' for rec_pos in additional_first_detector[(x, y)]])
    elif initial_double_syndrome_measurement == 'allZ':
        for (x, y), index_zstabilizer in pos_to_zstabilizer_index_dict.items():
            if initial_not_syndrome_measurement is not None and (x, y) in initial_not_syndrome_measurement:
                continue
            texts.append(f'DETECTOR({x}, {y}, 0) rec[{rec.get((x, y))}] rec[{rec.get((x, y), -2)}]')
            if additional_first_detector is not None and (x, y) in additional_first_detector:
                texts[-1] += ' ' + ' '.join([f'rec[{rec.get(rec_pos)}]' for rec_pos in additional_first_detector[(x, y)]])
    elif initial_double_syndrome_measurement is None:
        pass
    else:
        for (x, y) in initial_double_syndrome_measurement:
            assert not (initial_not_syndrome_measurement is not None and (x, y) in initial_not_syndrome_measurement)
            texts.append(f'DETECTOR({x}, {y}, 0) rec[{rec.get((x, y))}] rec[{rec.get((x, y), -2)}]')
            if additional_first_detector is not None and (x, y) in additional_first_detector:
                texts[-1] += ' ' + ' '.join([f'rec[{rec.get(rec_pos)}]' for rec_pos in additional_first_detector[(x, y)]])



    texts.append('SHIFT_COORDS(0, 0, 1)')

    repeat_num_rounds = T-1# if use_initial_syndrome_measurement else T
    if repeat_num_rounds >= 1:
        # start of repeat
        texts.append('REPEAT ' + str(repeat_num_rounds) + ' {')
        rec.repeat_start()

        syndrome_measurements()

        # set detectors
        for (x, y), index_xstabilizer in pos_to_xstabilizer_index_dict.items():
            texts.append(f'DETECTOR({x}, {y}, 0) rec[{rec.get((x, y))}] rec[{rec.get((x, y), -2)}]')
        for (x, y), index_zstabilizer in pos_to_zstabilizer_index_dict.items():
            texts.append(f'DETECTOR({x}, {y}, 0) rec[{rec.get((x, y))}] rec[{rec.get((x, y), -2)}]')

        texts.append('SHIFT_COORDS(0, 0, 1)')

        # end of repeat
        texts.append('}')
        rec.repeat_end(num_repeat=repeat_num_rounds)

    assert not (use_destructive_x_measurement and use_destructive_z_measurement)
    if use_destructive_x_measurement or use_destructive_z_measurement:
        if use_destructive_x_measurement:
            texts.append('MX ' + ' '.join(str(idx) for idx in pos_to_dataqubit_index_dict.values()))
        elif use_destructive_z_measurement:
            texts.append('M ' + ' '.join(str(idx) for idx in pos_to_dataqubit_index_dict.values()))
        # register with rec
        for (x, y) in pos_to_dataqubit_index_dict:
            rec.add((x, y))
        for (x, y) in pos_to_xstabilizer_index_dict if use_destructive_x_measurement else pos_to_zstabilizer_index_dict:
            texts.append(f'DETECTOR({x}, {y}, 0) ' + ' '.join([f'rec[{rec.get((x+shift_x, y+shift_y))}]' for shift_x, shift_y in product((-1, 1), repeat=2) if (x+shift_x, y+shift_y) in pos_to_dataqubit_index_dict]) + f' rec[{rec.get((x, y))}]')

    if use_observable_x:
        texts.append(f'OBSERVABLE_INCLUDE(0) ' + ' '.join([f'rec[{rec.get((x, 1))}]' for x, y in reversed(pos_to_dataqubit_index_dict) if y == 1]))
    if use_observable_z:
        texts.append(f'OBSERVABLE_INCLUDE(0) ' + ' '.join([f'rec[{rec.get((1, y))}]' for x, y in reversed(pos_to_dataqubit_index_dict) if x == 1]))


    circuit.append_from_stim_program_text('\n'.join(texts))


    if return_circuit_detail:
        return CircuitDetail(
                            circuit=circuit,
                            pos_to_dataqubit_index_dict=pos_to_dataqubit_index_dict,
                            pos_to_xstabilizer_index_dict=pos_to_xstabilizer_index_dict,
                            pos_to_zstabilizer_index_dict=pos_to_zstabilizer_index_dict,
                            qubit_length=qubit_index,
                            rec=rec,
                            )
    return circuit


if __name__ == '__main__':
    circuit = generate_rotated_surface_code(d=7, p=0.001, T=7)#.without_noise()
    #circuit = stim.Circuit.generated('surface_code:rotated_memory_x', distance=5, rounds=7,
    #after_clifford_depolarization=0.001,
    #before_round_data_depolarization=0.001,
    #before_measure_flip_probability=0.001,
    #after_reset_flip_probability=0.001,
    #).without_noise()
    print(circuit)
    with open('img/rotated_surface_code-timeline.svg', 'w') as f:
        print(circuit.diagram('timeline-svg'), file=f)
    with open('img/rotated_surface_code-timeslice.svg', 'w') as f:
        print(circuit.diagram('timeslice-svg'), file=f)
    with open('img/rotated_surface_code-detector-slice.svg', 'w') as f:
        print(circuit.diagram('detector-slice-svg'), file=f) # equivalent with detslice-svg
    with open('img/rotated_surface_code-detslice-with-ops.svg', 'w') as f:
        print(circuit.diagram('detslice-with-ops-svg'), file=f)
    with open('img/rotated_surface_code-match-graph.svg', 'w') as f:
        print(circuit.diagram('match-graph-svg'), file=f)
    with open('img/rotated_surface_code-match-graph-3d.html', 'w') as f:
        print(circuit.diagram('matchgraph-3d-html'), file=f)
