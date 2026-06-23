from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

import stim


class Rec:
    def __init__(self):
        self.d = {}
        self.length = 0
        self.repeat = None # this will be a queue

    def add(self, pos: Tuple[int, int]):
        if pos not in self.d:
            self.d[pos] = []
        self.d[pos].append(self.length)
        self.length += 1

        if self.repeat is not None:
            self.repeat.append(pos)

    def get(self, pos: Tuple[int, int], idx: int = -1) -> Optional[int]:
        if pos not in self.d:
            return None
        return self.d[pos][idx] - self.length

    def repeat_start(self):
        self.repeat = []

    def repeat_end(self, num_repeat: int):
        queue = self.repeat
        self.repeat = None
        for _ in range(num_repeat-1):
            for pos in queue:
                self.add(pos)


@dataclass
class CircuitDetail:
    circuit: stim.Circuit
    pos_to_dataqubit_index_dict: Dict[Tuple[int, int], int]
    pos_to_xstabilizer_index_dict: Dict[Tuple[int, int], int]
    pos_to_zstabilizer_index_dict: Dict[Tuple[int, int], int]
    qubit_length: int
    rec: Rec
    observables: Optional[List[List[Tuple[int, int]]]] = None
