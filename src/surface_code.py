from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from copy import deepcopy
import sys
import re
import json

import numpy as np
import stim
import pymatching
import networkx as nx


class SurfaceCode:
    def __init__(self,
                 *,
                 circuit: Optional["stim.Circuit"] = None,
                 d: Optional[int] = None,
                 p: Optional[float] = None,
                 T: Optional[int] = None,
                 ):
        self.circuit = circuit
        self.d = d

        # prepare detector error model
        self.dem = self.circuit.detector_error_model(decompose_errors=True)
        self.sampler = self.dem.compile_sampler()


    def sample(self) -> Tuple[List[bool], List[bool]]:
        det_data, obs_data, err_data = self.sampler.sample(shots=1, return_errors=True)
        return det_data[0], obs_data[0]


    def is_logical_error(self,
                         *,
                         actual_obs: List[bool],
                         estimated_obs: List[bool],
                         ) -> List[bool]:
        logical_operator_count_dict = {0: 0}
        for i, obs in enumerate(actual_obs):
            logical_operator_count_dict[i] += obs
        for i, obs in enumerate(estimated_obs):
            logical_operator_count_dict[i] += obs

        return [count % 2 == 1 for count in logical_operator_count_dict.values()]
