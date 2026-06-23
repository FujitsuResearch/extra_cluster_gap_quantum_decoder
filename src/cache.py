from typing import Union, Tuple, List, Optional, Dict
from dataclasses import dataclass
from glob import glob
import os
import json
import math

import numpy as np


class Cache:
    def __init__(self, filepath: str, load: bool = True):
        self.filepath = filepath
        if os.path.exists(filepath) and load:
            with open(filepath, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def load_all_cachedivs(self, targets: list[str] | None = None, excludes: list[str] | None = None):
        filepath_wo_ext = os.path.splitext(self.filepath)[0]
        cachedivs_dir = f'{filepath_wo_ext}-cachedivs'
        if targets is None:
            json_files = glob(f'{cachedivs_dir}/*.json')
            if excludes is not None:
                excludes = [f'{cachedivs_dir}/{exclude}.json' for exclude in excludes]
                json_files = [json_file for json_file in json_files if json_file not in excludes]
        else:
            json_files = [f'{cachedivs_dir}/{target}.json' for target in targets]

        for json_file in json_files:
            print(f'loading {json_file}...')
            try:
                with open(json_file, 'r') as f:
                    additional_cache = json.load(f)
            except Exception as e:
                print(e)
                continue

            for d, d_additional_cache in additional_cache.items():
                for p, pd_additional_cache in d_additional_cache.items():
                    for key, val in pd_additional_cache.items():
                        if d not in self.cache:
                            self.cache[d] = {p: {key: val}}
                        elif p not in self.cache[d]:
                            self.cache[d][p] = {key: val}
                        elif key not in self.cache[d][p]:
                            self.cache[d][p][key] = val
                        else:
                            self.cache[d][p][key] += val

    def add_value(self, d: int, p: float, key: str, value: Union[int, float]):
        self._control(d, p, key, value, add=True)

    def append_value(self, d: int, p: float, key: str, value: Union[int, float]):
        self._control(d, p, key, value, add=False)

    def _control(self, d: int, p: float, key: str, value: Union[int, float], add: bool):
        d = str(d)
        p = str(p)
        if d not in self.cache:
            self.cache[d] = {}
        if p not in self.cache[d]:
            self.cache[d][p] = {}
        if key not in self.cache[d][p]:
            if add:
                self.cache[d][p][key] = 0
            else:
                self.cache[d][p][key] = []

        if add:
            self.cache[d][p][key] += value
        else:
            self.cache[d][p][key].append(value)

    def exists(self, d: int, p: float) -> bool:
        d = str(d)
        p = str(p)
        if d not in self.cache:
            return False
        if p not in self.cache[d]:
            return False
        return True

    def reload(self, remain_d: int, remain_p: float):
        if not os.path.exists(self.filepath):
            return

        remain_d = str(remain_d)
        remain_p = str(remain_p)

        with open(self.filepath, 'r') as f:
            loaded_cache = json.load(f)
        for d in loaded_cache:
            if d != remain_d:
                self.cache[d] = loaded_cache[d]
                continue
            for p in loaded_cache[d]:
                if p != remain_p:
                    self.cache[d][p] = loaded_cache[d][p]

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.cache, f)

    def get_value(self, d: int, p: float, key: str, default: Optional[Union[int, float]] = None) -> Union[int, float]:
        d = str(d)
        p = str(p)
        if default is not None:
            if d not in self.cache:
                return default
            if p not in self.cache[d]:
                return default
            return self.cache[d][p].get(key, default)
        return self.cache[d][p][key]


    def get_extra(self, extra_param: str, ignore_none: bool = False, ignore_0: bool = False, factor: float = 1) -> Dict[float, Dict[str, List[Union[int, float]]]]:
        result = {}
        for d in sorted(int(d) for d in self.cache):
            d = str(d)
            cache_d = self.cache[d]
            for p in sorted(float(p) for p in cache_d):
                p = str(p)
                cache_d_p = cache_d[p]
                #import pdb; pdb.set_trace()
                if p not in result:
                    result[p] = {'d': [], 'rotated_surface_code_qubits': [], f'{extra_param}(mean)': [], f'{extra_param}(std)': [], f'{extra_param}(max)': [], f'{extra_param}(0ratio)': [], f'{extra_param}(0ratio-se)': []}
                result[p]['d'].append(int(d))
                result[p]['rotated_surface_code_qubits'].append(int(d)**2)
                iteration_min = np.array(cache_d_p[f'{extra_param}'])

                if iteration_min.ndim == 2:
                    iteration_min = iteration_min.max(axis=1)
                #iteration_min = iteration_min[iteration_min>0] # it is not required, maybe
                if not ignore_none:
                    iteration_min[iteration_min==None] = 0
                else:
                    iteration_min = iteration_min[iteration_min!=None]

                iteration_min *= factor

                result[p][f'{extra_param}(0ratio)'].append(len(iteration_min[iteration_min!=0])/len(iteration_min))
                result[p][f'{extra_param}(0ratio-se)'].append(math.sqrt(len(iteration_min[iteration_min!=0]))/len(iteration_min))
                if ignore_0:
                    iteration_min = iteration_min[iteration_min!=0]
                result[p][f'{extra_param}(mean)'].append(iteration_min.mean() if len(iteration_min) > 0 else 0)
                result[p][f'{extra_param}(std)'].append(iteration_min.std() if len(iteration_min) > 0 else 0)
                result[p][f'{extra_param}(max)'].append(iteration_min.max() if len(iteration_min) > 0 else 0)

        return result
