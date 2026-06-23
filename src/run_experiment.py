from multiprocessing import Pool, Manager
import sys

from circuit_experiment import experiment

D_LIST = [
    3,
    5,
    7,
    9,
    11,
    13,
    15,
]
P_LIST = [
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
]

NUM_EXPERIMENTS = 1000#000
MULTIPROCESSING = 10
CACHEDIV = None
#CACHEDIV = 4 # 0.01%,0.05%,0.1%,0.5%,1% (d<=15)
#CACHEDIV = 14 # 0.2%,0.3%,0.4% (d<=15)
#CACHEDIV = 20 # 0.5% (d>15) (only 10000 samples)
#CACHEDIV = 31 # for getting info of cluster nodes
#CACHEDIV = 41 # for getting cluster growth (40 is DEBUG=True, 41 is DEBUG=False)


if __name__ == '__main__':
    if MULTIPROCESSING is None:
        experiment(
                d_list=D_LIST,
                p_list=P_LIST,
                total_num_experiments=NUM_EXPERIMENTS,
                cachediv=CACHEDIV,
        )
    else:
        params = [
                {'d_list': [d], 'p_list': [p]} for d in D_LIST for p in P_LIST
        ]
        lock = Manager().Lock()
        def experiment_wrapper(dict_):
            experiment(
                    d_list=dict_['d_list'],
                    p_list=dict_['p_list'],
                    total_num_experiments=NUM_EXPERIMENTS,
                    multiprocessing_lock=lock,
                    cachediv=CACHEDIV,
            )
            
        with Pool(MULTIPROCESSING) as p:
            p.map(experiment_wrapper, params)
