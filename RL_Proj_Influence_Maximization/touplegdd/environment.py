import numpy as np
import statistics
from multiprocessing import Pool
import random
import time
import utils.graph_utils as graph_utils

random.seed(123)
np.random.seed(123)


class Environment:
    ''' environment that the agents run in '''
    def __init__(self, name, graphs, budget, method='RR', use_cache=False, training=True, alpha=1.0, beta=10.0):
        '''
            method: 'RR' or 'MC'
            use_cache: use cache to speed up
            alpha: weight for raw influence gain
            beta: weight for newly reached communities
        '''
        # sampled set of graphs
        self.name = name
        self.graphs = graphs
        # IM
        self.budget = budget
        self.method = method
        
        # --- NEW: Composite Reward Hyperparameters ---
        self.alpha = alpha
        self.beta = beta
        
        # useful only if run on the same graph multiple times
        self.use_cache = use_cache
        if self.use_cache:
            if self.method == 'MC':
                self.influences = {} 
            elif self.method == 'RR':
                self.RRs_dict = {}
        self.training = training 

    def reset_graphs(self, num_graphs=10):
        raise NotImplementedError()

    def reset(self, idx=None, training=True):
        ''' restart '''
        if idx is None:
            self.graph = random.choice(self.graphs)
        else:
            self.graph = self.graphs[idx]
        self.state = [0 for _ in range(self.graph.num_nodes)]
        
        # IM
        self.prev_inf = 0 
        
        # --- NEW: Track Communities Reached ---
        self.prev_comm = 0 
        
        if self.use_cache and self.method == 'RR':
            self.RRs = self.RRs_dict.setdefault(id(self.graph), [])
        self.states = []
        self.actions = []
        self.rewards = []
        self.training = training

    def compute_reward(self, S):
        num_process = 5 
        num_trial = 10000 
        
        need_compute = True
        if self.use_cache and self.method == 'MC':
            S_str = f"{id(self.graph)}.{','.join(map(str, sorted(S)))}"
            need_compute = S_str not in self.influences

        if need_compute:
            if self.method == 'MC':
                with Pool(num_process) as p:
                    results = p.map(graph_utils.workerMC, 
                        [[self.graph, S, int(num_trial / num_process)] for _ in range(num_process)])
                    es_inf = statistics.mean([r[0] for r in results])
                    es_comms = statistics.mean([r[1] for r in results])
            elif self.method == 'RR':
                if self.use_cache:
                    es_inf, es_comms = graph_utils.computeRR(self.graph, S, num_trial, cache=self.RRs)
                else:
                    es_inf, es_comms = graph_utils.computeRR(self.graph, S, num_trial)
            else:
                raise NotImplementedError(f'{self.method}')

            if self.use_cache and self.method == 'MC':
                self.influences[S_str] = (es_inf, es_comms)
        else:
            es_inf, es_comms = self.influences[S_str]

        # 1. Calculate Standard Influence Gain
        inf_gain = es_inf - self.prev_inf
        self.prev_inf = es_inf

        # 2. Calculate Communities Reached Gain (Based on Spread!)
        comm_gain = es_comms - self.prev_comm
        self.prev_comm = es_comms

        # 3. Composite Reward: Reward raw spread and diversity of spread
        reward = (self.alpha * inf_gain) + (self.beta * comm_gain)
        
        # store reward
        self.rewards.append(reward)

        return reward

    def step(self, node, time_reward=None):
        ''' change state and get reward '''
        if self.state[node] == 1:
            return
            
        self.states.append(self.state.copy())
        self.actions.append(node)
        self.state[node] = 1
        
        if self.name != 'IM':
            raise NotImplementedError(f'Environment {self.name}')

        S = self.actions
        done = len(S) >= self.budget

        if self.training:
            reward = self.compute_reward(S)
        else:
            if done:
                if time_reward is not None:
                    start_time = time.time()
                reward = self.compute_reward(S)
                if time_reward is not None:
                    time_reward[0] = time.time() - start_time
            else:
                reward = None

        return (reward, done)