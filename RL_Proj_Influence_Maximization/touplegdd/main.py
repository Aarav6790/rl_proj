import argparse
import sys
import os
import time
import datetime
import numpy as np
import torch
import utils.graph_utils as graph_utils
import rl_agents
import environment
import runner
import logging

torch.manual_seed(123)
np.random.seed(123)

# Set up logger
logging.basicConfig(
   format='%(asctime)s:%(levelname)s:%(message)s',
   level=logging.INFO
)

parser = argparse.ArgumentParser(description='INF-GNN-RL')
parser.add_argument('--budget', type=int, default=6, help='budget to select the source node set')
parser.add_argument('--graph', type=str, metavar='GRAPH_PATH', default='soc-dolphins.txt', help='path to the graph file')
parser.add_argument('--agent', type=str, metavar='AGENT_CLASS', default='Agent', help='class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--model', type=str, default='Tripling', help='model name')
parser.add_argument('--model_file', type=str, default='tripling.ckpt', help='model file name')
parser.add_argument('--epoch', type=int, metavar='nepoch', default=2000, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--bs', type=int, default=8, help='minibatch size for training')
parser.add_argument('--n_step', type=int, default=2, help='n step transitions in RL')
parser.add_argument('--cpu', action='store_true', default=False, help='use CPU')
parser.add_argument('--test', action='store_true', default=False, help='test performance of model')
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='IM', help='Class to use for the environment. Must be in the \'environment\' module')

# --- NEW: Community & Reward Arguments ---
parser.add_argument('--num_communities', type=int, default=5, help='Number of ground-truth communities in dataset')
parser.add_argument('--max_per_comm', type=int, default=3, help='Max seed nodes allowed per community')
parser.add_argument('--community_path', type=str, default=None, help='Path to community ground truth file')
parser.add_argument('--alpha', type=float, default=1.0, help='Reward weight for raw influence gain')
parser.add_argument('--beta', type=float, default=10.0, help='Reward weight for reaching a new community')

def main():
    ##### Load Arguments #####
    args = parser.parse_args()
    logging.info('Loading graph %s' % args.graph)

    ##### Set Device #####
    device = torch.device('cuda' if not(args.cpu) and torch.cuda.is_available() else 'cpu')
    args.device = device

    ##### Load Graph #####
    path_graphs = []
    path_comms = []
    if os.path.isdir(args.graph):
        graph_files = sorted([f for f in os.listdir(args.graph) if not f.startswith('.')])
        path_graphs = [os.path.join(args.graph, f) for f in graph_files]
        
        if args.community_path and os.path.isdir(args.community_path):
            comm_files = sorted([f for f in os.listdir(args.community_path) if not f.startswith('.')])
            if len(comm_files) != len(graph_files):
                logging.warning("Number of graph files and community files do not match!")
            path_comms = [os.path.join(args.community_path, f) for f in comm_files]
        else:
            path_comms = [args.community_path] * len(path_graphs)
    else: # read one graph
        path_graphs = [args.graph]
        path_comms = [args.community_path]
    
    # --- NEW: Pass corresponding community_path to the graph reader ---
    graph_lst = []
    for pg, pc in zip(path_graphs, path_comms):
        graph_lst.append(graph_utils.read_graph(pg, ind=0, directed=True, community_path=pc))
        
    # --- NEW: Autodetect global max communities and pad ---
    max_comms = max([g.num_communities for g in graph_lst]) if graph_lst else 0
    args.num_communities = max_comms
    for g in graph_lst:
        g.num_communities = max_comms

    for i in range(len(path_graphs)):
        graph_lst[i].path_graph = path_graphs[i]

    args.graphs = graph_lst

    args.double_dqn = True

    if not args.test: # for training of tripling
        time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(time_stamp):
            os.makedirs(time_stamp)

        args.model_file = os.path.join(time_stamp, args.model_file)

    args.T = 3
    args.memory_size = 50000
    args.reg_hidden = 32
    if args.model == 'Tripling':
        args.embed_dim = 50
    else:
        args.embed_dim = 64

    ##### Load Agent #####
    logging.info(f'Loading agent {args.model}')
    agent = rl_agents.Agent(args)
    
    ##### Load Environment #####
    # create environment
    logging.info('Loading environment %s' % args.environment_name)
    
    # --- NEW: Pass alpha and beta to the environments ---
    train_env = environment.Environment(args.environment_name, graph_lst, args.budget, method='RR', use_cache=True, alpha=args.alpha, beta=args.beta)
    test_env = environment.Environment(args.environment_name, graph_lst, args.budget, method='MC', use_cache=True, alpha=args.alpha, beta=args.beta)
    
    ##### Load Runner and Start Running #####
    print("Running a single instance simulation")
    my_runner = runner.Runner(train_env, test_env, agent, not(args.test))
    if not(args.test):
        my_runner.train(args.epoch, args.model_file, 'list_cumul_reward.txt')
    else:
        my_runner.test(num_trials=10)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Total time usage: {end_time - start_time:.2f} seconds')