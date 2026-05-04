import random
import time
import os
from collections import namedtuple, deque
import numpy as np
import models
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)


class DQAgent:
    ''' deep Q agent '''
    def __init__(self, args):
        '''
        lr: learning rate
        n_step: (s_t-n,a_t-n,r,s_t)
        '''
        self.model_name = args.model
        self.gamma = 0.99 # discount factor of future rewards
        self.n_step = args.n_step # num of steps to accumulate rewards

        self.training = not(args.test)
        self.T = args.T

        self.memory = ReplayMemory(args.memory_size)
        self.batch_size = args.bs # batch size for experience replay

        self.double_dqn = args.double_dqn
        self.device = args.device

        self.node_dim = 2
        self.edge_dim = 4
        self.reg_hidden = args.reg_hidden
        self.embed_dim = args.embed_dim
        
        # --- NEW: Community Parameters ---
        # Fallback to 0 / None if not explicitly passed in args to prevent crashes
        self.num_communities = getattr(args, 'num_communities', 0)
        self.max_per_comm = getattr(args, 'max_per_comm', None) 

        # store node embeddings of each graph, avoid multiprocess copy
        self.graph_node_embed = {}
        
        # model and graph input
        if self.model_name == 'S2V_DUEL':
            self.model = models.S2V_DUEL(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, edge_dim=4,
                T=self.T, w_scale=0.01, avg=False).to(self.device)
            # double dqn
            if self.training and self.double_dqn:
                self.target = models.S2V_DUEL(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, 
                    edge_dim=4, T=self.T, w_scale=0.01, avg=False).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            self.setup_graph_input = self.setup_graph_input_s2v

        elif self.model_name == 'S2V_DQN':
            # --- NEW: Passed num_communities to Model ---
            self.model = models.S2V_DQN(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, edge_dim=4,
                T=self.T, w_scale=0.01, avg=False, num_communities=self.num_communities).to(self.device)
            # double dqn
            if self.training and self.double_dqn:
                self.target = models.S2V_DQN(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, 
                    edge_dim=4, T=self.T, w_scale=0.01, avg=False, num_communities=self.num_communities).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            self.setup_graph_input = self.setup_graph_input_s2v

        elif self.model_name == 'Tripling':
            # --- NEW: Passed num_communities to Model ---
            self.model = models.Tripling(embed_dim=self.embed_dim, sgate_l1_dim=128, tgate_l1_dim=128, T=3, 
                hidden_dims=[50, 50, 50], w_scale=0.01, num_communities=self.num_communities).to(self.device)
            # double dqn
            if self.training and self.double_dqn:
                self.target = models.Tripling(embed_dim=self.embed_dim, sgate_l1_dim=128, tgate_l1_dim=128, T=3,
                    hidden_dims=[50, 50, 50], w_scale=0.01, num_communities=self.num_communities).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            self.setup_graph_input = self.setup_graph_input_tripling

        else:
            raise NotImplementedError(f'RL Model {self.model_name}')

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        if not self.training:
            cwd = os.getcwd()
            self.model.load_state_dict(torch.load(os.path.join(cwd, args.model_file)), strict=False)
            self.model.eval()

    def reset(self):
        ''' restart '''
        pass

    @torch.no_grad()
    def setup_graph_input_s2v(self, graphs, states, actions=None):
        sample_size = len(graphs)
        data = []
        for i in range(sample_size):
            x = torch.ones(graphs[i].num_nodes, self.node_dim)
            x[:, 1] = 1 - states[i] # selected node feature set 0
            edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous()
            edge_attr = torch.ones(graphs[i].num_edges, self.edge_dim)
            edge_attr[:, 1] = torch.tensor([p[-1] for p in graphs[i].from_to_edges_weight()], dtype=torch.float)
            edge_attr[:, 0] = states[i][edge_index[0]]
            edge_attr[:, 2] = torch.abs(states[i][edge_index[0]] - states[i][edge_index[1]])

            y = actions[i].clone() if actions is not None else None
            
            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            # --- NEW: Attach Community Features to PyG Data Object ---
            if self.num_communities > 0:
                d.comm = torch.tensor(graphs[i].get_community_features(), dtype=torch.float32)
            data.append(d)

        loader = DataLoader(data, batch_size=sample_size, shuffle=False)
        for batch in loader:
            if actions is not None:
                total_num = 0
                for i in range(1, sample_size):
                    total_num += batch[i - 1].num_nodes
                    batch[i].y += total_num
            return batch.to(self.device)


    def setup_graph_input_tripling(self, graphs, states, actions=None):
        sample_size = len(graphs)
        data = []
        for i in range(sample_size):
            if id(graphs[i]) not in self.graph_node_embed:
                self.graph_node_embed[id(graphs[i])] = models.get_init_node_embed(graphs[i], 30, self.device) 
            with torch.no_grad():
                x = self.graph_node_embed[id(graphs[i])].detach().clone()
                x = torch.cat((x, states[i].detach().clone().unsqueeze(dim=1)), dim=-1)
                edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous()
                edge_weight = torch.tensor([p[-1] for p in graphs[i].from_to_edges_weight()], dtype=torch.float)

                y = actions[i].detach().clone() if actions is not None else None
                
                d = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
                # --- NEW: Attach Community Features to PyG Data Object ---
                if self.num_communities > 0:
                    d.comm = torch.tensor(graphs[i].get_community_features(), dtype=torch.float32)
                data.append(d)

        with torch.no_grad():
            loader = DataLoader(data, pin_memory=False, num_workers=4, batch_size=sample_size, shuffle=False)
            for batch in loader:
                if actions is not None:
                    total_num = 0
                    for i in range(1, sample_size):
                        total_num += batch[i - 1].num_nodes
                        batch[i].y += total_num
                return batch.to(self.device)


    @torch.no_grad()
    def setup_graph_pred(self, graphs, states, actions):
        sample_size = len(states)
        data = []
        for i in range(sample_size):
            x = torch.ones(graphs[i].num_nodes, self.node_dim)
            x[:, 1] = 1 - states[i] 
            edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous()
            edge_attr = torch.ones(graphs[i].num_edges, self.edge_dim)
            edge_attr[:, 1] = torch.tensor([p[-1] for p in graphs[i].from_to_edges_weight()], dtype=torch.float)
            edge_attr[:, 0] = states[i][edge_index[0]]
            edge_attr[:, 2] = torch.abs(states[i][edge_index[0]] - states[i][edge_index[1]])

            y = actions[i].clone()
            
            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            if self.num_communities > 0:
                d.comm = torch.tensor(graphs[i].get_community_features(), dtype=torch.float32)
            data.append(d)

        loader = DataLoader(data, batch_size=sample_size, shuffle=False)
        for batch in loader:
            total_num = torch.tensor([0], dtype=torch.long)
            for i in range(1, sample_size):
                total_num.add_(batch[i - 1].num_nodes)
                batch[i].y = torch.add(batch[i].y, total_num)
            return batch.to(self.device)


    @torch.no_grad()
    def setup_graph_pred_all(self, graph, state):
        x = torch.ones(graph.num_nodes, self.node_dim)
        x[:, 1] = 1 - state 
        edge_index = torch.tensor(graph.from_to_edges(), dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(graph.num_edges, self.edge_dim)
        edge_attr[:, 1] = torch.tensor([p[-1] for p in graph.from_to_edges_weight()], dtype=torch.float)
        edge_attr[:, 0] = state[edge_index[0]]
        edge_attr[:, 2] = torch.abs(state[edge_index[0]] - state[edge_index[1]])
        
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if self.num_communities > 0:
            d.comm = torch.tensor(graph.get_community_features(), dtype=torch.float32)
            
        data = [d]
        loader = DataLoader(data, batch_size=1, shuffle=False)
        for batch in loader:
            return batch.to(self.device)


    def select_action(self, graph, state, epsilon, training=True, budget=None):
        ''' act upon state '''
        # --- NEW: Action Masking / Quota Enforcement Logic ---
        blocked_nodes = []
        if self.max_per_comm is not None and hasattr(graph, 'communities') and graph.communities:
            selected_nodes = state.nonzero().flatten().tolist()
            comm_counts = {}
            # Count current seeds in each community
            for node in selected_nodes:
                for comm in graph.communities.get(node, set()):
                    comm_counts[comm] = comm_counts.get(comm, 0) + 1
            
            # Identify communities that have reached the quota
            blocked_comms = {comm for comm, count in comm_counts.items() if count >= self.max_per_comm}
            
            # Identify all unselected nodes that belong to a blocked community
            if blocked_comms:
                for node in range(graph.num_nodes):
                    if state[node] == 0: 
                        node_comms = graph.communities.get(node, set())
                        if len(node_comms.intersection(blocked_comms)) > 0:
                            blocked_nodes.append(node)

        # TESTING MODE
        if not(training):
            graph_input = self.setup_graph_input([graph], state.unsqueeze(dim=0))
            with torch.no_grad():
                q_a = self.model(graph_input)
                
            q_a[state.nonzero()] = -1e5
            
            # Apply dynamic quota mask
            if blocked_nodes:
                q_a[blocked_nodes] = -1e5

            if budget is None:
                return torch.argmax(q_a).detach().clone()
            else: # return all seed nodes within budget at one time
                return torch.topk(q_a.squeeze(dim=1), budget)[1].detach().clone()
                
        # TRAINING MODE
        available = (state == 0).nonzero().flatten().tolist()
        
        # Remove blocked nodes from available choices during random exploration
        if blocked_nodes:
            available_filtered = [n for n in available if n not in blocked_nodes]
            # Fallback: if masking blocks ALL remaining nodes, ignore the mask to prevent crash
            if len(available_filtered) > 0:
                available = available_filtered
                
        if epsilon > random.random():
            return torch.tensor([random.choice(available)], dtype=torch.long, device=self.device)
        else:
            graph_input = self.setup_graph_input([graph], state.unsqueeze(dim=0))
            with torch.no_grad():
                q_a = self.model(graph_input)
            
            # Apply standard mask + dynamic quota mask
            q_a[state.nonzero()] = -1e5
            if blocked_nodes:
                q_a[blocked_nodes] = -1e5
                
            available_tensor = torch.tensor(available, dtype=torch.long, device=self.device)
            max_position = (q_a == q_a[available_tensor].max().item()).nonzero()
            
            # Ensure we only pick from valid available nodes
            intersect = np.intersect1d(available_tensor.cpu().numpy(), max_position.cpu().contiguous().view(-1).numpy())
            if len(intersect) == 0: 
                intersect = available_tensor.cpu().numpy()
                
            return torch.tensor([random.choice(intersect)], dtype=torch.long, device=self.device)


    def memorize(self, env):
        '''n step for stability'''
        sum_rewards = [0.0]
        for reward in reversed(env.rewards):
            reward /= env.graph.num_nodes
            sum_rewards.append(reward + self.gamma * sum_rewards[-1])
        sum_rewards = sum_rewards[::-1]

        for i in range(len(env.states)):
            if i + self.n_step < len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long), 
                    torch.tensor([env.actions[i]], dtype=torch.long), 
                    torch.tensor(env.states[i + self.n_step], dtype=torch.long),
                    torch.tensor([sum_rewards[i] - (self.gamma ** self.n_step) * sum_rewards[i + self.n_step]], dtype=torch.float),
                    env.graph)
            elif i + self.n_step == len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long), 
                    torch.tensor([env.actions[i]], dtype=torch.long), 
                    None,
                    torch.tensor([sum_rewards[i]], dtype=torch.float),  
                    env.graph)

    def fit(self):
        '''fit on a batch sampled from replay memory'''
        sample_size = self.batch_size if len(self.memory) >= self.batch_size else len(self.memory)
        transitions = self.memory.sample(sample_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
            dtype=torch.bool, device=self.device)

        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states_graphs = [batch.graph[i] for i, s in enumerate(batch.next_state) if s is not None]

        state_batch = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        graph_batch = batch.graph

        state_action_values = self.model(self.setup_graph_input(graph_batch, state_batch, action_batch)).squeeze(dim=1)
        next_state_values = torch.zeros(sample_size, device=self.device)

        if len(non_final_next_states) > 0:
            if self.double_dqn:
                batch_non_final = self.setup_graph_input(non_final_next_states_graphs, non_final_next_states)
                next_state_values[non_final_mask] = scatter_max(
                    self.target(batch_non_final).squeeze(dim=1).add_(torch.cat(non_final_next_states).to(self.device) * (-1e5)), 
                    batch_non_final.batch)[0].clamp_(min=0).detach()
            else:
                batch_non_final = self.setup_graph_input(non_final_next_states_graphs, non_final_next_states)
                next_state_values[non_final_mask] = scatter_max(
                    self.model(batch_non_final).squeeze(dim=1).add_(torch.cat(non_final_next_states).to(self.device) * (-1e5)), 
                    batch_non_final.batch)[0].clamp_(min=0).detach()

        expected_state_action_values = next_state_values * self.gamma ** self.n_step + reward_batch.to(self.device)

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        if self.double_dqn:
            self.target.load_state_dict(self.model.state_dict())
            return True
        return False

    def save_model(self, file_name):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), os.path.join(cwd, file_name))


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'graph'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

Agent = DQAgent