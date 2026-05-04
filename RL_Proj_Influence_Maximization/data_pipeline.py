import networkx as nx
import os
import random
from collections import defaultdict, deque

# --- CONFIGURATION ---
GRAPH_FILE = 'actual_graph.txt'
COMM_FILE = 'top_5000_communities.txt'
OUTPUT_DIR = 'train_data'

NUM_TRAIN_GRAPHS = 100    
SUBGRAPH_SIZE = 200       # Bumped up to 200 to capture rich overlaps
TRAIN_RATIO = 0.80        

def load_data():
    print(f"Loading LiveJournal Graph from {GRAPH_FILE}...")
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        G = nx.read_edgelist(f, delimiter='\t', nodetype=int)
    
    print(f"Loading Communities from {COMM_FILE}...")
    node_to_comms = defaultdict(set)
    comm_to_nodes = defaultdict(set) # NEW: We need to look up nodes by community
    
    with open(COMM_FILE, 'r', encoding='utf-8') as f:
        for comm_id, line in enumerate(f):
            nodes = list(map(int, line.strip().split('\t')))
            for n in nodes:
                node_to_comms[n].add(comm_id)
                comm_to_nodes[comm_id].add(n)
                
    return G, node_to_comms, comm_to_nodes

def macro_split(G, train_ratio):
    print("\nExecuting Snowball Macro-Split...")
    target_train_nodes = int(G.number_of_nodes() * train_ratio)
    
    train_nodes = set()
    visited = set()
    
    start_node = random.choice(list(G.nodes()))
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue and len(train_nodes) < target_train_nodes:
        current = queue.popleft()
        train_nodes.add(current)
        
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    G_train = G.subgraph(train_nodes).copy()
    print(f"Macro-Split Complete: {G_train.number_of_nodes()} nodes reserved for Training.")
    return G_train

def extract_subgraphs(G_train, node_to_comms, comm_to_nodes):
    print(f"\nExtracting {NUM_TRAIN_GRAPHS} Community-Anchored subgraphs...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # NEW: Find "Anchor" communities that exist entirely inside our Training Split 
    # and have a good core size (between 5 and 50 members)
    train_nodes_set = set(G_train.nodes())
    valid_anchors = []
    
    for c_id, members in comm_to_nodes.items():
        # Check how many members of this community actually ended up in the training split
        members_in_train = members.intersection(train_nodes_set)
        if 5 <= len(members_in_train) <= 50:
            valid_anchors.append((c_id, members_in_train))

    print(f"Found {len(valid_anchors)} valid Anchor Communities in the training split.")

    valid_graphs_generated = 0
    
    while valid_graphs_generated < NUM_TRAIN_GRAPHS:
        # 1. Pick a random Anchor Community and scoop its core members
        anchor_cid, core_members = random.choice(valid_anchors)
        sub_nodes = set(core_members)
        queue = deque(core_members)
        
        # 2. Random Walk outward from the core to fill the remaining slots up to 200
        while queue and len(sub_nodes) < SUBGRAPH_SIZE:
            curr = queue.popleft()
            neighbors = list(G_train.neighbors(curr))
            random.shuffle(neighbors) 
            
            for n in neighbors:
                if n not in sub_nodes:
                    sub_nodes.add(n)
                    queue.append(n)
                    if len(sub_nodes) >= SUBGRAPH_SIZE:
                        break

        G_sub = G_train.subgraph(sub_nodes).copy()
        
        # 3. Rename nodes (0 to N-1) for PyTorch Geometric
        mapping = {old_id: new_id for new_id, old_id in enumerate(G_sub.nodes())}
        G_sub_renamed = nx.relabel_nodes(G_sub, mapping)
        
        # 4. Save the Graph Structure
        graph_path = os.path.join(OUTPUT_DIR, f'train_graph_{valid_graphs_generated}.txt')
        nx.write_edgelist(G_sub_renamed, graph_path, delimiter='\t', data=False)
        
        # 5. Extract local communities
        local_comms = defaultdict(list)
        for old_id, new_id in mapping.items():
            if old_id in node_to_comms:
                for c_id in node_to_comms[old_id]:
                    local_comms[c_id].append(new_id)
        
        # 6. Save the Community Structure
        comm_path = os.path.join(OUTPUT_DIR, f'train_comm_{valid_graphs_generated}.txt')
        with open(comm_path, 'w', encoding='utf-8') as f:
            for c_id, members in local_comms.items():
                # We still enforce the rule of 2, but our Anchor guarantees we will pass!
                if len(members) >= 2:
                    f.write('\t'.join(map(str, members)) + '\n')
                    
        valid_graphs_generated += 1
        if valid_graphs_generated % 20 == 0:
            print(f"Generated {valid_graphs_generated} / {NUM_TRAIN_GRAPHS} subgraphs...")

if __name__ == '__main__':
    G, node_to_comms, comm_to_nodes = load_data()
    G_train = macro_split(G, TRAIN_RATIO)
    extract_subgraphs(G_train, node_to_comms, comm_to_nodes)
    print(f"\nSuccess! Data saved to ./{OUTPUT_DIR}/")