import networkx as nx
import os
import random
from collections import defaultdict, deque

# --- CONFIGURATION ---
GRAPH_FILE = 'actual_graph.txt'
COMM_FILE = 'top_5000_communities.txt'

# Output directories/files
OUTPUT_GRAPH_DIR = 'train_graphs'
OUTPUT_COMM_DIR = 'train_comms'
TEST_GRAPH_FILE = 'test_graph.txt'
TEST_COMM_FILE = 'test_comms.txt'

NUM_TRAIN_GRAPHS = 70
SUBGRAPH_SIZE = 100
TRAIN_RATIO = 0.80

# NEW: Maximum allowed overlap between two communities in a subgraph (0.0 to 1.0)
# 0.6 means if two communities are more than 60% identical, one is discarded.
MAX_JACCARD_OVERLAP = 0.6  

def load_data():
    print(f"Loading LiveJournal Graph from {GRAPH_FILE}...")
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        G = nx.read_edgelist(f, delimiter='\t', nodetype=int)
    
    print(f"Loading Communities from {COMM_FILE}...")
    node_to_comms = defaultdict(set)
    comm_to_nodes = defaultdict(set)
    
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
    
    all_nodes = set(G.nodes())
    test_nodes = all_nodes - train_nodes
    G_test = G.subgraph(test_nodes).copy()
    
    print(f"Macro-Split Complete:")
    print(f"  - {G_train.number_of_nodes()} nodes reserved for Training.")
    print(f"  - {G_test.number_of_nodes()} nodes reserved for Testing.")
    
    return G_train, G_test

def save_test_data(G_test, node_to_comms):
    print("\nProcessing and Saving Test Data (Single Giant Graph)...")
    mapping = {old_id: new_id for new_id, old_id in enumerate(G_test.nodes())}
    G_test_renamed = nx.relabel_nodes(G_test, mapping)
    nx.write_edgelist(G_test_renamed, TEST_GRAPH_FILE, delimiter='\t', data=False)
    
    test_comms = defaultdict(list)
    for old_id, new_id in mapping.items():
        for c_id in node_to_comms.get(old_id, []):
            test_comms[c_id].append(new_id)
            
    with open(TEST_COMM_FILE, 'w', encoding='utf-8') as f:
        for members in test_comms.values():
            if len(members) >= 2:
                f.write('\t'.join(map(str, members)) + '\n')
                
    print(f"Saved test graph to {TEST_GRAPH_FILE} and test communities to {TEST_COMM_FILE}")

# --- NEW: Function to eliminate identical/nested communities ---
def filter_distinct_communities(local_comms_dict, max_jaccard):
    # Keep only communities with at least 2 members
    candidates = {cid: set(members) for cid, members in local_comms_dict.items() if len(members) >= 2}
    
    # Sort by size descending (prioritize keeping larger communities)
    sorted_candidates = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)
    
    distinct_comms = {}
    
    for cid, members in sorted_candidates:
        is_distinct = True
        for d_cid, d_members in distinct_comms.items():
            intersection = len(members.intersection(d_members))
            union = len(members.union(d_members))
            jaccard = intersection / union if union > 0 else 0
            
            # If overlap is too high, reject this community as a duplicate
            if jaccard > max_jaccard:
                is_distinct = False
                break
                
        if is_distinct:
            distinct_comms[cid] = list(members)
            
    return distinct_comms

def extract_subgraphs(G_train, node_to_comms, comm_to_nodes):
    print(f"\nExtracting {NUM_TRAIN_GRAPHS} Subgraphs with STRICT DISTINCT Communities...")
    
    os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_COMM_DIR, exist_ok=True)

    train_nodes_set = set(G_train.nodes())
    valid_anchors = []
    
    for c_id, members in comm_to_nodes.items():
        members_in_train = members.intersection(train_nodes_set)
        if 5 <= len(members_in_train) <= 50:
            valid_anchors.append((c_id, members_in_train))

    valid_graphs_generated = 0
    attempts = 0
    
    while valid_graphs_generated < NUM_TRAIN_GRAPHS:
        attempts += 1
        anchor_cid, core_members = random.choice(valid_anchors)
        
        # Simple BFS to gather 100 nodes quickly
        start_node = random.choice(list(core_members))
        sub_nodes = {start_node}
        queue = deque([start_node])
        
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

        if len(sub_nodes) < SUBGRAPH_SIZE:
            continue

        G_sub = G_train.subgraph(sub_nodes).copy()
        
        # Relabel nodes 0 to N-1
        mapping = {old_id: new_id for new_id, old_id in enumerate(G_sub.nodes())}
        G_sub_renamed = nx.relabel_nodes(G_sub, mapping)
        
        # Gather all raw communities in this subgraph
        raw_local_comms = defaultdict(list)
        for old_id, new_id in mapping.items():
            for c_id in node_to_comms.get(old_id, []):
                raw_local_comms[c_id].append(new_id)
        
        # Apply the Jaccard Filter to remove duplicates
        distinct_local_comms = filter_distinct_communities(raw_local_comms, MAX_JACCARD_OVERLAP)
        
        # Strict check: Do we actually have 3 structurally distinct communities?
        if len(distinct_local_comms) < 3:
            continue
        
        # Success! Save files.
        graph_path = os.path.join(OUTPUT_GRAPH_DIR, f'train_graph_{valid_graphs_generated}.txt')
        nx.write_edgelist(G_sub_renamed, graph_path, delimiter='\t', data=False)
        
        comm_path = os.path.join(OUTPUT_COMM_DIR, f'train_comm_{valid_graphs_generated}.txt')
        with open(comm_path, 'w', encoding='utf-8') as f:
            for members in distinct_local_comms.values():
                f.write('\t'.join(map(str, members)) + '\n')
                    
        valid_graphs_generated += 1
        if valid_graphs_generated % 10 == 0:
            print(f"Generated {valid_graphs_generated} / {NUM_TRAIN_GRAPHS} training subgraphs (Total attempts: {attempts})")

if __name__ == '__main__':
    G, node_to_comms, comm_to_nodes = load_data()
    G_train, G_test = macro_split(G, TRAIN_RATIO)
    save_test_data(G_test, node_to_comms)
    extract_subgraphs(G_train, node_to_comms, comm_to_nodes)
    print("\nAll tasks completed successfully!")