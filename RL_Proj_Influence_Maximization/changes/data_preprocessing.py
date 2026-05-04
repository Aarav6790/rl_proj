import os

def extract_subgraph(edges_file, communities_file, out_edges_file, out_comms_file, num_communities_to_pick=5):
    print(f"Reading top {num_communities_to_pick} communities...")
    
    selected_nodes = set()
    raw_communities = []
    
    # 1. Read the ground-truth communities
    with open(communities_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_communities_to_pick:
                break
            # Each line is a space-separated list of node IDs in that community
            nodes_in_comm = [int(n) for n in line.strip().split()]
            raw_communities.append(nodes_in_comm)
            selected_nodes.update(nodes_in_comm)
            
    print(f"Selected {len(selected_nodes)} unique nodes across {num_communities_to_pick} communities.")

    # 2. Re-index nodes to be continuous (0 to N-1)
    # This prevents PyTorch from running out of memory with huge node IDs
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(selected_nodes))}
    
    # 3. Write the re-indexed communities to the new file
    print(f"Writing sub-communities to {out_comms_file}...")
    with open(out_comms_file, 'w') as f:
        for comm in raw_communities:
            # Map old IDs to new IDs and write them out
            new_comm_nodes = [str(node_mapping[n]) for n in comm]
            f.write(" ".join(new_comm_nodes) + "\n")

    # 4. Extract edges that belong ONLY to our selected nodes
    print(f"Scanning massive edge list. This might take a minute...")
    extracted_edges = []
    
    with open(edges_file, 'r') as f:
        for line in f:
            # Skip comments in the SNAP file
            if line.startswith('#'):
                continue
                
            src, dst = map(int, line.strip().split())
            
            # If BOTH nodes are in our selected subset, keep the edge
            if src in selected_nodes and dst in selected_nodes:
                extracted_edges.append((node_mapping[src], node_mapping[dst]))

    print(f"Found {len(extracted_edges)} edges connecting our subset.")

    # 5. Write the extracted edges to the new ToupleGDD format file
    print(f"Writing sub-edges to {out_edges_file}...")
    with open(out_edges_file, 'w') as f:
        for src, dst in extracted_edges:
            f.write(f"{src} {dst}\n")
            
    print("Extraction Complete! Your data is ready for ToupleGDD.")

if __name__ == "__main__":
    # Define your paths here
    RAW_EDGES = "com-lj.ungraph.txt"       # The massive SNAP edge file
    RAW_COMMS = "com-lj.top5000.cmty.txt"  # The massive SNAP community file
    
    # Where you want the ToupleGDD-ready files to go
    OUT_EDGES = "train_data/lj_subgraph_edges.txt"
    OUT_COMMS = "train_data/lj_subgraph_comms.txt"
    
    # Make sure output directory exists
    os.makedirs("train_data", exist_ok=True)
    
    extract_subgraph(RAW_EDGES, RAW_COMMS, OUT_EDGES, OUT_COMMS, num_communities_to_pick=5)