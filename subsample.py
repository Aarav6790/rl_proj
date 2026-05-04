import networkx as nx
import os
import random
from collections import defaultdict, deque

# =============================================================================
# CONFIGURATION
# =============================================================================
GRAPH_FILE = 'actual_graph.txt'
COMM_FILE  = 'top_5000_communities.txt'

OUTPUT_GRAPH_DIR = 'train_graphs'
OUTPUT_COMM_DIR  = 'train_comms'
TEST_GRAPH_FILE  = 'test_graph.txt'
TEST_COMM_FILE   = 'test_comms.txt'

NUM_TRAIN_GRAPHS = 70
SUBGRAPH_SIZE    = 100
TRAIN_RATIO      = 0.80

# Jaccard overlap: two communities with overlap above this are considered
# near-duplicates; the smaller one is discarded.
MAX_JACCARD_OVERLAP = 0.6

# How many structurally-distant anchor communities to seed the concurrent BFS
# from.  3-4 is good for 100-node subgraphs with 4-8 target communities.
NUM_ANCHORS = 4

# Community count bounds per subgraph AFTER Jaccard filtering.
# Lower bound: need genuine diversity for the trade-off to exist.
# Upper bound: too many communities makes community reward nearly free
#              (some community is always new), killing the trade-off again.
MIN_DISTINCT_COMMS = 4
MAX_DISTINCT_COMMS = 8

# Minimum Jaccard DISTANCE between anchor communities when selecting them.
# distance = 1 - jaccard_similarity.  1.0 = completely disjoint node sets.
# 0.9 means the two anchor communities share at most 10% of their nodes.
MIN_ANCHOR_DISTANCE = 0.9

# =============================================================================


def load_data():
    print(f"Loading graph from {GRAPH_FILE}...")
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        G = nx.read_edgelist(f, delimiter='\t', nodetype=int)

    print(f"Loading communities from {COMM_FILE}...")
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
    """
    Snowball BFS split of the full graph into a train region and a test region.
    The test region becomes the single large test graph for inference.
    """
    print("\nExecuting snowball macro-split...")
    target_train = int(G.number_of_nodes() * train_ratio)

    train_nodes = set()
    visited     = set()
    start       = random.choice(list(G.nodes()))
    queue       = deque([start])
    visited.add(start)

    while queue and len(train_nodes) < target_train:
        curr = queue.popleft()
        train_nodes.add(curr)
        for nb in G.neighbors(curr):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    G_train = G.subgraph(train_nodes).copy()
    G_test  = G.subgraph(set(G.nodes()) - train_nodes).copy()

    print(f"  Train region: {G_train.number_of_nodes()} nodes")
    print(f"  Test  region: {G_test.number_of_nodes()} nodes")
    return G_train, G_test


def save_test_data(G_test, node_to_comms):
    """Relabel and save the test graph and its communities."""
    print("\nSaving test data...")
    mapping = {old: new for new, old in enumerate(G_test.nodes())}
    G_test_r = nx.relabel_nodes(G_test, mapping)
    nx.write_edgelist(G_test_r, TEST_GRAPH_FILE, delimiter='\t', data=False)

    test_comms = defaultdict(list)
    for old, new in mapping.items():
        for c in node_to_comms.get(old, []):
            test_comms[c].append(new)

    with open(TEST_COMM_FILE, 'w', encoding='utf-8') as f:
        for members in test_comms.values():
            if len(members) >= 2:
                f.write('\t'.join(map(str, members)) + '\n')

    print(f"  Saved: {TEST_GRAPH_FILE}, {TEST_COMM_FILE}")


# =============================================================================
# CORE FIX 1 — Pick structurally distant anchor communities
# =============================================================================

def pick_diverse_anchors(comm_to_nodes, train_nodes_set,
                          k=NUM_ANCHORS,
                          min_size=5, max_size=50,
                          min_distance=MIN_ANCHOR_DISTANCE):
    """
    Greedily select k communities whose node sets are as disjoint as possible.

    Why this matters
    ----------------
    Single-anchor BFS creates a subgraph centred on one community.
    High-degree nodes within that community dominate influence AND cover most
    communities naturally, so the agent never faces a real trade-off.

    Starting from k distant communities ensures the subgraph genuinely spans
    different parts of the network, so the highest-influence hub nodes are
    spread across communities rather than all concentrated in one.

    Selection algorithm
    -------------------
    1. Pick the first anchor randomly from valid candidates.
    2. For each subsequent anchor, pick the candidate that maximises its
       minimum Jaccard distance to all already-selected anchors.
    3. Stop if no candidate exceeds min_distance from all selected anchors —
       the subgraph region is too homogeneous; the caller will retry.

    Returns list of (comm_id, frozenset_of_nodes) or None if selection fails.
    """
    candidates = []
    for c_id, members in comm_to_nodes.items():
        local = members.intersection(train_nodes_set)
        if min_size <= len(local) <= max_size:
            candidates.append((c_id, frozenset(local)))

    if len(candidates) < k:
        return None

    random.shuffle(candidates)
    selected = [candidates[0]]

    for _ in range(k - 1):
        best, best_dist = None, -1.0

        for cid, members in candidates:
            # Skip already selected
            if any(cid == s[0] for s in selected):
                continue

            # Minimum Jaccard distance to all currently selected anchors
            min_dist = min(
                1.0 - len(members & s_members) / len(members | s_members)
                for _, s_members in selected
            )

            if min_dist > best_dist:
                best_dist = min_dist
                best = (cid, members)

        # Only accept if this anchor is genuinely distant from all selected
        if best is None or best_dist < min_distance:
            break

        selected.append(best)

    return selected if len(selected) >= 2 else None


# =============================================================================
# CORE FIX 2 — Concurrent BFS from multiple anchors
# =============================================================================

def concurrent_bfs(G_train, anchor_communities, target_size):
    """
    Grow a subgraph simultaneously from multiple anchor communities.

    Why this matters
    ----------------
    Standard single-source BFS produces a ball that is dense at the centre
    (anchor community) and thin at the edges (other communities).
    The result: one dominant high-influence community, others as sparse
    fringe nodes.  That means picking the top-influence nodes naturally
    covers most communities — no trade-off exists.

    Concurrent BFS from k anchors grows k balls in parallel, merging at
    their boundaries.  The result is a subgraph that genuinely spans all
    k community regions with comparable density in each, so the model is
    forced to actually choose between high-influence-in-covered-community
    and lower-influence-in-novel-community.

    Algorithm
    ---------
    1. Seed each frontier with one random node from each anchor community.
    2. Each round, expand EVERY frontier by ONE neighbour (round-robin).
    3. Nodes claimed by the first frontier that reaches them stay in that
       frontier's territory, preventing one frontier from monopolising.
    4. Stop when total node count hits target_size.
    """
    # Seed frontiers with one node per anchor
    start_nodes = [random.choice(list(members))
                   for _, members in anchor_communities]

    sub_nodes = set(start_nodes)
    # One deque (frontier) per anchor
    frontiers = [deque([s]) for s in start_nodes]

    while len(sub_nodes) < target_size:
        any_progress = False

        for frontier in frontiers:
            if not frontier:
                continue

            curr = frontier.popleft()
            neighbors = list(G_train.neighbors(curr))
            random.shuffle(neighbors)  # random order prevents bias

            for nb in neighbors:
                if nb not in sub_nodes:
                    sub_nodes.add(nb)
                    frontier.append(nb)
                    any_progress = True
                    if len(sub_nodes) >= target_size:
                        break

            if len(sub_nodes) >= target_size:
                break

        if not any_progress:
            # All frontiers exhausted — connected region too small
            break

    return sub_nodes


# =============================================================================
# Jaccard filter (unchanged from original)
# =============================================================================

def filter_distinct_communities(local_comms_dict, max_jaccard):
    """
    Remove near-duplicate communities.

    Two communities with Jaccard similarity > max_jaccard are considered
    duplicates; the smaller one is discarded.
    Keeps communities with at least 2 members.
    """
    candidates = {cid: set(m) for cid, m in local_comms_dict.items()
                  if len(m) >= 2}

    # Process largest communities first so we keep the more representative one
    sorted_cands = sorted(candidates.items(),
                          key=lambda x: len(x[1]), reverse=True)
    distinct = {}

    for cid, members in sorted_cands:
        is_distinct = True
        for d_cid, d_members in distinct.items():
            inter = len(members & d_members)
            union = len(members | d_members)
            if union > 0 and inter / union > max_jaccard:
                is_distinct = False
                break
        if is_distinct:
            distinct[cid] = members

    return distinct


# =============================================================================
# Main extraction loop
# =============================================================================

def extract_subgraphs(G_train, node_to_comms, comm_to_nodes):
    """
    Extract NUM_TRAIN_GRAPHS subgraphs, each guaranteed to have between
    MIN_DISTINCT_COMMS and MAX_DISTINCT_COMMS structurally distinct
    communities, grown from multiple distant anchor communities so that
    influence and diversity are in genuine tension.
    """
    print(f"\nExtracting {NUM_TRAIN_GRAPHS} subgraphs "
          f"(anchors={NUM_ANCHORS}, "
          f"comms=[{MIN_DISTINCT_COMMS},{MAX_DISTINCT_COMMS}])...")

    os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_COMM_DIR,  exist_ok=True)

    train_nodes_set = set(G_train.nodes())
    generated = 0
    attempts  = 0

    while generated < NUM_TRAIN_GRAPHS:
        attempts += 1

        # --- Step 1: Pick k structurally distant anchor communities ---
        anchors = pick_diverse_anchors(comm_to_nodes, train_nodes_set)
        if anchors is None:
            # Couldn't find distant-enough anchors, retry
            continue

        # --- Step 2: Grow subgraph concurrently from all anchors ---
        sub_nodes = concurrent_bfs(G_train, anchors, SUBGRAPH_SIZE)

        if len(sub_nodes) < SUBGRAPH_SIZE:
            # Region too small (disconnected component), retry
            continue

        # --- Step 3: Build subgraph and relabel nodes 0..N-1 ---
        G_sub = G_train.subgraph(sub_nodes).copy()
        mapping = {old: new for new, old in enumerate(G_sub.nodes())}
        G_sub_r = nx.relabel_nodes(G_sub, mapping)

        # --- Step 4: Collect communities present in this subgraph ---
        raw_comms = defaultdict(list)
        for old, new in mapping.items():
            for c_id in node_to_comms.get(old, []):
                raw_comms[c_id].append(new)

        # --- Step 5: Remove near-duplicate communities ---
        distinct_comms = filter_distinct_communities(raw_comms,
                                                     MAX_JACCARD_OVERLAP)

        # --- Step 6: Enforce community count bounds ---
        # Lower bound: too few communities → easy for any strategy to cover all
        # Upper bound: too many communities → community reward becomes trivially
        #              free (some community is always new), killing the trade-off
        n_comms = len(distinct_comms)
        if not (MIN_DISTINCT_COMMS <= n_comms <= MAX_DISTINCT_COMMS):
            continue

        # --- Step 7: Save ---
        graph_path = os.path.join(OUTPUT_GRAPH_DIR,
                                  f'train_graph_{generated:03d}.txt')
        comm_path  = os.path.join(OUTPUT_COMM_DIR,
                                  f'train_comm_{generated:03d}.txt')

        # Write edge list (undirected: one line per edge, read_graph handles
        # both directions when called with directed=False)
        nx.write_edgelist(G_sub_r, graph_path, delimiter='\t', data=False)

        with open(comm_path, 'w', encoding='utf-8') as f:
            for members in distinct_comms.values():
                f.write('\t'.join(map(str, members)) + '\n')

        generated += 1
        if generated % 10 == 0:
            print(f"  {generated}/{NUM_TRAIN_GRAPHS} done "
                  f"(attempts so far: {attempts}, "
                  f"last subgraph: {n_comms} communities)")

    print(f"\nFinished. {generated} subgraphs in {attempts} attempts.")
    print(f"Graphs  → {OUTPUT_GRAPH_DIR}/")
    print(f"Comms   → {OUTPUT_COMM_DIR}/")


# =============================================================================

if __name__ == '__main__':
    G, node_to_comms, comm_to_nodes = load_data()
    G_train, G_test = macro_split(G, TRAIN_RATIO)
    save_test_data(G_test, node_to_comms)
    extract_subgraphs(G_train, node_to_comms, comm_to_nodes)
    print("\nAll tasks completed.")
