"""Controlled directed-graph reachability with matched edge lookup controls."""

from collections import deque
import hashlib
import random


def shortest_distances(num_nodes, edges, source):
    adjacency = [[] for _ in range(num_nodes)]
    for left, right in edges:
        adjacency[left].append(right)
    distance = [None] * num_nodes
    distance[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if distance[neighbor] is None:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def graph_id(edges):
    payload = ";".join(f"{left}>{right}" for left, right in sorted(edges))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _sample_graph(rng, num_nodes, edge_probability):
    return {
        (left, right)
        for left in range(num_nodes)
        for right in range(num_nodes)
        if left != right and rng.random() < edge_probability
    }


def _sample_exact_depth_graph(rng, num_nodes, depth, edge_probability):
    """Construct a graph with a certified source-target shortest path."""
    chain = rng.sample(range(num_nodes), depth + 1)
    source, target = chain[0], chain[-1]
    edges = {(chain[index], chain[index + 1]) for index in range(depth)}
    candidates = [(left, right) for left in range(num_nodes) for right in range(num_nodes)
                  if left != right and (left, right) not in edges]
    rng.shuffle(candidates)
    for edge in candidates:
        if rng.random() >= edge_probability:
            continue
        edges.add(edge)
        if shortest_distances(num_nodes, edges, source)[target] != depth:
            edges.remove(edge)
    return edges, source, target


def episode(rng, num_nodes, depth, reach_answer, lookup_answer,
            seen_graphs, split, episode_id, max_attempts=20_000):
    """Return one reachability row and one matched direct-edge control."""
    edge_probability = max(0.035, 0.16 - 0.01 * depth)
    for _ in range(max_attempts):
        fixed_pair = None
        if reach_answer:
            edges, reach_source, reach_target = _sample_exact_depth_graph(
                rng, num_nodes, depth, edge_probability,
            )
            fixed_pair = (reach_source, reach_target)
        else:
            edges = _sample_graph(rng, num_nodes, edge_probability)
        if len(edges) < max(2, depth):
            continue
        identifier = graph_id(edges)
        if identifier in seen_graphs:
            continue
        distances = [shortest_distances(num_nodes, edges, source)
                     for source in range(num_nodes)]
        if reach_answer:
            reach_candidates = [fixed_pair]
        else:
            reach_candidates = [
                (source, target)
                for source in range(num_nodes)
                for target in range(num_nodes)
                if source != target and distances[source][target] is None
            ]
        edge_set = set(edges)
        if lookup_answer:
            lookup_candidates = sorted(edge_set)
        else:
            lookup_candidates = [
                (source, target)
                for source in range(num_nodes)
                for target in range(num_nodes)
                if source != target and (source, target) not in edge_set
            ]
        if not reach_candidates or not lookup_candidates:
            continue
        reach_source, reach_target = rng.choice(reach_candidates)
        lookup_source, lookup_target = rng.choice(lookup_candidates)
        seen_graphs.add(identifier)
        base = {
            "edges": [list(edge) for edge in sorted(edge_set)],
            "depth": depth,
            "episode_id": f"{split}-{episode_id}",
            "graph_id": identifier,
            "split": split,
        }
        # states[k] says whether the target is reachable within k propagation
        # rounds. Positive examples switch only at their controlled shortest
        # path depth; negative examples remain zero.
        reach_states = [0] * (depth + 1)
        if reach_answer:
            reach_states[-1] = 1
        reach = dict(
            base, task="reach", source=reach_source, target=reach_target,
            answer=int(reach_answer), states=reach_states,
        )
        lookup = dict(
            base, task="lookup", source=lookup_source, target=lookup_target,
            answer=int(lookup_answer), states=[0, int(lookup_answer)],
        )
        return reach, lookup
    raise RuntimeError(
        f"Could not sample graph for depth={depth}, reach={reach_answer}, "
        f"lookup={lookup_answer} after {max_attempts} attempts"
    )


def generate(seed, num_nodes, depths, episodes, seen_graphs, split):
    rng = random.Random(seed)
    rows = []
    depth_count = len(depths)
    for index in range(episodes):
        depth = depths[index % depth_count]
        cycle = index // depth_count
        reach_answer = cycle % 2 == 0
        lookup_answer = (cycle // 2) % 2 == 0
        rows.extend(episode(
            rng, num_nodes, depth, reach_answer, lookup_answer,
            seen_graphs, split, index,
        ))
    return rows


def validate(rows, num_nodes):
    graph_ids = set()
    for row in rows:
        edges = {tuple(edge) for edge in row["edges"]}
        assert all(0 <= left < num_nodes and 0 <= right < num_nodes and left != right
                   for left, right in edges)
        assert graph_id(edges) == row["graph_id"]
        distances = shortest_distances(num_nodes, edges, row["source"])
        if row["task"] == "reach":
            if row["answer"]:
                assert distances[row["target"]] == row["depth"]
                assert row["states"] == [0] * row["depth"] + [1]
            else:
                assert distances[row["target"]] is None
                assert row["states"] == [0] * (row["depth"] + 1)
        else:
            assert row["task"] == "lookup"
            assert row["answer"] == int((row["source"], row["target"]) in edges)
            assert row["states"] == [0, row["answer"]]
        graph_ids.add(row["graph_id"])
    assert len(graph_ids) * 2 == len(rows)
