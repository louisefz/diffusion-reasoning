"""Pure-Python task definitions; metadata is explicitly excluded from inputs."""
import hashlib
import json
import random


def execute(tables, start, program):
    states = [start]
    for f in program:
        states.append(tables[f][states[-1]])
    return states


def table_key(tables):
    return hashlib.sha256(json.dumps(tables, separators=(",", ":")).encode()).hexdigest()


def vocabulary(n, m):
    return (["PAD", "COMPOSE", "LOOKUP", "TABLE", "START", "PROGRAM", "ANSWER"]
            + [f"S{i}" for i in range(n)] + [f"F{i}" for i in range(m)])


def encode(row, n, m):
    """The tables are lists: entry j is the image of symbol j."""
    tokens = [row["task"].upper()]
    for i, table in enumerate(row["tables"]):
        tokens += ["TABLE", f"F{i}"] + [f"S{x}" for x in table]
    tokens += ["START", f"S{row['start']}", "PROGRAM"]
    tokens += [f"F{f}" for f in row["program"]]
    tokens += ["ANSWER"]
    ids = {s: i for i, s in enumerate(vocabulary(n, m))}
    return [ids[s] for s in tokens]


def episode(rng, n, m, depth, forbidden, prefix, index):
    for _ in range(10000):
        tables = [rng.sample(range(n), n) for _ in range(m)]
        key = table_key(tables)
        if key not in forbidden:
            break
    else:
        raise ValueError("Too many episodes for the chosen permutation space")
    forbidden.add(key)
    program = [rng.randrange(m) for _ in range(depth)]
    start = rng.randrange(n)
    states = execute(tables, start, program)
    base = dict(episode_id=f"{prefix}-{index}", table_id=key, tables=tables,
                start=start, program=program, depth=depth)
    return [dict(base, task="compose", answer=states[-1], states=states),
            dict(base, task="lookup", answer=states[1], states=states[:2])]


def generate(seed, n, m, depths, count, forbidden, prefix):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        rows.extend(episode(rng, n, m, depths[i % len(depths)], forbidden, prefix, i))
    rng.shuffle(rows)
    return rows


class OnlineEpisodes:
    """Fresh paired queries with a private, checkpointable RNG.

    Holdout tables and previously emitted tables are rejected. Only table hashes
    are retained; examples and intermediate states are generated on demand.
    """
    def __init__(self, seed, n, m, depths, heldout):
        if not depths or min(depths) < 1:
            raise ValueError("Positive depths required")
        self.rng = random.Random(seed)
        self.n, self.m, self.depths = n, m, tuple(depths)
        self.heldout = set(heldout)
        self.forbidden = set(heldout)
        self.count = 0

    def next_batch(self, size):
        if size < 2 or size % 2:
            raise ValueError("Online batches need an even size >=2 for paired controls")
        rows = []
        for _ in range(size // 2):
            depth = self.depths[self.count % len(self.depths)]
            rows.extend(episode(self.rng, self.n, self.m, depth, self.forbidden,
                                "online", self.count))
            self.count += 1
        self.rng.shuffle(rows)
        return rows

    def state_dict(self):
        return dict(version=1, rng=self.rng.getstate(), count=self.count,
                    n=self.n, m=self.m, depths=self.depths,
                    seen=sorted(self.forbidden - self.heldout))

    def load_state_dict(self, state):
        if (state["version"], state["n"], state["m"], tuple(state["depths"])) != (
                1, self.n, self.m, self.depths):
            raise ValueError("Online generator configuration differs")
        seen = set(state["seen"])
        if seen & self.heldout or len(seen) != state["count"]:
            raise ValueError("Invalid online history or holdout overlap")
        self.forbidden = self.heldout | seen
        self.count = state["count"]
        self.rng.setstate(state["rng"])


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(rows, n, m):
    for r in rows:
        assert len(r["tables"]) == m
        assert all(sorted(p) == list(range(n)) for p in r["tables"])
        assert r["depth"] == len(r["program"]) > 0
        assert r["table_id"] == table_key(r["tables"])
        states = execute(r["tables"], r["start"], r["program"])
        expected = states if r["task"] == "compose" else states[:2]
        assert r["states"] == expected and r["answer"] == expected[-1]
