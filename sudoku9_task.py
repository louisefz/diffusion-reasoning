"""9x9 Sudoku generation with solver-derived difficulty diagnostics."""

import random


N, BOX = 9, 3
ALL = set(range(1, 10))


def candidates(grid, pos):
    row, col = divmod(pos, N)
    used = set(grid[row*N:(row+1)*N]) | {grid[r*N+col] for r in range(N)}
    br, bc = row // BOX * BOX, col // BOX * BOX
    used |= {grid[(br+r)*N+bc+c] for r in range(BOX) for c in range(BOX)}
    return ALL - used


def solve_with_metrics(puzzle, limit=2):
    grid, found = list(puzzle), []
    metrics = {"nodes": 0, "guesses": 0, "max_guess_depth": 0,
               "propagation_rounds": 0, "forced_moves": 0}

    def search(guess_depth):
        if len(found) >= limit:
            return
        metrics["nodes"] += 1
        local_rounds = 0
        forced = []
        while True:
            singles = []
            for pos, value in enumerate(grid):
                if value == 0:
                    options = candidates(grid, pos)
                    if not options:
                        for p, _ in reversed(forced): grid[p] = 0
                        return
                    if len(options) == 1:
                        singles.append((pos, next(iter(options))))
            if not singles:
                break
            local_rounds += 1
            # Apply sequentially and recheck validity on the next round.
            for pos, value in singles:
                if grid[pos] == 0 and value in candidates(grid, pos):
                    grid[pos] = value; forced.append((pos, value))
        metrics["propagation_rounds"] = max(metrics["propagation_rounds"], local_rounds)
        metrics["forced_moves"] = max(metrics["forced_moves"], len(forced))
        empty = [pos for pos, value in enumerate(grid) if value == 0]
        if not empty:
            found.append(tuple(grid))
        else:
            pos = min(empty, key=lambda p: len(candidates(grid, p)))
            options = sorted(candidates(grid, pos))
            if len(options) > 1:
                metrics["guesses"] += 1
                metrics["max_guess_depth"] = max(metrics["max_guess_depth"], guess_depth + 1)
            for value in options:
                grid[pos] = value; search(guess_depth + (len(options) > 1)); grid[pos] = 0
                if len(found) >= limit:
                    break
        for pos, _ in reversed(forced):
            grid[pos] = 0

    search(0)
    return found, metrics


def random_solution(rng):
    pattern = lambda r, c: (BOX * (r % BOX) + r // BOX + c) % N
    bands = [0, 1, 2]; stacks = [0, 1, 2]
    rng.shuffle(bands); rng.shuffle(stacks)
    rows = [b*BOX+r for b in bands for r in rng.sample(range(BOX), BOX)]
    cols = [s*BOX+c for s in stacks for c in rng.sample(range(BOX), BOX)]
    nums = rng.sample(range(1, 10), 9)
    return tuple(nums[pattern(r, c)] for r in rows for c in cols)


def episode(rng, blanks, max_attempts=100):
    """Carve a unique puzzle and return solver-derived difficulty metrics."""
    for _ in range(max_attempts):
        answer = random_solution(rng)
        puzzle = list(answer)
        positions = list(range(81)); rng.shuffle(positions)
        removed = 0
        for pos in positions:
            old = puzzle[pos]; puzzle[pos] = 0
            found, _ = solve_with_metrics(puzzle, limit=2)
            if len(found) == 1:
                removed += 1
                if removed == blanks:
                    solutions, metrics = solve_with_metrics(puzzle, limit=2)
                    return {"puzzle": puzzle, "answer": list(solutions[0]),
                            "blanks": blanks, **metrics}
            else:
                puzzle[pos] = old
    raise RuntimeError(f"could not generate unique 9x9 Sudoku with {blanks} blanks")


def verify(row, answer=None):
    grid = list(answer or row["answer"])
    if len(grid) != 81:
        return False
    for index, clue in enumerate(row["puzzle"]):
        if clue and grid[index] != clue:
            return False
    return (all(set(grid[r*N:(r+1)*N]) == ALL for r in range(N)) and
            all({grid[r*N+c] for r in range(N)} == ALL for c in range(N)) and
            all({grid[(br+r)*N+bc+c] for r in range(BOX) for c in range(BOX)} == ALL
                for br in range(0, N, BOX) for bc in range(0, N, BOX)))

