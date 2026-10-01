"""Generation and verification utilities for unique 4x4 Sudoku puzzles."""

import random


N = 4
BOX = 2


def valid(grid):
    wanted = {1, 2, 3, 4}
    if any(set(grid[r * N:(r + 1) * N]) != wanted for r in range(N)):
        return False
    if any({grid[r * N + c] for r in range(N)} != wanted for c in range(N)):
        return False
    for br in range(0, N, BOX):
        for bc in range(0, N, BOX):
            if {grid[(br + r) * N + bc + c] for r in range(BOX) for c in range(BOX)} != wanted:
                return False
    return True


def solutions(puzzle, limit=2):
    grid = list(puzzle)
    found = []
    def search():
        if len(found) >= limit:
            return
        try:
            pos = grid.index(0)
        except ValueError:
            found.append(tuple(grid)); return
        row, col = divmod(pos, N)
        used = set(grid[row*N:(row+1)*N]) | {grid[r*N+col] for r in range(N)}
        br, bc = row // BOX * BOX, col // BOX * BOX
        used |= {grid[(br+r)*N+bc+c] for r in range(BOX) for c in range(BOX)}
        for value in range(1, N + 1):
            if value not in used:
                grid[pos] = value; search(); grid[pos] = 0
    search()
    return found


def random_solution(rng):
    base = [1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]
    symbols = [1, 2, 3, 4]; rng.shuffle(symbols)
    mapped = [symbols[x - 1] for x in base]
    row_bands = [[0, 1], [2, 3]]; col_bands = [[0, 1], [2, 3]]
    rng.shuffle(row_bands); rng.shuffle(col_bands)
    for band in row_bands: rng.shuffle(band)
    for band in col_bands: rng.shuffle(band)
    rows = sum(row_bands, []); cols = sum(col_bands, [])
    return tuple(mapped[r*N+c] for r in rows for c in cols)


def episode(rng, blanks, max_attempts=1000):
    for _ in range(max_attempts):
        answer = random_solution(rng)
        puzzle = list(answer)
        positions = list(range(N*N)); rng.shuffle(positions)
        removed = 0
        for pos in positions:
            old = puzzle[pos]; puzzle[pos] = 0
            if len(solutions(puzzle, 2)) == 1:
                removed += 1
                if removed == blanks:
                    return {"puzzle": puzzle, "answer": list(answer), "blanks": blanks}
            else:
                puzzle[pos] = old
    raise RuntimeError(f"could not sample unique Sudoku with {blanks} blanks")


def verify(row, answer=None):
    candidate = list(answer or row["answer"])
    return valid(candidate) and all(p == 0 or p == a for p, a in zip(row["puzzle"], candidate))

