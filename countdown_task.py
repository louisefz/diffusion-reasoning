"""Small, verifier-backed Countdown arithmetic tasks.

Expressions use reverse Polish notation (RPN).  Every supplied number must be
used exactly once; subtraction stays positive and division must be exact.
"""

from collections import Counter
from functools import lru_cache
import random


OPS = ("+", "-", "*", "/")
NUMBER_POOL = tuple(range(1, 11)) + (25, 50, 75, 100)


def apply_op(left, right, op):
    if op == "+":
        return left + right
    if op == "-" and left > right:
        return left - right
    if op == "*":
        return left * right
    if op == "/" and right and left % right == 0:
        return left // right
    return None


def evaluate_rpn(expression, numbers):
    stack, used = [], []
    for token in expression.split():
        if token in OPS:
            if len(stack) < 2:
                raise ValueError("operator without two operands")
            right, left = stack.pop(), stack.pop()
            value = apply_op(left, right, token)
            if value is None:
                raise ValueError("invalid arithmetic operation")
            stack.append(value)
        else:
            value = int(token)
            used.append(value)
            stack.append(value)
    if len(stack) != 1 or Counter(used) != Counter(numbers):
        raise ValueError("expression must use every supplied number exactly once")
    return stack[0]


def solve_min_depth(numbers, target, cap=1000):
    """Return (minimum expression-tree depth, one RPN expression), or None."""
    n = len(numbers)
    states = {}
    for index, number in enumerate(numbers):
        states[1 << index] = {number: (0, str(number))}
    for size in range(2, n + 1):
        for mask in range(1, 1 << n):
            if mask.bit_count() != size:
                continue
            values = {}
            sub = (mask - 1) & mask
            while sub:
                other = mask ^ sub
                if other and sub < other and sub in states and other in states:
                    for lv, (ld, le) in states[sub].items():
                        for rv, (rd, re) in states[other].items():
                            for left, right, lex, rex in ((lv, rv, le, re), (rv, lv, re, le)):
                                for op in OPS:
                                    value = apply_op(left, right, op)
                                    if value is None or value <= 0 or value > cap:
                                        continue
                                    candidate = (max(ld, rd) + 1, f"{lex} {rex} {op}")
                                    if value not in values or candidate[0] < values[value][0]:
                                        values[value] = candidate
                sub = (sub - 1) & mask
            states[mask] = values
    return states[(1 << n) - 1].get(target)


def episode(rng, depth, max_attempts=20_000):
    """Sample a problem whose certified minimum expression depth is `depth`."""
    count = depth + 1
    for _ in range(max_attempts):
        numbers = [rng.choice(NUMBER_POOL) for _ in range(count)]
        # Generate a reachable target, then certify its minimum tree depth.
        value = numbers[0]
        for number in numbers[1:]:
            valid = [(op, apply_op(value, number, op)) for op in OPS]
            valid = [(op, result) for op, result in valid if result and result <= 500]
            if not valid:
                break
            _, value = rng.choice(valid)
        else:
            solved = solve_min_depth(numbers, value)
            if solved is not None and solved[0] == depth:
                return {"numbers": numbers, "target": value, "depth": depth,
                        "answer": solved[1], "shape": "minimum",
                        "minimum_depth_certified": True}
    raise RuntimeError(f"could not sample certified Countdown depth {depth}")


def structured_episode(rng, depth, shape="left", max_attempts=20_000):
    """Generate a valid expression with a known tree depth and orientation.

    Unlike :func:`episode`, this certifies the supplied solution's tree depth,
    not the minimum depth among every possible alternative solution.
    """
    if shape not in ("left", "right"):
        raise ValueError("shape must be left or right")
    for _ in range(max_attempts):
        numbers = [rng.choice(NUMBER_POOL) for _ in range(depth + 1)]
        nodes = [(number, str(number), 0) for number in numbers]
        while len(nodes) > 1:
            if shape == "left":
                left, right = nodes.pop(0), nodes.pop(0)
                insert_at = 0
            else:
                left, right = nodes.pop(-2), nodes.pop(-1)
                insert_at = len(nodes)
            candidates = []
            for first, second in ((left, right), (right, left)):
                for op in OPS:
                    value = apply_op(first[0], second[0], op)
                    if value is not None and 0 < value <= 500:
                        candidates.append((value, f"{first[1]} {second[1]} {op}",
                                           max(first[2], second[2]) + 1))
            if not candidates:
                break
            nodes.insert(insert_at, rng.choice(candidates))
        if len(nodes) == 1 and nodes[0][2] == depth:
            value, expression, actual_depth = nodes[0]
            row = {"numbers": numbers, "target": value, "depth": actual_depth,
                   "answer": expression, "shape": shape,
                   "minimum_depth_certified": False}
            if verify(row):
                return row
    raise RuntimeError(f"could not sample structured Countdown depth {depth}")


def verify(row, expression=None):
    return evaluate_rpn(expression or row["answer"], row["numbers"]) == row["target"]
