"""Coding pass-rate evaluation."""
from ..oracle import coding_correct


def coding_pass_rate(skill, n_tasks=64):
    return sum(1 for i in range(n_tasks) if coding_correct("task-%d" % i, skill)) / n_tasks
