__all__ = ["block_counts"]


def block_counts(N, P):
    counts = [N // P] * P
    for i in range(N % P):
        counts[i] += 1
    return counts
