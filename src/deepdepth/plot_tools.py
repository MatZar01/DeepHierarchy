"""
plot tools for graph/groups in deepdepth
"""

import matplotlib.pyplot as plt
from collections import Counter


def plot_partition(partition):
    """
    Plots a histogram of group sizes from a dictionary partition where keys are words
    and values are group labels. The histogram shows how many groups have a given size.

    Parameters:
    partition (dict): A dictionary where keys are words and values are group labels.
    """
    # Count the frequency of each group
    group_counts = Counter(partition.values())

    # Count the sizes of each group
    size_counts = Counter(group_counts.values())

    # Sort group sizes for plotting
    sizes = sorted(size_counts.keys())
    counts = [size_counts[size] for size in sizes]

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.bar(sizes, counts, width=0.7, edgecolor="black", alpha=0.7)
    plt.xlabel("Group Size", fontsize=14)
    plt.ylabel("Number of Groups", fontsize=14)
    plt.title("Histogram of Group Sizes", fontsize=16)
    plt.xticks(sizes)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()
