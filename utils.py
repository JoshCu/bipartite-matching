import numpy as np
import argparse
from pathlib import Path
from functools import lru_cache


@lru_cache
def load_file(filename: str) -> np.ndarray:
    """
    Load a file containing a matrix of weights and convert it into a numpy array.

    Args:
        filename (str): The path to the file.

    Returns:
        np.ndarray: The matrix of weights.
    """
    with open(filename, "r") as file:
        file_content = [[int(weight) for weight in line.strip().split()] for line in file]
    matrix = np.array(file_content)
    return matrix


def matrix_to_adj_list(matrix: np.ndarray) -> dict:
    """
    Convert a matrix of weights into an adjacency list.

    Args:
        matrix (np.ndarray): The matrix of weights.

    Returns:
        dict: The adjacency list.
    """
    adj_hash = {}
    for i in range(matrix.shape[0]):
        # Find indices of nodes that i is connected to
        connected_nodes = np.where(matrix[i] == 1)[0]
        adj_hash[i] = list(connected_nodes)
    return adj_hash


def parse_args():
    """
    Parse command line arguments.

    Returns:
        str: The file name.
    """
    parser = argparse.ArgumentParser(description="Travelling Salesman Problem")
    parser.add_argument("filename", type=str, help="Path to the file")
    args = parser.parse_args()
    return Path(args.filename)
