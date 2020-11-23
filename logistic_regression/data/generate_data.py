import argparse
from typing import Tuple

import numpy as np


def generate_data(n: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    alpha = np.random.uniform(-1, 1)
    beta = np.random.uniform(-1, 1)
    x = np.random.randn(n)
    y = (alpha * x + beta >= 0).astype(np.int)

    return x[:, None], y[:, None]


def main(path_to_store: str, n: int = 1000, seed: int = 42) -> None:
    np.random.seed(seed)
    x, y = generate_data(n)
    yx = np.hstack((y, x))
    np.savetxt(path_to_store, yx, delimiter="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_store", type=str, required=True, help="path to store the data")
    parser.add_argument("--n", type=int, default=1000, help="number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    main(**vars(args))
