# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import shuffle

NUM_CITIES = 42
STEADY_STATE = 1000


class Tsp:

    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()
        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(
                c, pos=(np.random.random(), np.random.random()))

    def distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0])**2 +
                                                         (pos1[1] - pos2[1])**2))

    def evaluate_solution(self, solution: np.array) -> float:
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self.distance(n1, n2)
        return total_cost

    def plot(self, path: np.array = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        plt.figure(figsize=(12, 5))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink')
        if path is not None:
            plt.title(f"Current path: {self.evaluate_solution(path):,}")
        plt.show()

    @property
    def graph(self) -> nx.digraph:
        return self._graph


def get_indexes(max_int, counts):

    # Select indexes that were the least used

    i1, i2 = np.random.choice(range(NUM_CITIES), 2, p=counts/sum(counts))

    while i1 == i2:
        i2 = np.random.choice(range(NUM_CITIES), 1, p=counts/sum(counts))[0]

    counts[0:i1] += 1
    counts[i1+1::] += 1
    counts[0:i2] += 1
    counts[i2+1::] += 1

    if i1 > i2:
        tmp = i1
        i1 = i2
        i2 = tmp

    return i1, i2


def swap(new_solution, indexes):

    i1, i2 = indexes[0], indexes[1]
    temp = new_solution[i1]
    new_solution[i1] = new_solution[i2]
    new_solution[i2] = temp

    return new_solution


def scramble(new_solution, counts):

    for i in range(np.random.randint(new_solution.shape[0])):
        new_solution = swap(new_solution, get_indexes(
            new_solution.shape[0], counts))

    return new_solution


def insert(new_solution, indexes):

    i1, i2 = indexes[0], indexes[1]

    tmp = list(new_solution[i1+1:i2])
    new_solution[i1+1] = new_solution[i2]

    new_solution[i1+2:i2+1] = tmp

    return new_solution


def inversion(new_solution, indexes):

    i1, i2 = indexes[0], indexes[1]
    new_solution[i1:i2] = new_solution[i1:i2][::-1]

    return new_solution


def tweak(solution: np.array, counts, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    p = None

    while p is None or p < pm:

        # Randomly select a mutation strategy

        rand = np.random.randint(0, NUM_ALGO)

        if rand == 0:
            new_solution = swap(
                new_solution, get_indexes(new_solution.shape[0], counts))
        elif rand == 1:
            new_solution = scramble(
                new_solution, counts)
        elif rand == 2:
            new_solution = insert(
                new_solution, get_indexes(new_solution.shape[0], counts))
        else:
            new_solution = inversion(
                new_solution, get_indexes(new_solution.shape[0], counts))

        p = np.random.random()

    return new_solution, counts


NUM_ALGO = 4


def main():

    problem = Tsp(NUM_CITIES)

    counts = np.ones(NUM_CITIES)

    solution = np.array(range(NUM_CITIES))
    np.random.shuffle(solution)
    solution_cost = problem.evaluate_solution(solution)
    problem.plot(solution)

    history = [(0, solution_cost)]
    steady_state = 0
    step = 0
    while steady_state < STEADY_STATE:
        step += 1
        steady_state += 1
        new_solution, counts = tweak(solution, counts, pm=.5)
        new_solution_cost = problem.evaluate_solution(new_solution)
        if new_solution_cost < solution_cost:
            solution = new_solution
            solution_cost = new_solution_cost
            history.append((step, solution_cost))
            steady_state = 0
    problem.plot(solution)


if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    main()
