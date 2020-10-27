import torch
import random
import numpy as np
import copy


class RandomMemory:
    RMsize = 0
    n_saved_classes = 0

    def __init__(self, patterns_shape, rmsize=1500):
        self.RMsize = rmsize
        self.patterns_shape = patterns_shape
        self.memory = {'activations': torch.zeros((self.RMsize, *patterns_shape)),
                       'labels': None}

    def addPatterns(self, patterns, labels):
        self.n_saved_classes += 1
        h = self.RMsize // self.n_saved_classes

        add_id = np.random.choice(range(patterns.size[0]), size=h)
        replace_id = np.random.choice(range(self.RMsize), size=h)

        self.memory['activations'][replace_id] = copy.deepcopy(patterns[add_id])
        self.memory['labels'][replace_id] = copy.deepcopy(labels[add_id])

    def getMemory(self):
        return self.memory

if __name__ == '__main__':
    rm = RandomMemory((10, 10, 10))
    print(rm.memory)

