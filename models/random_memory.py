import torch
import random
import numpy as np
import copy


class RandomMemory:
    RMsize = 0
    n_saved_batches = 0

    def __init__(self, patterns_shape=(0, 0, 0), rmsize=1500):
        self.RMsize = rmsize
        self.patterns_shape = patterns_shape
        self.memory = {'activations': torch.zeros((self.RMsize, *patterns_shape[1:]), dtype=torch.float32),
                       'labels': torch.zeros(self.RMsize, dtype=torch.long)}

    def addPatterns(self, patterns, labels, n_batches=1):

        if self.n_saved_batches == 0:
            h = self.RMsize
        else:
            h = self.RMsize // self.n_saved_batches

        self.n_saved_batches += n_batches

        add_id = np.random.choice(range(patterns.shape[0]), size=h)
        replace_id = np.random.choice(range(self.RMsize), size=h)

        print(f'add {h} patterns to RM with size {self.RMsize}')

        self.memory['activations'][replace_id] = copy.deepcopy(patterns[add_id])
        self.memory['labels'][replace_id] = copy.deepcopy(labels[add_id])




    def getMemory(self):
        return self.memory

    def getsize(self):
        return self.RMsize

    def getLabels(self):
        return self.memory['labels']

    def getActivations(self):
        return self.memory['activations']

if __name__ == '__main__':
    rm = RandomMemory((10, 10, 10))
    print(rm.memory)

