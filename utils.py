import numpy as np
import random

def get_pre(branch, inputlen):
    arr = np.zeros((branch, inputlen))
    count = inputlen // branch
    remainder = inputlen % branch
    if count > 0:
        index = list(np.arange(inputlen))
        for i in range(branch):
            slice = random.sample(index, count)
            for item in slice:
                arr[i][item] = 1.
                index.remove(item)
        dx = random.sample(range(branch), remainder)
        for (x, item) in zip(dx, index):
            arr[x][item] = 1.
    else:
        dx = random.sample(range(branch), inputlen)
        dy = random.sample(range(inputlen), inputlen)
        for (x, y) in zip(dx, dy):
            arr[x][y] = 1.
    return arr

def get_random_mask(branch, inputlen, outputlen):
    pre = get_pre(branch, inputlen)
    for i in range(outputlen - 1):
        tem = get_pre(branch, inputlen)
        pre = np.concatenate((pre, tem), axis=0)
    return pre.astype(np.float32)

def get_arry1(branch, inputlen, outputlen, replace=False):
    mask = np.zeros([branch * outputlen, inputlen])
    section_size = int(inputlen / branch)
    length = branch * section_size
    def mapping(i):
        idx_map = np.random.choice(inputlen, length, replace=replace)
        idx_map = idx_map.reshape([branch, -1])
        mask[(i * branch + np.arange(0, branch))[:, np.newaxis], idx_map] = 1
    list(map(mapping, range(outputlen)))
    return mask.astype(np.float32)

def get_mask(branch, inputlen, outputlen):
    if inputlen % branch == 0:
        return get_arry1(branch, inputlen, outputlen)
    else:
        return get_random_mask(branch, inputlen, outputlen)
