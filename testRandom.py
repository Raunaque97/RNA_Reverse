import numpy as np
import rnalib as rnalib
import random
import RNA
import sys
import os
import pickle as pkl
import matplotlib.pyplot as plt

def TestRandomPolicy():
    structure = random.choice(list(train_dict.keys()))
    objectiveSeq = train_dict[structure][0]
    N = len(objectiveSeq)
    env = rnalib.RNAEnvironment(goal = structure, max_steps = 10000)

    max_step = 10e4
    steps = 0
    while not env.terminated and steps < max_step:
        a = (random.randint(0,N-1),random.randint(0,3))
        r = env.step(a)
        steps += 1

    return steps

def generate_train_set(length = 10):
    if os.path.isfile('train_dict_{}.pkl'.format(length)):
        return pkl.load(open('train_dict_{}.pkl'.format(length),'rb'))
    else:   
        seq_dict = {}
        for i in range(10000): ####### hyper parameter
            random_seq = [random.randint(0,3) for i in range(length)]
            structure, energy = RNA.fold(rnalib.sequence_to_string(random_seq))

            if structure not in seq_dict:
                seq_dict[structure] = [random_seq]
            else:
                seq_dict[structure].append(random_seq)

        with open("train_dict_{}.pkl".format(length), "wb") as f:
            pkl.dump(seq_dict, f)

        return seq_dict

lens = [10, 12, 15, 20, 30]
train_dict = {}

data = []
smax = []
smin = []
savg = []

for l in lens:
    print(l)
    train_dict = generate_train_set(length = l)
    s = []
    for _ in range(1000):
        s.append(TestRandomPolicy())
    # plt.boxplot(s)
    # plt.show()
    data.append(s)
    smax.append(max(s))
    smin.append(min(s))
    savg.append(sum(s)/len(s))

# plt.boxplot(data)
plt.plot(lens, smax)
plt.plot(lens, smin)
plt.plot(lens, savg)
plt.show()

# TestRandomPolicy()
