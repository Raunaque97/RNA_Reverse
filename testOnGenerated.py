import numpy as np
import rnalib as rnalib
import time
import random
import RNA
import sys
import os
import torch
import pickle as pkl
import matplotlib.pyplot as plt

def generate_train_set(length = 10):
    if os.path.isfile('train_dict_{}.pkl'.format(length)):
        return pkl.load(open('train_dict_{}.pkl'.format(length),'rb'))
    else:   
        seq_dict = {}
        for i in range(500):
            random_seq = [random.randint(0,3) for i in range(length)]
            structure, energy = RNA.fold(rnalib.sequence_to_string(random_seq))
            if structure not in seq_dict:
                seq_dict[structure] = [random_seq]
            else:
                seq_dict[structure].append(random_seq)
        with open("train_dict_{}.pkl".format(length), "wb") as f:
            pkl.dump(seq_dict, f)
        return seq_dict

def array2actionTuple(a):
    idx = np.argmax(np.max(a,1))
    base = np.argmax(a[idx])
    base_list = []
    for i in range(a.shape[0]):
        for b in range(4):
            if a[i,b] == a[idx,base]:
                base_list.append((i,b))
    return random.choice(base_list)

def findRNA(structure,policy=None ,useRandomPolicy=False):
    env = rnalib.RNAEnvironment(goal=structure,max_steps=1000)
    while not env.terminated:
        if useRandomPolicy:
            a = (random.randint(0,len(structure)-1),random.randint(0,3))
        else:
            if policy is None:
                print('Need RNAPolicy if not random')
            a = array2actionTuple(policy.get_action(np.expand_dims(env.state.T, axis=0)))
        prevState = env.state
        r = env.step(a)
        if r == 5:
            return 1
        if(env.state == prevState).all():
            a = (random.randint(0,len(structure)-1),random.randint(0,3))
            r = env.step(a)
        if r == 5:
            return 1
    return 0

lens = [10, 12, 15, 20, 30]
randomSuccess=[]
DQNSuccess=[]

# policy = rnalib.RNA_BiLSTM_Policy(hidden_size= 15, num_layers= 4)
policy = torch.load('DQN_policy')

for l in lens:
    train_dict = generate_train_set(length = l)
    testStrucks =  random.sample(list(train_dict.keys()),min(100, len(train_dict.keys())))
    start_time = time.time()
    print(str(l)+' '+str(len(train_dict.keys())), end=':\t')
    successCount = 0
    for i in range(100):
        st = testStrucks[i%len(testStrucks)]
        successCount += findRNA(structure=st, useRandomPolicy=True)
    randomSuccess.append(successCount)

    successCount = 0
    for i in range(100):
        st = testStrucks[i%len(testStrucks)]
        successCount += findRNA(structure=st,policy=policy ,useRandomPolicy=False)
    DQNSuccess.append(successCount)
    print('finished '+str(time.time() - start_time)+' rand '+str(randomSuccess[-1])+' DQN '+str(DQNSuccess[-1]))

plt.plot(lens, randomSuccess)
plt.plot(lens, DQNSuccess)
plt.show()




