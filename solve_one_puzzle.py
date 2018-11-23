import numpy as np
import rnalib as rnalib
import time
import random
import RNA
import sys
import os
import torch
import pickle as pkl

def array2actionTuple(a):
    idx = np.argmax(np.max(a,1))
    base = np.argmax(a[idx])

    base_list = []
    for i in range(a.shape[0]):
        for b in range(4):
            if a[i,b] == a[idx,base]:
                base_list.append((i,b))

    return random.choice(base_list)
# Attempt to solve a single puzzle.  The target structure in dot bracket notation
# should be supplied as a command line argument.
puzzle = sys.argv[1]
if puzzle[0] == 'A' or puzzle[0] == 'U' or puzzle[0] == 'G' or puzzle[0] == 'C':
    structure, _ = RNA.fold(puzzle)
else:
    structure = puzzle

env = rnalib.RNAEnvironment(goal=structure,max_steps=1000)

policy = rnalib.RNA_BiLSTM_Policy(hidden_size= 15, num_layers= 4)
policy = torch.load('DQN_policy')

print(len(structure))
print(puzzle)
print(structure)
steps = 0
start_time = time.time()
end_time = start_time + 30*60
f = 0
while not env.terminated and time.time() < end_time:
    a = array2actionTuple(policy.get_action(np.expand_dims(env.state.T, axis=0)))
    # a = (random.randint(0,len(structure)-1),random.randint(0,3))
    r = env.step(a)
    steps += 1
    if steps%100==0:
        print(str(steps)+'\t'+rnalib.sequence_to_string(env.sequence)+'\t'+rnalib.sequence_to_bracket(env.sequence)) 

    if r == 5:
        print('SUCCESS')
        print(structure)
        print(rnalib.sequence_to_string(env.sequence))
        print('time:', time.time()-start_time)
        print('steps:', steps)
        f = 1
        break

if f == 0:
    print('FAILED')
    print('time:', time.time()-start_time)
    print('steps:', steps)
