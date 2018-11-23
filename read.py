import numpy as np
import rnalib as rnalib
import time
import random
import RNA
import sys
import os
import torch
import pickle as pkl

output = []

f = open( 'journal.pcbi.1006176.s001.csv', 'rU' ) #open the file in read universal mode
for line in f:
	cells = line.split( "," )
	output.append( cells[ 2 ].strip()  ) #since we want the first, second and third column

f.close()
output = output[1:]
# print(output)


def array2actionTuple(a):
	idx = np.argmax(np.max(a,1))
	base = np.argmax(a[idx])

	base_list = []
	for i in range(a.shape[0]):
		for b in range(4):
			if a[i,b] == a[idx,base]:
				base_list.append((i,b))

	return random.choice(base_list)


def DQNPolicy(random_policy=False):
	couter,total_steps = 0,0
	for puzzle in output:
		if puzzle[0] == 'A' or puzzle[0] == 'U' or puzzle[0] == 'G' or puzzle[0] == 'C':
			structure, _ = RNA.fold(puzzle)
		else:
			structure = puzzle

		env = rnalib.RNAEnvironment(goal=structure,max_steps=1000)

		policy = rnalib.RNA_BiLSTM_Policy(hidden_size= 15, num_layers= 4)
		policy = torch.load('DQN_policy')

		# print(len(structure))

		steps = 0
		start_time = time.time()
		end_time = start_time + 30*60
		f = 0
		while not env.terminated and time.time() < end_time:
			if random_policy:
				a = (random.randint(0,len(structure)-1),random.randint(0,3))
			else:
				a = array2actionTuple(policy.get_action(np.expand_dims(env.state.T, axis=0)))
			r = env.step(a)
			steps += 1
			if r == 5:
				print('SUCCESS',structure,len(structure),steps)
				counter += 1
				total_steps += steps
				f = 1
				break

		if f == 0:
			print('FAILED',structure,len(structure),steps)
			total_steps += steps

	print("TOTAL SOLVED : ",counter, "AVERAGE :", total_steps/100 )


print("Testing RANDOM POLICY")
DQNPolicy(random_policy=True)
print("\n\n\n\n")
print("Testing DQN")
DQNPolicy()



