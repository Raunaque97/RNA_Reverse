import numpy as np
from multiprocessing.dummy import Pool
import rnalib as rnalib
import random
import RNA

def getBestActions(init_seq, final_seq):
	ret = []
	for i in range(len(init_seq)):
		if init_seq[i] != final_seq[i]:
			ret.append((i, final_seq[i]))
	return ret

def actionTuple2Array(rna_length,a):
	x = np.zeros((rna_length,4))
	x[a[0]][a[1]] = 1
	return x

def array2actionTuple(a):
	idx = np.argmax(np.max(a,1))
	base = np.argmax(a[idx])
	return (idx, base)

def train_supervised(policy):
	step = 50000
	policy.train()
	for i in range(step):
		structure = random.choice(list(train_dict.keys()))
		objectiveSeq = train_dict[structure][0]				# TODO select randomly insr=tead of 1st 
		N = len(objectiveSeq)
		structureTemp, energy = RNA.fold(rnalib.sequence_to_string(objectiveSeq))
		if structureTemp != structure:
			print('error 4534')
			exit()
		env = rnalib.RNAEnvironment(goal = structure, max_steps = 1000)

		# randomSeq = getRandomRNASeq(len(objectiveSeq))
		optimum_actions = getBestActions(env.sequence, objectiveSeq)
		# print(optimum_actions)
		if i%1000 == 0:
			print('i='+str(i)+'\t'+structure+'\tobj= ' +str(objectiveSeq)+'\tinitSeq='+str(env.sequence))
		for a_tup in optimum_actions:
			rnalib.update(actionTuple2Array(N, a_tup), policy, env)
			reward = env.step(a_tup)
			###########
			if i%10000 == 0:
				a = array2actionTuple(policy.get_action(np.expand_dims(env.state.T, axis=0)))
				print('train_action = '+str(a_tup)+'\t policy = '+str(a))



def TestPolicy(policy):
	structure = random.choice(list(train_dict.keys()))
	objectiveSeq = train_dict[structure][0]
	N = len(objectiveSeq)
	env = rnalib.RNAEnvironment(goal = structure, max_steps = 50)
	# optimum_actions = getBestActions(env.sequence, objectiveSeq)
	# print(env.terminated)
	while not env.terminated:
		a = array2actionTuple(policy.get_action(np.expand_dims(env.state.T, axis=0)))
		print('step: '+str(env.count)+'\tseq = '+str(env.sequence)+'\t policy = '+str(a)+'\t objSeQ: '+str(objectiveSeq))
		reward = env.step(a)

	pass

def generate_train_set():
	seq_dict = {}
	length = 10
	for i in range(1000):
		random_seq = [random.randint(0,3) for i in range(length)]
		structure, energy = RNA.fold(rnalib.sequence_to_string(random_seq))

		if structure not in seq_dict:
			seq_dict[structure] = [random_seq]
		else:
			seq_dict[structure].append(random_seq)
	return seq_dict


train_dict = generate_train_set()
structure = random.choice(list(train_dict.keys()))
print(structure)
print(train_dict.keys())

policy = rnalib.RNAPolicy()

TestPolicy(policy)
train_supervised(policy)
TestPolicy(policy)