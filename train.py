import numpy as np
import rnalib as rnalib
import random
import RNA
import sys

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
        objectiveSeq = train_dict[structure][0]             # TODO select randomly insr=tead of 1st 
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

def train_DQN(policy):
    numEpisodes = 100
    epsilon = 0.2
    miniBatchSz = 128
    alpha = 0.05
    gamma = 1.0
    policy.train()

    for e in range(numEpisodes):
        structure = random.choice(list(train_dict.keys()))
        N = len(structure)
        env = rnalib.RNAEnvironment(goal = structure, max_steps = 10000)

        exp_replay = []

        while len(exp_replay) <= 1000:
            S = env.state
            a = epsilonGreedyActionDQN(S, policy, epsilon= epsilon) #actionTuple 
            while not env.terminated:
                r = env.step(a)
                Sprime = env.state
                aprime = epsilonGreedyActionDQN(Sprime, policy, epsilon= 1.0)
                Y = r if env.terminated else r + gamma*np.max(policy.get_action(np.expand_dims(Sprime.T, axis=0)))
                
                if r == 1:
                    # print('reward = 1 on DQNtrain')
                    exp_replay.append((S,a,Y,None))
                    break
                exp_replay.append((S,a,Y,Sprime))

                S = Sprime
                a = aprime
            env.reset()
            # print('##################### env reset exp_replay len = ' + str(len(exp_replay)))
        # if True:
        #     for (_,a,y,Sprime) in exp_replay[:10]:
        #         print(a,y,(Sprime is None))
        #     print('---------')
        #     print(policy.get_action(np.expand_dims(exp_replay[0][0].T, axis=0)))
        #     # exit()


        print('training from exp_replay len = '+str(len(exp_replay))+' \tEpisode: '+str(e))
        losses = []
        # for p in policy.parameters():
        #     if p.requires_grad:
        #         print(p.name, p.data)
        #         break
        for j in range(2000):
            miniBatch = random.sample(exp_replay, miniBatchSz)
            losses.append(rnalib.updateParam(miniBatch,model = policy, lr = alpha))


            if j%250 == 0:
                print(losses[-1])
        # print(losses[0:10000:50])
        
        # for p in policy.parameters():
        #     if p.requires_grad:
        #         print(p.name, p.data)
        #         break   



def TestPolicy(policy):
    structure = random.choice(list(train_dict.keys()))
    objectiveSeq = train_dict[structure][0]
    N = len(objectiveSeq)
    env = rnalib.RNAEnvironment(goal = structure, max_steps = 50)
    # optimum_actions = getBestActions(env.sequence, objectiveSeq)
    # print(env.terminated)
    print('objective structure: '+structure)
    while not env.terminated:
        a = array2actionTuple(policy.get_action(np.expand_dims(env.state.T, axis=0)))
        prevState = env.state
        r = env.step(a)
        print('step: '+str(env.count)+'\tseq = '+str(env.sequence)+'\t policy Actn = '+str(a)+'\t objSeQ: '+str(objectiveSeq))
        if r == 1:
            print('***SUCCESS***')
            break
        flag = 0
        while (env.state == prevState).all() and not env.terminated:
            a = epsilonGreedyActionDQN(env.state, policy, epsilon = 1.0)
            r = env.step(a)
            flag = 1
        if flag == 1:
            print('step: '+str(env.count)+'\tseq = '+str(env.sequence)+'\t RANDOM Actn = '+str(a)+'\t objSeQ: '+str(objectiveSeq))
        if r == 1:
            print('***SUCCESS***')

def TestRandomPolicy():
    structure = random.choice(list(train_dict.keys()))
    objectiveSeq = train_dict[structure][0]
    N = len(objectiveSeq)
    env = rnalib.RNAEnvironment(goal = structure, max_steps = 10000)

    steps = 0
    while not env.terminated:
        a = (random.randint(0,N-1),random.randint(0,3))
        r = env.step(a)
        steps += 1

    return steps


def TestTabularPolicy(policy):
    structure = random.choice(list(train_dict.keys()))
    objectiveSeq = train_dict[structure][0]
    N = len(objectiveSeq)
    env = rnalib.RNAEnvironment(goal = structure, max_steps = 100)
    # optimum_actions = getBestActions(env.sequence, objectiveSeq)
    # print(env.terminated)
    print('objective structure: '+structure)
    while not env.terminated:
        a = epsilonGreedyAction(env.state, policy, epsilon = 0.0)
        prevState = env.state
        r = env.step(a)
        print('step: '+str(env.count)+'\tseq = '+str(env.sequence)+'\t policy Actn = '+str(a)+'\t objSeQ: '+str(objectiveSeq))
        if r == 1:
            print('Success')
        flag = 0
        while (env.state == prevState).all() and not env.terminated:
            a = epsilonGreedyAction(env.state, policy, epsilon = 1.0)
            r = env.step(a)
            flag = 1
        if flag == 1:
            print('step: '+str(env.count)+'\tseq = '+str(env.sequence)+'\t RANDOM Actn = '+str(a)+'\t objSeQ: '+str(objectiveSeq))
        if r == 1:
            print('***SUCCESS***')
def generate_train_set():
    seq_dict = {}
    length = 9
    for i in range(500):
        random_seq = [random.randint(0,3) for i in range(length)]
        structure, energy = RNA.fold(rnalib.sequence_to_string(random_seq))

        if structure not in seq_dict:
            seq_dict[structure] = [random_seq]
        else:
            seq_dict[structure].append(random_seq)
    return seq_dict

def list_to_tuple(l):
    if type(l) == np.ndarray:
        l = l.tolist()
    t = ()
    if type(l) == list:
        if type(l[0]) != list:
            return tuple(l)
        for i in l:
            t = t + (list_to_tuple(i),)
        return t

def epsilonGreedyActionDQN(st, policy, epsilon= 0.5):
    N = st.shape[0]
    if random.random() < epsilon:
        return (random.randint(0,N-1),random.randint(0,3))
    else:
        # find best action
        return array2actionTuple(policy.get_action(np.expand_dims(st.T, axis=0)))


def epsilonGreedyAction(st, policy, epsilon = 0.1):
    all_actns = []
    N = st.shape[0]
    for i in range(N):
        for b in range(4):
            all_actns.append((i,b))

    for a in all_actns:
        if (list_to_tuple(st),a) not in policy:
            policy[(list_to_tuple(st),a)] = random.random() - 0.5 #[-.5,+.5]

    if random.random() < epsilon:
        return (random.randint(0,N-1),random.randint(0,3))
    else:
        # find best action
        best_act = None
        maxV = -float('inf')
        for a in all_actns:
            if policy[(list_to_tuple(st),a)] > maxV:
                best_act, maxV = a, policy[(list_to_tuple(st),a)]
        return best_act



train_dict = generate_train_set()
structure = random.choice(list(train_dict.keys()))
print(structure)
print(train_dict.keys())

# policy = rnalib.RNAPolicy()
# TestPolicy(policy)
# train_DQN(policy)
# TestPolicy(policy)
# TestPolicy(policy)
# TestPolicy(policy)

TestRandomPolicy()

##########################################
# policy = {}

# alpha = 0.5
# gamma = 1.0
# numEpisodes = 500
# maxPolicySize = 10*1024*1024 # in bytes

# for e in range(numEpisodes):
#   structure = random.choice(list(train_dict.keys()))
#   N = len(structure)
#   env = rnalib.RNAEnvironment(goal = structure, max_steps = 10000)

#   exp_replay = []
#   S = env.state
#   a = epsilonGreedyAction(S, policy, epsilon= 0.6)  
#   while not env.terminated:
#       r = env.step(a)
#       # if r == 1:
#       #   print('reward = 1 on  train')
#       Sprime = env.state
#       aprime = epsilonGreedyAction(Sprime, policy, epsilon= 0.6)

#       exp_replay.append((S,a,r,Sprime,aprime))
#       S = Sprime
#       a = aprime

#   for j in range(10):
#       for (S,a,r,Sprime,aprime) in exp_replay:
#           policy[(list_to_tuple(S),a)] = policy[(list_to_tuple(S),a)] + alpha*(r + gamma*policy[(list_to_tuple(Sprime),aprime)] - policy[(list_to_tuple(S),a)])

#   print(str(e)+'\tpolicy Size: '+str(sys.getsizeof(policy)/(1024*1024))+' Mb')
#   if sys.getsizeof(policy) > maxPolicySize:
#       break
# # print(policy)
# TestTabularPolicy(policy)
# TestTabularPolicy(policy)
# TestTabularPolicy(policy)
##########################################




