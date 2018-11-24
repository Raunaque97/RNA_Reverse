import numpy as np
import rnalib as rnalib
import random
import RNA
import sys
import os
import torch
import pickle as pkl

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

    base_list = []
    for i in range(a.shape[0]):
        for b in range(4):
            if a[i,b] == a[idx,base]:
                base_list.append((i,b))

    return random.choice(base_list)

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
    numEpisodes = 50
    epsilon = 0.3
    miniBatchSz = 50
    alpha = 0.3
    gamma = 1.0
    frac = 0.3
    policy.train()

    for e in range(numEpisodes):
        structure = random.choice(list(train_dict.keys()))
        N = len(structure)
        env = rnalib.RNAEnvironment(goal = structure, max_steps = 10000)

        exp_replay = []
        print('STRUCTURE: '+structure)
        num_success = 0
        while len(exp_replay) <= 500 or num_success <= miniBatchSz*frac:
            S = env.state
            a = epsilonGreedyActionDQN(S, policy, epsilon= epsilon) #actionTuple 
            while not env.terminated:
                r = env.step(a)
                Sprime = env.state
                aprime = epsilonGreedyActionDQN(Sprime, policy, epsilon= epsilon)
                Y = r if env.terminated else r + gamma*np.max(policy.get_action(np.expand_dims(Sprime.T, axis=0)))
                
                if env.terminated:
                    # print('reward = 1 on DQNtrain')
                    exp_replay.append((S,a,Y,None))
                    num_success += 1
                    break
                exp_replay.append((S,a,Y,Sprime))

                S = Sprime
                a = aprime
            env.reset()
            # print('##################### env reset exp_replay len = ' + str(len(exp_replay))+' num_success'+str(num_success))
        # if True:
        #     print('---------')
        #     for (_,a,y,Sprime) in exp_replay[:5]:
        #         print(a,y,(Sprime is None))
        #     tempS = None
        #     for (tempS,a,y,Sprime) in [e for e in exp_replay if e[-1] is None][:5]:
        #         print(a,y,(Sprime is None))
        #     print(policy.get_action(np.expand_dims(tempS.T, axis=0)))
        #     print(policy.get_action(np.expand_dims(random.choice(exp_replay)[0].T, axis=0)))

        #     # exit()
        print('training from exp_replay len = '+str(len(exp_replay))+' \tEpisode: '+str(e))
        losses = []
        for j in range(1000):
            ## force x% of minibatch to be terminal states
            miniBatch = random.sample(exp_replay, int(miniBatchSz*(1-frac)))
            miniBatch += random.sample([e for e in exp_replay if e[-1] is None], int(miniBatchSz*frac))
            # print(miniBatch)
            losses.append(rnalib.updateParam(miniBatch,model = policy, lr = alpha))
            if j%200 == 0:
                print(losses[-1])
        # print(losses[0:10000:50])
        
        # for p in policy.parameters():
        #     if p.requires_grad:
        #         print(p.name, p.data)
        #         break   

def train_DynaQ(alpha = 0.5, gamma = 1.0, numEpisodes = 500, maxPolicySize = 10*1024*1024):
    policy = {}
    for e in range(numEpisodes):
      structure = random.choice(list(train_dict.keys()))
      N = len(structure)
      env = rnalib.RNAEnvironment(goal = structure, max_steps = 10000)
      exp_replay = []
      S = env.state
      a = epsilonGreedyAction(S, policy, epsilon= 0.6)  
      while not env.terminated:
          r = env.step(a)
          Sprime = env.state
          aprime = epsilonGreedyAction(Sprime, policy, epsilon= 0.6)
          exp_replay.append((S,a,r,Sprime,aprime))
          S = Sprime
          a = aprime
      for j in range(100):
          for (S,a,r,Sprime,aprime) in random.sample(exp_replay, len(exp_replay)):
              policy[(list_to_tuple(S),a)] = policy[(list_to_tuple(S),a)] + alpha*(r + gamma*policy[(list_to_tuple(Sprime),aprime)] - policy[(list_to_tuple(S),a)])

      print(str(e)+'\tpolicy Size: '+str(sys.getsizeof(policy)/(1024*1024))+' Mb')
      if sys.getsizeof(policy) > maxPolicySize:
          break
    # print(policy)
    # s = []
    # testStrucks =  random.sample(list(train_dict.keys()),min(100, len(train_dict.keys())))

    # def findRNA(structure, policy, max_steps = 1000):
    #     env = rnalib.RNAEnvironment(goal = structure, max_steps = max_steps)
    #     while not env.terminated:
    #         a = epsilonGreedyAction(env.state, policy, epsilon = 0.0)
    #         prevState = env.state
    #         r = env.step(a)
    #         if r == 5:
    #             return 1
    #         while (env.state == prevState).all() and not env.terminated:
    #             a = epsilonGreedyAction(env.state, policy, epsilon = 1.0)
    #             r = env.step(a)
    #         if r == 5:
    #             return 1
    #     return 0 
    # count = 0
    # for i in range(100):
    #     print(i,count) 
    #     st = testStrucks[i%len(testStrucks)]
    #     count +=findRNA(st, policy)
    # print('************')
    # print(count)
    return policy
    
def TestPolicy(policy, max_steps = 500, showStat = True):
    structure = random.choice(list(train_dict.keys()))
    objectiveSeq = train_dict[structure][0]
    N = len(objectiveSeq)
    env = rnalib.RNAEnvironment(goal = structure, max_steps = max_steps)
    if showStat:
        print('objective structure: '+structure)
    while not env.terminated:
        a = array2actionTuple(policy.get_action(np.expand_dims(env.state.T, axis=0)))
        prevState = env.state
        r = env.step(a)
        if showStat:
            print('step: '+str(env.count)+'\tseq = '+str(env.sequence)+'\t policy Actn = '+str(a)+'\t objSeQ: '+str(objectiveSeq))
        if r == 5:
            if showStat:
                print('***SUCCESS***')
            return env.count
            break
        flag = 0
        while (env.state == prevState).all() and not env.terminated:
            a = epsilonGreedyActionDQN(env.state, policy, epsilon = 1.0)
            r = env.step(a)
            flag = 1
        if flag == 1:
            if showStat:
                print('step: '+str(env.count)+'\tseq = '+str(env.sequence)+'\t RANDOM Actn = '+str(a)+'\t objSeQ: '+str(objectiveSeq))
        if r == 5:
            if showStat:
                print('***SUCCESS***')
            return env.count
    return -1
def TestRandomPolicy(max_steps = 500, showStat = True):
    structure = random.choice(list(train_dict.keys()))
    objectiveSeq = train_dict[structure][0]
    N = len(objectiveSeq)
    env = rnalib.RNAEnvironment(goal = structure, max_steps = max_steps)

    max_step = 10e4
    steps = 0
    while not env.terminated and steps < max_step:
        a = (random.randint(0,N-1),random.randint(0,3))
        r = env.step(a)
        steps += 1

    if steps == max_steps:
        return -1
    return steps
def TestTabularPolicy(policy, max_steps = 500, showStat = True):
    structure = random.choice(list(train_dict.keys()))
    objectiveSeq = train_dict[structure][0]
    N = len(objectiveSeq)
    env = rnalib.RNAEnvironment(goal = structure, max_steps = 100)
    # optimum_actions = getBestActions(env.sequence, objectiveSeq)
    # print(env.terminated)
    if showStat:    
        print('objective structure: '+structure)
    while not env.terminated:
        a = epsilonGreedyAction(env.state, policy, epsilon = 0.0)
        prevState = env.state
        r = env.step(a)
        if showStat:    
            print('step: '+str(env.count)+'\tseq = '+str(env.sequence)+'\t policy Actn = '+str(a)+'\t objSeQ: '+str(objectiveSeq))
        if r == 5:
            if showStat:    
                print('Success')
        flag = 0
        while (env.state == prevState).all() and not env.terminated:
            a = epsilonGreedyAction(env.state, policy, epsilon = 1.0)
            r = env.step(a)
            flag = 1
        if flag == 1:
            if showStat:    
                print('step: '+str(env.count)+'\tseq = '+str(env.sequence)+'\t RANDOM Actn = '+str(a)+'\t objSeQ: '+str(objectiveSeq))
        if r == 5:
            if showStat:    
                print('***SUCCESS***')
            return env.count
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



##################################################################################
##################################################################################

train_dict = generate_train_set(20)
structure = random.choice(list(train_dict.keys()))
# print(train_dict.keys())
# print(structure)

ch = sys.argv[1]
if ch == 'dyna':
    policy = train_DynaQ()
    TestTabularPolicy(policy)

if ch == 'cnn':
    policy = rnalib.RNA_CNN_Policy()
    train_DQN(policy)
    torch.save(policy, 'DQN_policy')
    policy = torch.load('DQN_policy')
    # TestPolicy(policy, max_steps = 50)
    TestPolicy(policy, max_steps = 500)

if ch == 'lstm':
    policy = rnalib.RNA_BiLSTM_Policy(hidden_size= 15, num_layers= 4)
    train_DQN(policy)
    torch.save(policy, 'DQN_policy')
    policy = torch.load('DQN_policy')
    # TestPolicy(policy, max_steps = 50)
    TestPolicy(policy, max_steps = 500)

else:
    print('Invalid argument, possible cnn/lstm/dyna')
# ss = []
# for i in range(50):
#     print(i)
#     ss.append(TestPolicy(policy, max_steps = 10000, showStat = False))
# print(ss)
