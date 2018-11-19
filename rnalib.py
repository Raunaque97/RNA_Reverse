import numpy as np
import tensorflow as tf
import RNA
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# This file defines the classes for the environment and the policy network, as well
# as various useful functions.

width = 80 # Number of channels for internal layers
pairs = {0:(3,), 1:(2,), 2:(1,3), 3:(0,2)} # Which other types of bases each type can pair with

def bracket_to_bonds(structure):
    """Given a structure in dot bracket notation, compute the list of bonds."""
    bonds = [None]*len(structure)
    opening = []
    for i,c in enumerate(structure):
        if c == '(':
            opening.append(i)
        elif c == ')':
            j = opening.pop()
            bonds[i] = j
            bonds[j] = i
    return bonds

def sequence_to_string(sequence):
    """Convert a one hot encoded sequence to a string."""
    bases = ['A', 'C', 'G', 'U']
    return ''.join(bases[i] for i in sequence)

def sequence_to_bracket(sequence):
    """Compute the native structure (in dot bracket notation) of a one hot encoded sequence."""
    structure, energy = RNA.fold(sequence_to_string(sequence))
    return structure

class RNAEnvironment():
    """This class implements the environment our agent will interact with."""

    def __init__(self, goal, max_steps):
        """Create a new RNAEnvironment.

        Parameters
        ----------
        puzzles: list
            the list of training puzzles.  A random one is selected for each episode.
        max_steps: int
            the maximum number of steps for any episode.  If the puzzle has not been
            solved after this many steps, it will give up.  Pass -1 to not set a limit.
        """
        self.max_steps = max_steps
        self.length = len(goal)
        self.count = 0
        self.terminated = False

                
        self.goal = goal                                 # bracket struct
        self.target_bonds = bracket_to_bonds(self.goal)
        self.state = None
        self.sequence = [random.randint(0, 3) for i in range(self.length)]      # [0-3]*
        while True:
            self.sequence = [random.randint(0, 3) for i in range(self.length)]
            self.update_state()
            if not self.terminated:
                break
        RNAEnvironment.update_state(self)

    def step(self, action): # action tuple
        """Perform one action on the environment."""
        index, base = action
        self.count += 1
        reward = -1
        if self.count == self.max_steps:
            self.terminated = True # Give up.
            return reward
        if self.sequence[index] == base:
            # This action doesn't change anything.
            return reward
        elif not self.terminated:
            self.sequence[index] = base
            pair_index = self.target_bonds[index]
            if pair_index is not None:
                if self.sequence[pair_index] not in pairs[base]:
                    self.sequence[pair_index] = pairs[base][0]
            self.update_state()
            reward = 1 if self.terminated else -1 #########
        return reward

    def reset(self):
        """Reset the environment and begin a new episode."""
        self.count = 0
        self.terminated = False
        while True:
            self.sequence = [random.randint(0, 3) for i in range(self.length)]
            self.update_state()
            if not self.terminated:
                break

    def update_state(self):
        """Update the state vectors encoding the current sequence and list of bonds."""
        bracket = sequence_to_bracket(self.sequence)
        self.terminated = (bracket == self.goal)
        
        current_bonds = bracket_to_bonds(bracket)

        # Compute the state from current_bonds, self.target_bonds & self.sequence
        
        state = np.zeros((self.length, 7))
        state[np.arange(self.length), self.sequence] = 1

        for i in range(self.length):
            if current_bonds[i] is not None and self.target_bonds[i] is not None:
                state[i,4] = (current_bonds[i] - self.target_bonds[i])/self.length
                state[i,5] = 0
                state[i,6] = 0
            elif current_bonds[i] is not None and self.target_bonds[i] is None:
                state[i,4] = 1
                state[i,5] = 0
                state[i,6] = 1
            elif current_bonds[i] is None and self.target_bonds[i] is not None:
                state[i,4] = 1
                state[i,5] = 1
                state[i,6] = 0
            elif current_bonds[i] is None and self.target_bonds[i] is None:
                state[i,4] = 0
                state[i,5] = 1
                state[i,6] = 1

        self.state = state

class RNAPolicy(nn.Module):
    """This class implements the policy network."""

    def __init__(self):
        super(RNAPolicy, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=5, kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=5,stride=1,padding=2)
        self.conv3 = nn.Conv1d(in_channels=10, out_channels=4, kernel_size=5,stride=1,padding=2)

    def get_action(self, state):
        x = RNAPolicy.forward(self, state)
        x=x.detach().numpy().reshape(-1,4)
        return x
        # return RNAPolicy.get_action(self, state)

    def forward(self, state):
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float()
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        sigmoid = nn.Sigmoid()
        x = sigmoid(self.conv3(x))
        return x

def update(S,A,R,SPrime,APrime, model, environment):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  
    optimizer.zero_grad()
    
    output = model(np.expand_dims(environment.state.T, axis=0))
    # convert best_action to torch type
    best_action = torch.from_numpy(best_action.T).float()
    
    loss = ((output - best_action)**2).mean()
    
    loss.backward()
    optimizer.step()

def updateParam(batch, model, lr = 0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    optimizer.zero_grad()
    N = batch[0][0].shape[0] #length of RNA
    states = np.zeros((len(batch),7,N))
    for i in range(len(batch)):
        states[i] = batch[i][0].T

    Q = model(states)
    loss = torch.FloatTensor([0])
    for i in range(len(batch)):
        (S,a,Y,Sprime) = batch[i]
        loss = loss + ((torch.FloatTensor([Y]) - Q[i, a[1], a[0]])**2)
        # loss += Q[i, a[1], a[0]]

    loss = loss/len(batch)

    loss.backward()
    optimizer.step()

    return loss.detach().numpy()[0]