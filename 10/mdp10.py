import pdb
import random
import numpy as np
from dist import uniform_dist, delta_dist, mixture_dist, DDist
from util import argmax_with_val, argmax
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import Adam

class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn,
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

    def state2vec(self, s):
        '''
        Return one-hot encoding of state s; used in neural network agent implementations
        '''
        v = np.zeros((1, len(self.states)))
        v[0,self.states.index(s)] = 1.
        return v

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

def value_iteration(mdp, q, eps = 0.01, max_iters = 1000):
    # Your code here (COPY FROM HW9)
    for s in q.states:
        for a in q.actions:
            q.set(s,a,0)
    
    for i in range(max_iters):
        q_new = q.copy()
        for s in q.states:
            for a in q.actions:
                v = mdp.reward_fn(s,a)
                for t in q.states:
                    v += mdp.discount_factor*mdp.transition_model(s,a).prob(t)*max([q.get(t,b) for b in q.actions])
                q_new.set(s,a,v)
        
        p = True
        for s in q.states:
            hi = max([q_new.get(s,a)-q.get(s,a) for a in q.actions])
            lo = min([q_new.get(s,a)-q.get(s,a) for a in q.actions])
            if hi >= eps or lo <= -eps:
                p = False
                q = q_new
                break
        if p==True:
            return q_new
    return q_new

# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    # Your code here (COPY FROM HW9)
    return max([q.get(s,a) for a in q.actions])

# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    # Your code here (COPY FROM HW9)
    q_star, a_star = q.get(s,q.actions[0]), q.actions[0]
    for a in q.actions:
        if q.get(s,a) > q_star:
            q_star = q.get(s,a)
            a_star = a
    return a_star

def epsilon_greedy(q, s, eps = 0.5):
    if random.random() < eps:  # True with prob eps, random action
        return uniform_dist(q.actions).draw()
    else:
        return greedy(q,s)

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]
    def update(self, data, lr):
        # Your code here
        for d in data:
            self.set(d[0], d[1], (1-lr)*self.get(d[0],d[1]) + lr*d[2])

def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5, interactive_fn=None):
    # Your code here
    s = mdp.init_state()
    for i in range(iters):
        # include this line in the iteration, where i is the iteration number
        if interactive_fn: interactive_fn(q, i)
        a = epsilon_greedy(q,s,eps)
        r, s_prime = mdp.sim_transition(s,a)
        
        if not mdp.terminal(s):
            r += mdp.discount_factor*value(q,s_prime)
        q.update([(s,a,r)],lr)
        s = s_prime
        print(s)
    return q


def tinyTerminal(s):
    return s==4
def tinyR(s, a):
    if s == 1: return 1
    elif s == 3: return 2
    else: return 0
def tinyTrans(s, a):
    if s == 0:
        if a == 'a':
            return DDist({1 : 0.9, 2 : 0.1})
        else:
            return DDist({1 : 0.1, 2 : 0.9})
    elif s == 1:
        return DDist({1 : 0.1, 0 : 0.9})
    elif s == 2:
        return DDist({2 : 0.1, 3 : 0.9})
    elif s == 3:
        return DDist({3 : 0.1, 0 : 0.5, 4 : 0.4})
    elif s == 4:
        return DDist({4 : 1.0})

'''
def testQ():
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = TabularQ(tiny.states, tiny.actions)
    qf = Q_learn(tiny, q)
    return list(qf.q.items())

random.seed(0)
print(testQ())
'''

# Simulate an episode (sequence of transitions) of at most
# episode_length, using policy function to select actions.  If we find
# a terminal state, end the episode.  Return accumulated reward a list
# of (s, a, r, s') where s' is None for transition from terminal state.
# Also return an animation if draw=True.
def sim_episode(mdp, episode_length, policy, draw=False):
    episode = []
    reward = 0
    s = mdp.init_state()
    all_states = [s]
    for i in range(int(episode_length)):
        a = policy(s)
        (r, s_prime) = mdp.sim_transition(s, a)
        reward += r
        if mdp.terminal(s):
            episode.append((s, a, r, None))
            break
        episode.append((s, a, r, s_prime))
        if draw: 
            mdp.draw_state(s)
        s = s_prime
        all_states.append(s)
    animation = animate(all_states, mdp.n, episode_length) if draw else None
    return reward, episode, animation

# Create a matplotlib animation from all states of the MDP that
# can be played both in colab and in local versions.
def animate(states, n, ep_length):
    try:
        from matplotlib import animation, rc
        import matplotlib.pyplot as plt
        from google.colab import widgets

        plt.ion()
        plt.figure(facecolor="white")
        fig, ax = plt.subplots()
        plt.close()

        def animate(i):
            if states[i % len(states)] == None or states[i % len(states)] == 'over':
                return
            ((br, bc), (brv, bcv), pp, pv) = states[i % len(states)]
            im = np.zeros((n, n+1))
            im[br, bc] = -1
            im[pp, n] = 1
            ax.cla()
            ims = ax.imshow(im, interpolation = 'none',
                        cmap = 'viridis', 
                        extent = [-0.5, n+0.5,
                                    -0.5, n-0.5],
                        animated = True)
            ims.set_clim(-1, 1)
        rc('animation', html='jshtml')
        anim = animation.FuncAnimation(fig, animate, frames=ep_length, interval=100)
        return anim
    except:
        # we are not in colab, so the typical animation should work
        return None

# Return average reward for n_episodes of length episode_length
# while following policy (a function of state) to choose actions.
def evaluate(mdp, n_episodes, episode_length, policy):
    score = 0
    length = 0
    for i in range(n_episodes):
        # Accumulate the episode rewards
        r, e, _ = sim_episode(mdp, episode_length, policy)
        score += r
        length += len(e)
        # print('    ', r, len(e))
    return score/n_episodes, length/n_episodes

def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,
                  episode_length=10, n_episodes=2,
                  interactive_fn=None):
    all_exps = []
    for i in range(iters):
        # include this line in the iteration, where i is the iteration number
        if interactive_fn: interactive_fn(q, i)
        for j in range(n_episodes):
            all_exps = all_exps + sim_episode(mdp, episode_length, lambda x: epsilon_greedy(q,x))[1]
        all_q_targets = []
        for e in all_exps:
            s,a,r,s_prime = e
            if s_prime == None:
                all_q_targets.append((s,a,r))
            else:
                all_q_targets.append((s,a,r+mdp.discount_factor*value(q,s_prime)))
        q.update(all_q_targets, lr)
    return q

'''
def testBatchQ():
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = TabularQ(tiny.states, tiny.actions)
    qf = Q_learn_batch(tiny, q)
    return list(qf.q.items())

random.seed(0)
print(testBatchQ())
'''

def make_nn(state_dim, num_hidden_layers, num_units):
    model = Sequential()
    model.add(Dense(num_units, input_dim = state_dim, activation='relu'))
    for i in range(num_hidden_layers-1):
        model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model

class NNQ:
    def __init__(self, states, actions, state2vec, num_layers, num_units, epochs=1):
        self.actions = actions
        self.states = states
        self.epochs = epochs
        self.state2vec = state2vec
        self.models = []              # Your code here
        for a in self.actions:
            self.models.append(make_nn(len(self.states), num_layers, num_units))
            
    def get(self, s, a):
        i = self.actions.index(a)
        return self.models[i].predict(self.state2vec(s))
        
    def to_data(self, data):
        X = [np.zeros((0,len(self.states)))]*(len(self.actions))
        Y = [np.zeros((0,1))]*(len(self.actions))
        for d in data:
            i = self.actions.index(d[1])
            X[i] = np.concatenate((X[i],self.state2vec(d[0])),axis=0)
            Y[i] = np.concatenate((Y[i],np.array([[d[2]]])),axis=0)
        return X,Y
        
    def update(self, data, lr, epochs=1):
        # Your code here
        X, Y = self.to_data(data)
        for i in range(len(self.models)):
            if np.shape(X[i]) != (0,len(self.states)):
                self.models[i].fit(X[i],Y[i],epochs=epochs)

#a = np.zeros((0,5))
#b = np.concatenate((a,np.array([[1,2,3,4,5]])),axis=0)
#print(b)
'''
def test_NNQ(data):
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = NNQ(tiny.states, tiny.actions, tiny.state2vec, 2, 10)
    q.update(data, 1)
    return [q.get(s,a) for s in q.states for a in q.actions]
'''

#print(test_NNQ([(0,'a',0.3),(1,'a',0.1),(0,'a',0.1),(1,'a',0.5)]))
