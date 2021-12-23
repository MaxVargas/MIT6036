from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: a list of inputs to feed into SM
           returns:   a list of outputs of SM'''
        n = len(input_seq)
        outs = [None]*(n)
        state  = self.start_state
        for i in range(n):
            state = self.transition_fn(state,input_seq[i])
            outs[i] = self.output_fn(state)
        return outs


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = [0,0] # (current_digit,carry)

    def transition_fn(self, s, x):
        # Your code here
        s[0] = (x[0]+x[1]+s[1])%2
        s[1] = (x[0]+x[1]+s[1])//2
        return s

    def output_fn(self, s):
        # Your code here
        return s[0]


class Reverser(SM):
    start_state = [[],1]

    def transition_fn(self, s, x):
        # Your code here
        if s[1]==1:
            if x=='end':
                s[1]=0
            else:
                s[0].append(x)
        else:
            s[0] = s[0][:-1]
        return s

    def output_fn(self, s):
        # Your code here
        #print(s)
        if s[1]==1 or (s[0]==[] and s[1]==0):
            return None
        return s[0][-1]

class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        self.Wsx, self.Wss, self.Wo = Wsx, Wss, Wo
        self.Wss_0, self.Wo_0 = Wss_0, Wo_0
        self.f1, self.f2 = f1, f2

        d,n = np.shape(self.Wsx)

        self.start_state = np.zeros((d,1))

    def transition_fn(self, s, x):
        # Your code here
        return self.f1(np.dot(self.Wss,s) + np.dot(self.Wsx,x) + self.Wss_0)

    def output_fn(self, s):
        # Your code here
        return self.f2(np.dot(self.Wo,s) + self.Wo_0)

Wsx = np.array([[1],[0],[0]])   # Your code here
Wss = np.array([[0,0,0],[1,0,0],[0,1,0]])   # Your code here
Wo = np.array([[1,-2,3]])     # Your code here
Wss_0 = np.array([[0],[0],[0]])  # Your code here
Wo_0 = np.array([[0]])  # Your code here
f1 = lambda x: x # Your code here
f2 = lambda x: x # Your code here
auto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)

#print(auto.transduce([5,5,-5,0,25]))


A = np.matrix([[-1,0.09,0.81,0], [0.81,-0.91,0,0], [0,0,-0.91,0.81], [0.81,0,0,-0.91]])
b = np.matrix([[0], [-1], [0], [-2]])
v = np.linalg.solve(A,b)
print(v)

