experience = [(0, 'b', 0), #t = 0
              (2, 'b', 0),
              (3, 'b', 2),
              (0, 'b', 0), #t = 3
              (2, 'b', 0),
              (3, 'c', 2),
              (0, 'c', 0), #t = 6
              (1, 'b', 1),
              (0, 'b', 0),
              (2, 'c', 0), #t = 9
              (3, 'c', 2),
              (0, 'c', 0),
              (1, 'c', 1), #t = 12
              (0, 'c', 0),
              (2, 'b', 0),
              (3, 'b', 2), #t = 15
              (0, 'b', 0),
              (2, 'c', 0),
              (3, '', 0), #t = 18
              ]

actions = ['b','c']
states = [0,1,2,3]
Q = {}

for s in states:
    for a in actions:
        Q[(s,a)] = 0

n = len(experience)

#print(experience[0][1])

for i in range(n-1):
    s,a,r,s_next = experience[i][0], experience[i][1], experience[i][2], experience[i+1][0]
    Q[(s,a)] = 0.5*Q[(s,a)] + 0.5*(r + 0.9*max([Q[(s_next,b)] for b in actions]))
    print(Q[(s,a)])
