import math as m

#print(m.log2(2))

def log2(x):
	if x==0:
		return 0
	else:
		return m.log2(x)

def entropy(a,b,c,d):
	tot = a+b+c+d
	tot1 = a+b
	tot2 = c+d
	return (tot1/tot)*((a/tot1)*log2(a/tot1) + (b/tot1)*log2(b/tot1)) + (tot2/tot)*((c/tot2)*log2(c/tot2) + (d/tot2)*log2(d/tot2))

E = -entropy(1,0,2,1)
print(E)

E = -entropy(2,1,1,0)
print(E)

E = -entropy(2,0,1,1)
print(E)
