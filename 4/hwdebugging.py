import numpy as np

def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def hinge(v):
    return np.where(1-v < 0, 0, 1-v)


def hinge_loss(x, y, th, th0):
	L_h = np.sum(hinge(y*(np.dot(th.T,x) + th0)))
	return L_h

def svm_obj(x, y, th, th0, lam):
	d,n = x.shape
	return hinge_loss(x, y, th, th0)/n + lam*np.linalg.norm(th)**2

def d_hinge(v):
    return np.where(v<1, -1, 0)

def d_hinge_loss_th(x, y, th, th0):
    d,n = x.shape
    ans = np.zeros((d,n))
    for j in range(d):
        for i in range(n):
            ans[j][i] = np.asscalar(d_hinge(y[0,i]*(np.dot(th.T,x[:,i]) + th0))*y[0,i]*x[j,i])
    return ans

def d_hinge_loss_th0(x, y, th, th0):
	d,n = x.shape
	ans = np.zeros((1,n))
	for i in range(n):
		ans[0][i] = np.asscalar(d_hinge(y[0,i]*(np.dot(th.T,x[:,i]) + th0))*y[0,i])
	return ans

def d_svm_obj_th(x, y, th, th0, lam):
	d,n = x.shape
	return cv(np.sum(d_hinge_loss_th(x, y, th, th0)/n, axis = 1)) + 2*lam*th

def d_svm_obj_th0(x, y, th, th0, lam):
	d,n = x.shape
	return cv(np.sum(d_hinge_loss_th0(x, y, th, th0)/n, axis = 1))

def svm_obj_grad(x, y, th, th0, lam):
    return np.vstack((d_svm_obj_th(x, y, th, th0, lam), d_svm_obj_th0(x, y, th, th0, lam)))

def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    y = np.array([[1, -1, 1, -1]])
    return X, y
sep_m_separator = np.array([[ 2.69231855], [ 0.67624906]]), np.array([[-3.02402521]])

#print hinge_loss(x_1,y_1,th1,th1_0)
#print d_svm_obj_th(X2[:,0:1], y2[:,0:1], th2, th20, 0.01).tolist()
#print d_svm_obj_th(X2, y2, th2, th20, 0.01).tolist()

#print d_svm_obj_th0(X2, y2, th2, th20, 0.01).tolist()

th_tot = np.zeros((5,1))
print th_tot.T[:,4:]