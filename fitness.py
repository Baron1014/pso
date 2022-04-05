import numpy as np

def test(x):
    return x*x

def F2(x):
    # accum_sum = np.cumsum(np.absolute(x), axis=1)
    # accum_prod = np.cumprod(np.absolute(x), axis=1)
    sum_x = np.sum(np.absolute(x), axis=1)
    prod_x = np.prod(np.absolute(x), axis=1)

    return sum_x + prod_x

def F3(x):
    accum_sum = np.cumsum(np.absolute(x), axis=1)
    power_x = np.power(accum_sum, 2)
    
    return np.sum(power_x, axis=1).reshape(-1, 1)

def F11(x):
    sum_x = np.sum(np.power(x, 2), axis=1)
    dim_index = np.arange(float(x.shape[1])) +1
    prod_x = np.prod(np.divide(np.cos(x), np.sqrt(dim_index)), axis=1)
    
    return (sum_x/4000) - prod_x + 1

def F12(x, a=10, k=100, m=4):
    y = 1 + (x+1)/4
    # u
    u1 = np.where(x>a, k * np.power((x - a), m), 0)
    u2 = np.where(x<(-1*a), k * np.power((-1*(x) - a), m), 0)
    u = u1 + u2

    # f
    first = 10*(np.sin(np.pi*y[:, :1])**2)
    second_1 = np.sum(np.multiply(((y[:, :-1]-1)**2), (10*(np.sin(np.pi*y[:, 1:])**2) + 1)), axis=1)
    second_2 = (y[:, -1]-1)**2
    second = np.array(second_1+second_2).reshape(-1, 1)
    third = np.sum(u, axis=1).reshape(-1, 1)

    return np.add((np.pi/x.shape[1])*(np.add(first, second)), third)

def F13(x, a=5, k=100, m=4):
    # u
    u1 = np.where(x>a, k * np.power((x - a), m), 0)
    u2 = np.where(x<(-1*a), k * np.power((-1*(x) - a), m), 0)
    u = u1 + u2

    # F
    p_1 = np.sin(3*np.pi*x[:, :1])**2
    p_2 = np.sum(np.multiply((x[:, :-1]-1)**2, (1 + np.sin(3*np.pi*x[:, 1:])**2)), axis=1).reshape(-1, 1)
    p_3 = np.multiply((x[:, -1:]-1)**2, (1 + np.sin(2*np.pi*x[:, -1:])**2))
    p = 0.1*(np.add(np.add(p_1, p_2), p_3))

    return np.add(p, np.sum(u, axis=1).reshape(-1, 1))

def F14(x):
    # define a 
    ai = [e for _ in range(5) for e in [-32, -16, 0, 16, 32]]
    aj = [-32 + (i//5)*16 for i in range(25)]
    aij = np.array([ai, aj])

    # function
    p = np.array([1/(np.add(np.power(x[:, :1]-aij[0, j], 6), np.power(x[:, 1:]-aij[1, j], 6)) + j) for j in range(1, 25)]).reshape(-1, 2)

    return 1 / (0.002 + np.sum(p, axis=0))