import numpy as np
import random
from copy import copy

class PSO:
    def __init__(self, n, dim, lower_bound, upper_bound, max_iters = 500):
        self.__population_size = n
        self.__dimension = dim
        self.__weight = 0.9
        self.__velocity = 0.1
        self.__acceleration_1 = 2
        self.__acceleration_2 = 2
        self.__max_iteration = max_iters
        self.__lower = lower_bound
        self.__upper = upper_bound
        self.__velocity_max = upper_bound*2
        self.__velocity_min = lower_bound*2

        self.__population = np.random.randint(low=lower_bound, high=upper_bound+1, size=( self.__population_size, self.__dimension))
        self.__position = self.__population + (self.__population * self.__velocity)
        
    def optimal(self, function):
        old_position = self.__position
        for i in range(self.__max_iteration):
            old_position = np.clip(old_position, self.__lower, self.__upper)
            fitness = function(old_position)
            g_best_index = np.argmin(fitness, axis=0)
            min_fitness = min(fitness)
            if i==0:
                # init g_best and x_best at first interation
                g_best = old_position[g_best_index]
                x_best = old_position
                best_fitness = min_fitness
                old_fitness = fitness
                old_v = old_position*self.__velocity
            else:
                # update g_best
                if abs(min_fitness)<abs(best_fitness):
                    g_best = old_position[g_best_index]
                    best_fitness = min_fitness
                # update x_best
                for i in range(old_fitness.shape[0]):
                    if abs(fitness[i]) < abs(old_fitness[i]):
                        x_best[i] = old_position[i]
                
            # update velocity & position
            new_v = self.calculate_velocity(old_v, old_position, x_best, g_best)
            new_v = np.clip(new_v, self.__velocity_min, self.__velocity_max)
            new_position = old_position + new_v
            
            old_position = np.round(new_position, 1)
            old_v = new_v
            

        return g_best, np.round(best_fitness, 4)
    
    def calculate_velocity(self, vi, curr_x, x_best, g_best):
        return self.__weight*vi + self.__acceleration_1*np.random.rand()*(x_best-curr_x) + self.__acceleration_2*np.random.rand()*(g_best-curr_x)


def evaluation(best_each_gbest, best_each_fitness, runs):
    run_index = np.arange(float(runs)) +1
    accum_g_best = np.cumsum(best_each_gbest, axis=0)
    accum_fitness = np.cumsum(best_each_fitness)
    # average best-so-far
    iter_gbest_mean = np.round(np.divide(accum_g_best, run_index.reshape(-1, 1)), 4)
    # average fitness 
    iter_fitness_mean = np.round(accum_fitness/run_index, 4)
    # median
    g_best_median = np.median(best_each_gbest, axis=0)
    print("median:", g_best_median)
    print("mean fitness: ", np.mean(best_each_fitness))
    print("best_each_fitness: ", np.round(best_each_fitness.flatten()))
    print("best fitness: ", np.round(iter_fitness_mean))
    

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

def main():
    runs = 50
    pop = 50
    # # F2
    # print("optimize F2 function")
    # dim = 30
    # best_each_gbest = np.zeros((runs, dim)) 
    # best_each_fitness = np.zeros((runs, 1)) 
    # for i in range(runs):
    #     pso = PSO(pop, dim, -10, 10, 500)
    #     best_sofar, best_fitness = pso.optimal(F2)
    #     best_each_gbest[i] = best_sofar
    #     best_each_fitness[i] = best_fitness
    # evaluation(best_each_gbest, best_each_fitness, runs)

    # #F3
    # print("optimize F3 function")
    # dim = 30
    # best_each_gbest = np.zeros((runs, dim)) 
    # best_each_fitness = np.zeros((runs, 1)) 
    # for i in range(runs):
    #     pso = PSO(pop, dim, -100, 100, 500)
    #     best_sofar, best_fitness = pso.optimal(F3)
    #     best_each_gbest[i] = best_sofar
    #     best_each_fitness[i] = best_fitness
    # evaluation(best_each_gbest, best_each_fitness, runs)

    # #F11
    # print("optimize F11 function")
    # dim = 30
    # best_each_gbest = np.zeros((runs, dim)) 
    # best_each_fitness = np.zeros((runs, 1)) 
    # for i in range(runs):
    #     pso = PSO(pop, dim, -600, 600, 500)
    #     best_sofar, best_fitness = pso.optimal(F11)
    #     best_each_gbest[i] = best_sofar
    #     best_each_fitness[i] = best_fitness
    # evaluation(best_each_gbest, best_each_fitness, runs)

    #F12
    print("optimize F12 function")
    dim = 30
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    for i in range(runs):
        pso = PSO(pop, dim, -50, 50, 500)
        best_sofar, best_fitness = pso.optimal(F12)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
    evaluation(best_each_gbest, best_each_fitness, runs)

if __name__=="__main__":
    main()
    # te_array = np.array([[2, 3, 2],[4, 6, 1]])
    # te_array = np.array([[0, 0, 0],[0, 0, 0]])
    # print(F11(te_array))
    # print(F12(te_array, 10, 100, 4))