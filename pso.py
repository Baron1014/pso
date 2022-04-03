import numpy as np

class PSO:
    def __init__(self, n, dim, lower_bound, upper_bound, max_iters = 500):
        self.__population_size = n
        self.__dimension = dim
        self.__weight = 0.9
        self.__velocity = 0.1
        self.__acceleration_1 = 2
        self.__acceleration_2 = 2
        self.__max_iteration = max_iters

        self.__population = np.random.randint(low=lower_bound, high=upper_bound+1, size=( self.__population_size, self.__dimension))
        self.__position = self.__population + (self.__population * self.__velocity)
        
    def optimal(self, function):
        old_position = self.__position
        for i in range(self.__max_iteration):
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
                    if fitness[i] < old_fitness[i]:
                        x_best[i] = old_position[i]
                
            # update velocity & position
            new_v = self.calculate_velocity(old_v, old_position, x_best, g_best)
            new_position = old_position + new_v
            
            old_position = new_position
            old_v = new_v
            

        return np.round(g_best, 4), np.round(best_fitness, 4)
    
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
    print("best fitness: ", np.round(iter_fitness_mean))
    print("mean fitness: ", np.mean(best_each_fitness))

def test(x):
    return x*x

def F2(x):
    # accum_sum = np.cumsum(np.absolute(x), axis=1)
    # accum_prod = np.cumprod(np.absolute(x), axis=1)
    sum_x = np.sum(np.absolute(x), axis=1)
    prod_x = np.prod(np.absolute(x), axis=1)

    return sum_x + prod_x

def main():
    runs = 50
    dim = 30
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    for i in range(runs):
        pso = PSO(50, dim, -10, 10, 500)
        best_sofar, best_fitness = pso.optimal(F2)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
    evaluation(best_each_gbest, best_each_fitness, runs)

if __name__=="__main__":
    main()
    # te_array = np.array([[2, 3, 2],[4, 6, 1]])
    # print(F2(te_array))