import numpy as np

class PSO:
    def __init__(self, n, dim, lower_bound, upper_bound):
        self.__population_size = n
        self.__dimension = dim
        self.__weight = 0.9
        self.__velocity = 0.1
        self.__acceleration_1 = 2
        self.__acceleration_2 = 2
        self.__max_iteration = 100

        self.__population = np.random.randint(low=lower_bound, high=upper_bound+1, size=(n,dim))
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
                if min_fitness<best_fitness:
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
        

def test(x):
    return x*x

pso = PSO(5, 1, -10, 10)
result = pso.optimal(test)
print(result)