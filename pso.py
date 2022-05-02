import numpy as np
import fitness as fit
import matplotlib.pyplot as plt

class PSO:
    def __init__(self, n, dim, lower_bound, upper_bound, max_iters = 500, GL=False):
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

        if GL:
           self.__population = np.random.uniform(low=lower_bound, high=upper_bound, size=(self.__population_size, 1))
           y = np.random.uniform(-1, 1, size=(self.__population.shape))
           self.__population = np.concatenate([self.__population, y], axis=1)
        #    print("max:{}, min:{}".format(np.max(self.__population, axis=0), np.min(self.__population, axis=0)))
        else: 
            self.__population = np.random.randint(low=lower_bound, high=upper_bound+1, size=( self.__population_size, self.__dimension))
        self.__position = self.__population + (self.__population * self.__velocity)
        
    def optimal(self, function):
        old_position = self.__position
        best_fitness_array = np.zeros((1, self.__max_iteration))
        for iter in range(self.__max_iteration):
            old_position = np.clip(old_position, self.__lower, self.__upper)
            fitness = function(old_position)
            g_best_index = np.argmin(fitness, axis=0)
            min_fitness = min(fitness)
            if iter==0:
                # init g_best and x_best at first interation
                g_best = old_position[g_best_index]
                x_best = old_position
                best_fitness = min_fitness
                old_fitness = fitness
                old_v = old_position*self.__velocity
            else:
                # update g_best
                if -1 <= min_fitness < best_fitness:
                    g_best = old_position[g_best_index]
                    best_fitness = min_fitness
                # update x_best
                for i in range(old_fitness.shape[0]):
                    if fitness[i] < old_fitness[i]:
                        x_best[i] = old_position[i]
                
            # update velocity & position
            new_v = self.calculate_velocity(old_v, old_position, x_best, g_best)
            new_v = np.clip(new_v, self.__velocity_min, self.__velocity_max)
            new_position = old_position + new_v
            # best-so-far solution
            best_fitness_array[0, iter] = best_fitness
            
            # old_position = np.round(new_position, 1)
            old_position = new_position
            old_v = new_v
            

        return g_best, np.round(best_fitness, 4), best_fitness_array
    
    def calculate_velocity(self, vi, curr_x, x_best, g_best):
        return self.__weight*vi + self.__acceleration_1*np.random.rand()*(x_best-curr_x) + self.__acceleration_2*np.random.rand()*(g_best-curr_x)


def evaluation(best_each_gbest, best_each_fitness, each_best_so_far, runs, save_name):
    # run_index = np.arange(float(runs)) +1
    # accum_g_best = np.cumsum(best_each_gbest, axis=0)
    # accum_fitness = np.cumsum(best_each_fitness)
    # # average best-so-far
    # iter_gbest_mean = np.round(np.divide(accum_g_best, run_index.reshape(-1, 1)), 4)
    # # average fitness 
    # iter_fitness_mean = np.round(accum_fitness/run_index, 4)
    # median
    # g_best_median = np.median(best_each_gbest, axis=0)
    # print("median:", g_best_median)
    average_best_so_far = np.round(np.mean(best_each_fitness), 4)   
    std_best_so_far = np.round(np.std(best_each_fitness), 4)
    average_mean_fitness = np.round(np.mean(each_best_so_far), 4)
    median_best_so_far = np.round(np.median(best_each_fitness), 4)
    # print("best_each_fitness: ", np.round(best_each_fitness.flatten(), 2))
    # print("best fitness: ", np.round(iter_fitness_mean, 2))
    # print("average-best-so-far vs iteration: ", np.mean(each_best_so_far, axis=0)) # need to plot
    plt.plot(np.mean(each_best_so_far, axis=0))
    plt.title('{} average best fitness of PSO'.format(save_name))
    plt.xlabel("iteration")
    plt.ylabel("fitness")
    plt.savefig("report/{}.png".format(save_name))
    plt.close()
    print("average best-so-far: {}, best-so-far std: {}, average mean fitness: {}, median best-so-far: {}".format(average_best_so_far, std_best_so_far, average_mean_fitness, median_best_so_far))
    
    


def main():
    runs = 50
    pop = 50
    epochs = 2500
    print("##########\n epoch= {}\n##########".format(epochs))
    # F2
    print("optimize F2 function")
    dim = 30
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    each_best_so_far = np.zeros((runs, epochs))
    for i in range(runs):
        pso = PSO(pop, dim, -10, 10, epochs)
        best_sofar, best_fitness, best_so_far = pso.optimal(fit.F2)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
        each_best_so_far[i] = best_so_far
    evaluation(best_each_gbest, best_each_fitness, each_best_so_far, runs, save_name='F2_{}'.format(epochs))

    #F3
    print("optimize F3 function")
    dim = 30
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    each_best_so_far = np.zeros((runs, epochs))
    for i in range(runs):
        pso = PSO(pop, dim, -100, 100, epochs)
        best_sofar, best_fitness, best_so_far = pso.optimal(fit.F3)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
        each_best_so_far[i] = best_so_far
    evaluation(best_each_gbest, best_each_fitness, each_best_so_far, runs, save_name='F3_{}'.format(epochs))

    #F11
    print("optimize F11 function")
    dim = 30
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    each_best_so_far = np.zeros((runs, epochs))
    for i in range(runs):
        pso = PSO(pop, dim, -600, 600, epochs)
        best_sofar, best_fitness, best_so_far = pso.optimal(fit.F11)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
        each_best_so_far[i] = best_so_far
    evaluation(best_each_gbest, best_each_fitness, each_best_so_far, runs, save_name='F11_{}'.format(epochs))

    #F12
    print("optimize F12 function")
    dim = 30
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    each_best_so_far = np.zeros((runs, epochs))
    for i in range(runs):
        pso = PSO(pop, dim, -50, 50, epochs)
        best_sofar, best_fitness, best_so_far = pso.optimal(fit.F12)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
        each_best_so_far[i] = best_so_far
    evaluation(best_each_gbest, best_each_fitness, each_best_so_far, runs, save_name='F12_{}'.format(epochs))

    #F13
    print("optimize F13 function")
    dim = 30
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    each_best_so_far = np.zeros((runs, epochs))
    for i in range(runs):
        pso = PSO(pop, dim, -50, 50, epochs)
        best_sofar, best_fitness, best_so_far = pso.optimal(fit.F13)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
        each_best_so_far[i] = best_so_far
    evaluation(best_each_gbest, best_each_fitness, each_best_so_far, runs, save_name='F13_{}'.format(epochs))

    #F14
    print("optimize F14 function")
    dim = 2
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    each_best_so_far = np.zeros((runs, epochs))
    for i in range(runs):
        pso = PSO(pop, dim, -65, 65, epochs)
        best_sofar, best_fitness, best_so_far = pso.optimal(fit.F14)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
        each_best_so_far[i] = best_so_far
    evaluation(best_each_gbest, best_each_fitness, each_best_so_far, runs, save_name='F14_{}'.format(epochs))

    #F15
    print("optimize F15 function")
    dim = 4
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    each_best_so_far = np.zeros((runs, epochs))
    for i in range(runs):
        pso = PSO(pop, dim, -5, 5, epochs)
        best_sofar, best_fitness, best_so_far = pso.optimal(fit.F15)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
        each_best_so_far[i] = best_so_far
    evaluation(best_each_gbest, best_each_fitness, each_best_so_far, runs, save_name='F15_{}'.format(epochs))

    # Gomez and Levy function(modified)
    print("optimize Gomez and Levy function")
    dim = 2
    best_each_gbest = np.zeros((runs, dim)) 
    best_each_fitness = np.zeros((runs, 1)) 
    each_best_so_far = np.zeros((runs, epochs))
    for i in range(runs):
        pso = PSO(pop, dim, -1, 0.75, epochs, GL=True)
        best_sofar, best_fitness, best_so_far = pso.optimal(fit.Gomez_Levy)
        best_each_gbest[i] = best_sofar
        best_each_fitness[i] = best_fitness
        each_best_so_far[i] = best_so_far
    evaluation(best_each_gbest, best_each_fitness, each_best_so_far, runs, save_name='GL_{}'.format(epochs))

if __name__=="__main__":
    main()
    # te_array = np.array([[2, 3, 2],[4, 6, 1]])
    # te_array = np.array([[0, 0, 0],[0, 0, 0]])
    # te_array = np.array([[0, 0],[0, 0], [0, 0]])
    # te_array = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 0]])
    # print(F11(te_array))
    # print(fit.Gomez_Levy(te_array))