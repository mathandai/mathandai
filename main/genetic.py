import random
import numpy as np
from sklearn.model_selection import train_test_split
from params import params
import copy

class Genetic:
    def __init__(self, select="", crossover="", mutate="", len_of_individual=100,
                 size_of_population=200, max_generations=100, elite_size=10,
                 p_mutations=0.10, p_crossover=0.9, fitness="", optimization_policy="max",
                 estimator_list=['LinearRegression']):
        self.max_generations = max_generations
        self.size_of_population = size_of_population
        self.len_of_individual = len_of_individual
        self.p_mutations = p_mutations
        self.p_crossover = p_crossover
        self.elite_size = elite_size

    def calcul_Q(self, subset_features):
        self.estimator.fit(self.X_train[:, subset_features], self.y_train)
        Y_pred = self.estimator.predict(self.X_test[:, subset_features])
        if self.task == 'Regression':
             return (np.square(np.subtract(self.y_test, Y_pred)).mean()) ** 1/2
            # return np.divide(np.abs(np.subtract(self.y_test, Y_pred)), self.y_test).mean()
        else:
            pass

    # generate population
    def generate_binary_string(self):
        self.list_of_individual = []
        for i in range(self.size_of_population):
            self.list_of_individual.append(random.choices([0, 1], k=len_of_individual))

    # selection methods
    def fitness_proportionate_selection(self, optimization_policy="max"):
        if optimization_policy == "min":
            self.q_list = [1 / q_i for q_i in self.q_list]
        sum_of_q = sum(self.q_list)
        list_of_probabilities = [q_i / sum_of_q for q_i in self.q_list]
        return np.random.choice(self.list_of_individual,
                                size=self.size_of_population - self.elite_size,
                                replace=False, p=list_of_probabilities)

    def ranking_selection(self, optimization_policy="max"):
        z = list(zip(self.q_list, self.list_of_individual))
        if optimization_policy == "min":
            z.sort(key=lambda x: -x[0])
        else:
            z.sort()
        self.list_of_individual = [individ for q, individ in z]
        sum_of_rank = ((1 + self.size_of_population) * self.size_of_population) / 2
        list_of_probabilities = [i / sum_of_rank for i in range(1, self.size_of_population + 1)]
        return np.random.choice(self.list_of_individual, size=3, replace=False, p=list_of_probabilities)

    def scaling_selection(self, optimization_policy="max", a=0, b=1):
        max_elem = -1e12
        min_elem = 1e12
        for i in self.q_list:
            if i > max_elem:
                max_elem = i
            if i < min_elem:
                min_elem = i
        res = np.linalg.solve([[min_elem, 1], [max_elem, 1]], [a, b])
        self.q_list = [res[0] * i + res[1] for i in self.q_list]
        return self.fitness_proportionate_selection(optimization_policy=optimization_policy)

    def tournament_selection(self, optimization_policy="max", number_of_participants=3):
        pass

    # crossover methods
    def one_point_crossing(self, parent_1, parent_2):
        if random.uniform(0, 1) <= self.p_crossover:
            point = random.randint(1, len(parent_1) - 1)
            child_1 = parent_1[:point] + parent_2[point:]
            child_2 = parent_2[:point] + parent_1[point:]
            return child_1, child_2
        return parent_1, parent_2

    def two_point_crossing(self, parent_1, parent_2):
        if random.uniform(0, 1) <= self.p_crossover:
            point_1 = random.randint(1, len(parent_1) // 2)
            point_2 = random.randint(len(parent_1) // 2 + 1, len(parent_1) - 1)
            child_1 = parent_1[:point_1] + parent_2[point_1:point_2] + parent_1[point_2:]
            child_2 = parent_2[:point_1] + parent_1[point_1:point_2] + parent_2[point_2:]
            return child_1, child_2
        return parent_1, parent_2

    def k_point_crossing(self, parent_1, parent_2, k=3):
        if random.uniform(0, 1) <= self.p_crossover:
            for i in range(k):
                point = random.randint(1, len(parent_1) - 1)
                child_1 = parent_1[:point] + parent_2[point:]
                child_2 = parent_2[:point] + parent_1[point:]
                parent_1, parent_2 = child_1, child_2
            return child_1, child_2
        return parent_1, parent_2

    def uniformly(self, parent_1, parent_2):
        if random.uniform(0, 1) <= self.p_crossover:
            child_1 = []
            child_2 = []
            for i in range(len(parent_1)):
                if random.uniform(0, 1) <= 0.5:
                    child_1.append(parent_1[i])
                    child_2.append(parent_2[i])
                else:
                    child_1.append(parent_2[i])
                    child_2.append(parent_1[i])
            return child_1, child_2
        return parent_1, parent_2

    # mutate methods
    def inverting_bit(self, individual):
        if random.uniform(0, 1) <= self.p_mutations:
            point = random.randint(0, len(individual) - 1)
            if individual[point] == 0:
                individual[point] = 1
            else:
                individual[point] = 0
        return individual

    # other functions
    # def make_elite:

    def find_q(self, optimization_policy="max"):
        self.q_list = [self.calcul_Q(individual) for individual in self.list_of_individual]
        list_zip = list(zip(self.q_elite + self.q_list, self.list_of_elite + self.list_of_individual))
        if optimization_policy == "max":
            list_zip.sort(key=lambda x: -x[0])
        else:
            list_zip.sort()
        self.q_list = [list_zip[i][0] for i in range(self.elite_size, self.size_of_population)]
        self.list_of_individual = [list_zip[i][1] for i in range(self.elite_size, self.size_of_population)]
        self.q_elite = [list_zip[i][0] for i in range(self.elite_size)]
        self.list_of_elite = [list_zip[i][1] for i in range(self.elite_size)]

    # feature selection
    def fit(self, X_train, X_test, y_train, y_test, task='Regression', optimization_policy="max", d_steps=5):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.task = task
        self.dim = X_train.shape[1]

        for estim in self.estimator_list:
            self.estimator = params[self.task][estim]['estimator']
            # make population (size = size_of_population) that consists of individual(size = len_of_individual)
            generate_binary_string()

            self.q_elite = []
            self.list_of_elite = []

            self.Q_best = 1e12
            self.best_sub_features = []

            sub_features = [[]]
            history_Q = []

            local_Q_best = 1e12
            local_best_sub_features = []

            print(self.estimator)

            for j in range(self.max_generations):
                self.find_q(optimization_policy="max")
                if self.elite_size != 0:
                    local_Q_best = self.q_elite[0]
                else:
                    local_Q_best = self.q_list[0]
                if optimization_policy == "max" and local_Q_best > self.Q_best:
                    self.Q_best = local_Q_best
                    j_0 = j
                elif optimization_policy == "min" and local_Q_best < self.Q_best:
                    self.Q_best = local_Q_best
                    j_0 = j
                if j - j_0 > d_steps:
                    break
                # make crossover
                random_list = fitness_proportionate_selection(self, optimization_policy="max")
                if len(random_list) % 2 == 0:
                    for i, j in zip(range(0, len(random_list), 2), range(1, len(random_list), 2)):
                        random_list[i], random_list[j] = self.uniformly(random_list[i], random_list[j])
                else:
                    for i, j in zip(range(0, len(random_list) - 1, 2), range(1, len(random_list) - 1, 2)):
                        random_list[i], random_list[j] = self.uniformly(random_list[i], random_list[j])
                        # make mutation
                for i in range(len(random_list)):
                    random_list[i] = self.inverting_bit(random_list[i])
