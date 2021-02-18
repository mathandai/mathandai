import random
import numpy as np
from sklearn.model_selection import train_test_split
from params import params
import copy

import random
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from feature_selection import Full_search
from params import params

import logging
import logging.config
import yaml
from collections import defaultdict

logger = logging.getLogger("genetic")


class Genetic:
    def __init__(self,
                 selection="fitness_proportionate_selection",
                 crossover="two_point_crossing",
                 mutate="inverting_bit",
                 score=mean_squared_error,
                 max_generations=100,
                 size_of_population=200,
                 len_of_individual=100,
                 elite_size=10,
                 p_mutations=0.10,
                 p_crossover=0.9,
                 optimization_policy="max",
                 score_func="mse",
                 estimator_list=['LinearRegression']):
        self.selection = selection
        self.crossover = crossover
        self.mutate = mutate
        self.score = score
        self.max_generations = max_generations
        self.size_of_population = size_of_population
        self.len_of_individual = len_of_individual
        self.elite_size = elite_size
        self.p_mutations = p_mutations
        self.p_crossover = p_crossover
        self.optimization_policy = optimization_policy
        self.estimator_list = estimator_list

    def calcul_Q(self, subset_features):
        self.estimator.fit(self.X_train[:, subset_features], self.y_train)
        Y_pred = self.estimator.predict(self.X_test[:, subset_features])
        res = self.score(self.y_test, Y_pred)
        # logger.info('score: {1:9.6f} for features: {1}'.format(res, subset_features))
        return res

    # generate population
    def generate_binary_string(self):
        self.list_of_individual = []
        for i in range(self.size_of_population):
            self.list_of_individual.append(random.choices([0, 1], k=self.len_of_individual))

    # selection methods
    def fitness_proportionate_selection(self):
        if self.optimization_policy == "min":
            list_of_probabilities = [1 / q_i for q_i in self.q_list]
        else:
            list_of_probabilities = self.q_list[:]
        sum_of_q = sum(self.q_list)
        list_of_probabilities = [q_i / sum_of_q for q_i in list_of_probabilities]
        return random.choices(self.list_of_individual, weights=list_of_probabilities,
                              k=self.size_of_population - self.elite_size)

    def ranking_selection(self):
        z = list(zip(self.q_list, self.list_of_individual))
        if self.optimization_policy == "min":
            z.sort(key=lambda x: -x[0])
        else:
            z.sort()
        self.list_of_individual = [individ for q, individ in z]
        sum_of_rank = ((1 + self.size_of_population) * self.size_of_population) / 2
        list_of_probabilities = [i / sum_of_rank for i in range(1, self.size_of_population + 1)]
        return np.random.choice(self.list_of_individual, size=3, replace=False, p=list_of_probabilities)

    def scaling_selection(self, a=0, b=1):
        max_elem = -1e12
        min_elem = 1e12
        for i in self.q_list:
            if i > max_elem:
                max_elem = i
            if i < min_elem:
                min_elem = i
        res = np.linalg.solve([[min_elem, 1], [max_elem, 1]], [a, b])
        self.q_list = [res[0] * i + res[1] for i in self.q_list]
        return fitness_proportionate_selection()

    def tournament_selection(self, number_of_participants=3):
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
    def binary_to_integer(self, line):
        list_of_features = [i for i, j in enumerate(line) if j == 1]
        if len(list_of_features) != 0:
            return list_of_features
        else:
            return [0, ]

    def find_q(self):
        logger.info('calculating the error for the descendants of the previous generation')
        self.q_list = [self.calcul_Q(self.binary_to_integer(individual)) for individual in self.list_of_individual]
        logger.info('list of errors for each individual:\n {0}'.format(self.q_list))
        logger.info('the list of errors for each of the individual elite:\n {0}'.format(self.q_elite))
        list_zip = list(zip(self.q_elite + self.q_list, self.list_of_elite + self.list_of_individual))
        if self.optimization_policy == "max":
            list_zip.sort(key=lambda x: -x[0])
        else:
            list_zip.sort()
        self.q_list = [list_zip[i][0] for i in range(0, self.size_of_population)]
        self.list_of_individual = [list_zip[i][1] for i in range(0, self.size_of_population)]
        self.q_elite = [list_zip[i][0] for i in range(self.elite_size)]
        self.list_of_elite = [list_zip[i][1] for i in range(self.elite_size)]

    # feature selection
    def fit(self, X_train, X_test, y_train, y_test, task='Regression', d_steps=5):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.task = task
        self.dim = X_train.shape[1]

        func_selection = self.__class__.__dict__[self.selection]
        func_crossover = self.__class__.__dict__[self.crossover]
        func_mut = self.__class__.__dict__[self.mutate]

        for estim in self.estimator_list:
            self.estimator = params[self.task][estim]['estimator']

            # make population (size = size_of_population) that consists of individual(size = len_of_individual)
            self.generate_binary_string()

            self.q_elite = []
            self.list_of_elite = []

            self.q_best = 1e12
            self.best_sub_features = []
            local_q_best = 1e12
            local_best_sub_features = []

            logger.info('the estimator used is {}'.format(self.estimator))
            j_0 = 0
            for j in range(self.max_generations):
                logger.info('Generation number is {}'.format(j))
                self.find_q()
                if self.elite_size != 0:
                    local_q_best = self.q_elite[0]
                    local_best_sub_features = self.list_of_elite[0]
                else:
                    local_q_best = self.q_list[0]
                    local_best_sub_features = self.list_of_individual[0]

                if self.optimization_policy == "max" and local_q_best > self.q_best:
                    self.q_best = local_q_best
                    self.best_sub_features = local_best_sub_features[:]
                    j_0 = j
                elif self.optimization_policy == "min" and local_q_best < self.q_best:
                    self.q_best = local_q_best
                    self.best_sub_features = local_best_sub_features[:]
                    j_0 = j

                # the stop criteria
                if j - j_0 > d_steps:
                    break

                logger.info(
                    'best value of error and individual on generation {0}: {1:9.6f} - {2}'.format(j, self.q_best,
                                                                                                  self.best_sub_features))

                # make selection
                logger.info('list of elite individual: {}'.format(self.list_of_elite))
                random_list = func_selection(self)

                # make crossover
                if len(random_list) % 2 == 0:
                    for i, j in zip(range(0, len(random_list), 2), range(1, len(random_list), 2)):
                        # random_list[i], random_list[j] = self.two_point_crossing(random_list[i], random_list[j])
                        random_list[i], random_list[j] = func_crossover(self, random_list[i], random_list[j])
                else:
                    for i, j in zip(range(0, len(random_list) - 1, 2), range(1, len(random_list) - 1, 2)):
                        # random_list[i], random_list[j] = self.two_point_crossing(random_list[i], random_list[j])
                        random_list[i], random_list[j] = func_crossover(self, random_list[i], random_list[j])

                # make mutation
                for i in range(len(random_list)):
                    # random_list[i] = self.inverting_bit(random_list[i])
                    random_list[i] = func_mut(self, random_list[i])
                self.list_of_individual = random_list[:]


def setup_logging():
    """set loging with yaml"""
    with open("logging.conf.yml") as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))

if __name__ == '__main__':
    setup_logging()
    print('Dataset: ', 'diabetes')
    full = Full_search(estimator_list=['LinearRegression'])
    gen = Genetic(len_of_individual=10, size_of_population=50, max_generations=100, elite_size=5,
                  p_mutations=0.15, p_crossover=0.95, optimization_policy="min",
                  estimator_list=['LinearRegression'])
    data = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)

    print('FullSearch')
    full.fit(X_train, X_test, y_train, y_test)
    print('F best: ', full.best_sub_features, ' Q best: ', full.q_best)

    print('Genetic')
    gen.fit(X_train, X_test, y_train, y_test, d_steps=25)
    print('F best: ', gen.best_sub_features, ' Q best: ', gen.q_best)
