"""
Genetic algorithm
"""

import random
import logging
import logging.config
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
import yaml
from feature_selection import Full_search
from params import params

logger = logging.getLogger("genetic")


class Genetic:
    """
    Class genetic algorithm with different methods of selection, crossover and mutation.

    You can read more about the methods of selection, crossing and mutation in the book by
    Eyal Wirsansky with title "Hands-On Genetic Algorithms with Python".

    Parameters
    ----------
    selection:
            {str} Choose one of the following selection methods:
                - "fitness_proportionate_selection"
                - "ranking_selection"
                - "scaling_selection"
                - "tournament_selection"
    crossover:
            {str} Choose one of the following crossover methods:
                - "one_point_crossing"
                - "two_point_crossing"
                - "k_point_crossing"
                - "uniformly"
    mutate:
            {str} Choose one of the following crossover methods:
                - "inverting_bit"
                - ......
    score:
            {function} Put function score.
    max_generations:
            {int} Maximum number of generations.
    size_of_population:
            {int} Population size in each generation.
    len_of_individual:
            {int} The length of the sequence describing the individual.
    elite_size:
            {int} Elite size in each generation. The number of best individuals to be copied
             to the next generation.
    p_mutations:
            {float} Probability of mutation. 0 < p_mutations < 1.
    p_crossover:
            {float} Probability of crossover. 0 < p_mutations < 1.
    optimization_policy:
            {str} Choose "max" if you want to minimize the score value.
            Else choose "min".
    estimator_list:
            {list} List of estimators.

    Attributes
    ----------
    list_of_individual:
            The list of individuals in the binary version if you have solved the problem of
            selecting features. If you want to get a list of attributes, then use the function
            (self.binary_to_integer()) on each individual.
    q_list:
            List of score corresponding to the list_of_individual.
    X_train:
            Train dataset
    y_train:
            target for X_train
    X_test:
            validate dataset
    y_test:
            target for X_test
    q_best:
            Best value of score
    best_sub_features:
            List of features with best value of score

    Method
    ----------
    fit():
            A feature selection method using a genetic algorithm.

    Examples
    ----------
    gen = Genetic(len_of_individual=10, size_of_population=50, max_generations=100, elite_size=5,
                  p_mutations=0.15, p_crossover=0.95, optimization_policy="min",
                  estimator_list=['LinearRegression'])
    data = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                        data.target,
                                                        test_size=0.33,
                                                        random_state=42)
    gen.fit(X_train, X_test, y_train, y_test, d_steps=25)
    print('Best features: ', gen.best_sub_features, 'Q best: ', gen.q_best)
    """
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
                 estimator_list=["LinearRegression"]):
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

    def __repr__(self):
        our_arguments = ["selection", "crossover", "mutate", "score", "max_generations",
                         "size_of_population", "len_of_individual", "elite_size", "p_mutations",
                         "p_crossover", "optimization_policy", "estimator_list"]
        res = []
        for argument in our_arguments:
            if argument != "score":
                res.append(argument + "=" + str(gen.__dict__[argument]))
            else:
                res.append(argument + "=" + gen.__dict__[argument].__name__)
        res = ", ".join(res)
        res = "Genetic(" + res + ")"
        return res

    def calcul_Q(self, subset_features):
        """
        Calculate score on this list of features
        Input
        ----------
        subset_features:
              list of features

        Return
        ----------
        value of score
        """
        self.estimator.fit(self.X_train[:, subset_features], self.y_train)
        Y_pred = self.estimator.predict(self.X_test[:, subset_features])
        res = self.score(self.y_test, Y_pred)
        # logger.info('score: {1:9.6f} for features: {1}'.format(res, subset_features))
        return res

    # generate population
    def generate_binary_string(self):
        """
        Calculate score on this list of features
        Input
        ----------
        Nothing

        Return
        ----------
        {list of lists} Where each binary list of size self.len_of_individual.
        """
        self.list_of_individual = []
        for _ in range(self.size_of_population):
            self.list_of_individual.append(random.choices([0, 1], k=self.len_of_individual))

    # selection methods.
    def fitness_proportionate_selection(self):
        """
        The roulette selection method, or fitness proportional selection (FPS), is designed
        so that the probability of selecting an individual is directly proportional to his fitness.

        Return
        ----------
        self.list_of_individual of relative probabilities.
         """
        if self.optimization_policy == "min":
            list_of_probabilities = [1 / q_i for q_i in self.q_list]
        else:
            list_of_probabilities = self.q_list[:]
        sum_of_q = sum(self.q_list)
        list_of_probabilities = [q_i / sum_of_q for q_i in list_of_probabilities]
        return random.choices(self.list_of_individual, weights=list_of_probabilities,
                              k=self.size_of_population - self.elite_size)

    def ranking_selection(self):
        """
        Selection based on rank by score values

        Return
        ----------
        self.list_of_individual of relative probabilities.
        """
        z = list(zip(self.q_list, self.list_of_individual))
        if self.optimization_policy == "min":
            z.sort(key=lambda x: -x[0])
        else:
            z.sort()
        self.list_of_individual = [individ for q, individ in z]
        sum_of_rank = ((1 + self.size_of_population) * self.size_of_population) / 2
        list_of_probabilities = [i / sum_of_rank for i in range(1, self.size_of_population + 1)]
        return random.choices(self.list_of_individual, weights=list_of_probabilities,
                              k=self.size_of_population - self.elite_size)

    def scaling_selection(self, left=0, right=1):
        """
        Selection is based on scaling score values

        Return
        ----------
        self.list_of_individual of relative probabilities.
        """
        max_elem = -1e12
        min_elem = 1e12
        for i in self.q_list:
            if i > max_elem:
                max_elem = i
            if i < min_elem:
                min_elem = i
        res = np.linalg.solve([[min_elem, 1], [max_elem, 1]], [left, right])
        self.q_list = [res[0] * i + res[1] for i in self.q_list]
        return self.fitness_proportionate_selection()

    def tournament_selection(self, number_of_participants=3):
        """
        Selection is based on tournaments.

        Return
        ----------
        self.list_of_individual of relative probabilities.
        """
        pass

    # crossover methods
    def one_point_crossing(self, parent_1, parent_2):
        """
        Input
        ----------
        Two individuals (parents)

        Return
        ----------
        Two new individuals (childs)
        """
        if random.uniform(0, 1) <= self.p_crossover:
            point = random.randint(1, len(parent_1) - 1)
            child_1 = parent_1[:point] + parent_2[point:]
            child_2 = parent_2[:point] + parent_1[point:]
            return child_1, child_2
        return parent_1, parent_2

    def two_point_crossing(self, parent_1, parent_2):
        """
        Input
        ----------
        Two individuals (parents)

        Return
        ----------
        Two new individuals (childs)
        """
        if random.uniform(0, 1) <= self.p_crossover:
            point_1 = random.randint(1, len(parent_1) // 2)
            point_2 = random.randint(len(parent_1) // 2 + 1, len(parent_1) - 1)
            child_1 = parent_1[:point_1] + parent_2[point_1:point_2] + parent_1[point_2:]
            child_2 = parent_2[:point_1] + parent_1[point_1:point_2] + parent_2[point_2:]
            return child_1, child_2
        return parent_1, parent_2

    def k_point_crossing(self, parent_1, parent_2, k=3):
        """
        Input
        ----------
        Two individuals (parents)

        Return
        ----------
        Two new individuals (childs)
        """
        if random.uniform(0, 1) <= self.p_crossover:
            for i in range(k):
                point = random.randint(1, len(parent_1) - 1)
                child_1 = parent_1[:point] + parent_2[point:]
                child_2 = parent_2[:point] + parent_1[point:]
                parent_1, parent_2 = child_1, child_2
            return child_1, child_2
        return parent_1, parent_2

    def uniformly(self, parent_1, parent_2):
        """
        Input
        ----------
        Two individuals (parents)

        Return
        ----------
        Two new individuals (childs)
        """
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
        """
        Randomly selects a position and inverts the bit with probability equal to self.p_mutations.
        Input
        ----------
        individual

        Return
        ----------
        individuals
        """
        if random.uniform(0, 1) <= self.p_mutations:
            point = random.randint(0, len(individual) - 1)
            if individual[point] == 0:
                individual[point] = 1
            else:
                individual[point] = 0
        return individual

    # other functions
    def binary_to_integer(self, line):
        """
        Сonverting a binary form to an integer

        Input
        ----------
        Individual in binary form.

        Return
        ----------
        List of item numbers where units were placed
        """
        list_of_features = [i for i, j in enumerate(line) if j == 1]
        if len(list_of_features) != 0:
            return list_of_features
        else:
            return [0, ]

    def find_q(self):
        """
        Calculating the error for the self.list_of_individual.
        """
        logger.info('calculating the error for the descendants of the previous generation')
        self.q_list = [self.calcul_Q(self.binary_to_integer(individual)) for individual in \
                       self.list_of_individual]
        logger.info('list of errors for each individual:\n {0}'.format(self.q_list))
        logger.info('the list of errors for each of the individual elite:\n {0}'.format(
            self.q_elite))
        list_zip = list(zip(self.q_elite + self.q_list,
                            self.list_of_elite + self.list_of_individual))
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
        """

        X_train:
            Train dataset
        y_train:
            target for X_train
        X_test:
            validate dataset
        y_test:
            target for X_test
        task:
            {str} type of machine learning task:
                - 'Regression'
                - 'Classification'
        d_steps:
            {int} Stop condition. Number of steps without improving score.

        Return
        ----------
            Trained instances with q_best and best_subfeatures attributes.
        """
        logger.info('Feature selection begins by genetic algorithm')

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

            # make population (size = size_of_population) that consists of
            # individual(size = len_of_individual)
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
                    'best value of error and individual on generation {0}: {1:9.6f} - {2}'.format(
                        j, self.q_best, self.best_sub_features)
                )

                # make selection
                logger.info('list of elite individual: {}'.format(self.list_of_elite))
                random_list = func_selection(self)

                # make crossover
                if len(random_list) % 2 == 0:
                    for i, j in zip(range(0, len(random_list), 2), range(1, len(random_list), 2)):
                        # random_list[i], random_list[j] =
                        # self.two_point_crossing(random_list[i], random_list[j])
                        random_list[i], random_list[j] = func_crossover(self, random_list[i],
                                                                        random_list[j])
                else:
                    for i, j in zip(range(0, len(random_list) - 1, 2),
                                    range(1, len(random_list) - 1, 2)):
                        # random_list[i], random_list[j] = s
                        # elf.two_point_crossing(random_list[i], random_list[j])
                        random_list[i], random_list[j] = func_crossover(self,
                                                                        random_list[i],
                                                                        random_list[j])

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
    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                        data.target,
                                                        test_size=0.33,
                                                        random_state=42)

    print('FullSearch')
    full.fit(X_train, X_test, y_train, y_test)
    print('F best: ', full.best_sub_features, ' Q best: ', full.q_best)

    print('Genetic')
    gen.fit(X_train, X_test, y_train, y_test, d_steps=25)
    print('F best: ', gen.best_sub_features, ' Q best: ', gen.q_best)
