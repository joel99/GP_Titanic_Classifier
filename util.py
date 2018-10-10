#!/usr/bin/env python
"""
AAD - Titanic Dataset Paretodominance Demo
Data Parser Driver
== Team 4 ==
Aaron McDaniel
Jeffrey Minowa
Joshua Reno
Joel Ye
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
import csv
import random

data_dir = 'data/'
train_fn = 'train.csv'
test_fn = 'test.csv'
test_label_fn = 'gender_submission.csv'
folds = 5

def load_data(filename):
    url = data_dir + filename
    df = pd.read_csv(url, sep=',')
    print("Loaded " + filename)
    return df.values

# Returns: clean train_data, test_data
def load_split_all():
    le = LabelEncoder()
    train_data = load_data(train_fn)
    test_data = load_data(test_fn)
    test_labels = load_data(test_label_fn)

    # Note test data has different data order
    # Convert sex column (col 4)
    le.fit(["male", "female"])
    train_data[:, 4] = le.transform(train_data[:, 4])
    test_data[:, 3] = le.transform(test_data[:, 3])

    # Convert embark column (col 11)
    # le.fit(["S", "C", "Q", None])
    # print(train_data[:, 11])
    # train_data[:, 11] = le.transform(train_data[:, 11])
    # test_data[:, 10] = le.transform(test_data[:, 10])
    
    # Feature selection:
    # Trim passenger_id (c0), name (c3), ticket number (c8), cabin number (c10)
    # As we're unsure about cabin_number domain effect, we're just dropping it
    # Dropping embark since we think it's not too helpful, and has NaN
    train_data = np.delete(train_data, [0, 3, 8, 10, 11], axis = 1)
    test_data = np.delete(test_data, [2, 7, 9, 10], axis = 1)

    # Fill in NaN
    # test_data = pd.DataFrame(test_data)
    # test_data = test_data.fillna(test_data.mean())
    #
    # train_data = pd.DataFrame(train_data)
    # train_data = train_data.fillna(train_data.mean())

    train_data = np.where(pd.isnull(train_data), -1, train_data)
    test_data = np.where(pd.isnull(test_data), -1, test_data)
    x_test = np.where(pd.isnull(test_data), -1, test_data)
    y_test = test_labels

    # Separate train_data into x and y
    x_train = train_data[:, 1:].astype('float')
    y_train = train_data[:, 0].astype('int')
    return ((x_train, y_train), (x_test, y_test))

def normalize(x_train, x_largest_in_each_col):
    for indx, x in enumerate(x_train):
        for indy, y in enumerate(x_train[indx]):
            x_train[indx][indy] = y/x_largest_in_each_col[indy]

def pareto_dominance_max(ind1, ind2):
    """
    returns true if ind1 dominates ind2 by the metrics that should be maximized

    :param ind1: tuple of precision and recall scores
    :param ind2: tuple of precision and recall scores
    :return: boolean representing if ind1 dominates ind2 using metrics that should be maximized
    """

    not_equal = False
    for value_1, value_2 in zip(ind1.fitness.values, ind2.fitness.values):
        if value_1 < value_2:
            return False
        elif value_1 > value_2:
            not_equal = True
    return not_equal

def pareto_dominance_min(ind1, ind2):
    """
    returns true if ind1 dominates ind2 by the metrics that should be minimized

    :param ind1: tuple of FP and FN
    :param ind2: tuple of FP and FN
    :return: boolean representing if ind1 dominates ind2 using the metrics that should be minimized
    """
    not_equal = False
    for value_1, value_2 in zip(ind1.fitness.values, ind2.fitness.values):
        if value_1 > value_2:
            return False
        elif value_1 < value_2:
            not_equal = True    
    return not_equal


def update_front(front, ind, comp):
    """
    Makes a new pareto front out of the old pareto front and new individual
    In this context an individual consists of scores and their hyper parameters
    For example ind[0] is a tuple of precision and recall scores
    and ind[1] is a list of the hyper-parameters needed to recreate the classifier

    :param front: the old pareto front to be updated
    :param ind: the new individual that may or may not change the old pareto front
    :param comp: the method used to compare individuals as being pareto dominant or not
    :return: the new pareto front
    """

    # A member belongs on the front if it dominates or is not dominated by new ind
    # New ind belongs on front if it is not dominated by any
    # If new ind dominated, rest of front won't be dominated
    newFront = []
    isNewDominated = False
    for i in range(len(front)):
        old = front[i]
        if comp(old, ind): # Careful to compare the scores
            isNewDominated = True
            break
        if not comp(ind, old):
            newFront.append(old)
    if isNewDominated:
        newFront.extend(front[i:]) # add rest of old front
    else:
        newFront.append(ind)
    return newFront

def generate_front(population, comp):
    front = []
    for individual in population:
        front = update_front(front, individual, comp)
    return front

def generate_min_front(population):
    return generate_front(population, pareto_dominance_min)

def area_under_curve(fitnesses):
    return np.sum(np.abs(np.diff(fitnesses[:,0]))*fitnesses[:-1,1])

"""    
    Takes in a classifier and writes predictions to a csv file after 
    being trained
    :param clf: classifier
    :param train_data: training data
    :param train_label: training label
    :param test_data: testing data
    :param clf_name: String of the name of the classifier that you want to make csv of
"""
def convert_to_csv(clf, train_data, train_label, test_data, clf_name):
    clf.fit(train_data, train_label)
    id_column = test_data[:, 0]
    test_data = test_data[:, 1:]
    predictions = np.asarray(clf.predict(test_data))
    final = np.column_stack((id_column, predictions))
    df = pd.DataFrame(final)
    df.to_csv(clf_name, index=False, header=["PassengerId", "Survived"])


def sim_aneal_select(population, k, tourn_size, anneal_rate):
    """
    A tournament select style with similarities ot Simulated Annealing.
    The probability of choosing a random tournament candidate decreases
    exponentially as the number of selections goes up.
    Should find a good balance between diversity and performance.

    :param population: the set of individuals to select from
    :param k: The number of individuals to select
    :param tourn_size: the size of the tournament rounds
    :param anneal_rate: the factor to decrease the probability of picking randomly
    :return: A set of selected individuals
    """
    random_rate = 1.0
    selected = []
    for i in range(k):
        tourn = []
        best = None
        for j in range(tourn_size):
            # build list for tournament
            tourn.append(random.choice(population))
            # determine best individual
            if j == 0:
                best = tourn[-1]
            elif pareto_dominance_min(tourn[-1], best):
                best = tourn[-1]
        # pick randomly with likelihood of random_rate
        if random.random() > random_rate:
            selected.append(best)
        else:
            selected.append(random.choice(tourn))
        # update random_rate
        random_rate *= anneal_rate
    return selected
