import random
import operator

import numpy as np
import matplotlib.pyplot as plt
from util import pareto_dominance_min, generate_min_front, area_under_curve, load_split_all
from primitives import if_then_else

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# Evaluation of individual
# Note: There is no train, test, just testing of our individuals, we need to maintain all train data and evaluate
# Consider partial evaluations
def evalSymbReg(individual, pset, data, labels):
    predictor = gp.compile(expr=individual, pset=pset)
    predictions = 1 * np.asarray([predictor(*person) for person in data])
    # casting to array for printing
    # labels = np.asarray(labels)
    fp = 0
    fn = 0
    for pred, label in zip(predictions, labels):
        if pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    # Better use of np is welcome
    print(fp, fn)
    return fp, fn


def main():
    # Import data
    x_train, y_train = load_split_all()[0]
    data = x_train
    labels = y_train
    print(x_train.shape)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Arguments
    random.seed(25)
    crossover_rate = 0.5
    mutation_rate = 0.2

    input_types = []
    for i in range(x_train.shape[1]):  # multiplication op doesn't work
        input_types.append(float) 
    pset = gp.PrimitiveSetTyped("MAIN", input_types, bool)
    # logic
    pset.addPrimitive(np.logical_not, [bool], bool)
    pset.addPrimitive(np.logical_and, [bool], bool)  # Demorgan's rule says all logic ops can be made with not & and
    pset.addPrimitive(operator.xor, [bool, bool], bool)
    # arithmetic
    pset.addPrimitive(np.negative, [float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    # structure
    pset.addPrimitive(if_then_else, [bool, float, float], float)
    pset.addTerminal(3.0, float)
    pset.addTerminal(1, bool)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalSymbReg, pset=pset, data=data, labels=labels)
    # select
    toolbox.register("select", tools.selTournament, tournsize=3)
    # crossover
    toolbox.register("mate", gp.cxOnePoint)
    # mutate
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    gen = range(10)
    avg_list = []
    max_list = []
    min_list = []

    pop = toolbox.population(n=300)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    a_given_individual = toolbox.individual()
    a_given_individual.fitness.values = toolbox.evaluate(a_given_individual)

    # redundant
    '''fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit'''

    # Begin the evolution
    for g in gen:
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_rate:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness .... define invalid???
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        g_max = max(fits)
        g_min = min(fits)
    
        avg_list.append(mean)
        max_list.append(g_max)
        min_list.append(g_min)
    
        # print("  Min %s" % g_min)
        # print("  Max %s" % g_max)
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)

    print("-- End of (successful) evolution --")
   
    hof_pop = generate_min_front(pop)
    # Extract fitnesses and sort so HoF draws correctly
    hof = np.asarray([ind.fitness.values for ind in hof_pop])
    hof = hof[np.argsort(hof[:, 0])]

    # Charts
    pop_1 = [ind.fitness.values[0] for ind in pop]
    pop_2 = [ind.fitness.values[1] for ind in pop]

    plt.scatter(pop_1, pop_2, color='b')
    plt.scatter(hof[:, 0], hof[:, 1], color='r')
    plt.plot(hof[:, 0], hof[:, 1], color='r', drawstyle='steps-post')
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Tree Size")
    plt.title("Pareto Front")
    plt.show()

    print(area_under_curve(hof))
main()
