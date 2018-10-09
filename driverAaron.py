import random
import operator
import math

import numpy as np
import matplotlib.pyplot as plt
from util import pareto_dominance_min, generate_min_front, area_under_curve, load_split_all
from primitives import if_then_else, is_greater, is_equal_to, relu

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
    return fp, fn


def main():
    # Import data
    x_train, y_train = load_split_all()[0]
    data = x_train
    labels = y_train
    num_positives = np.count_nonzero(labels)
    num_negatives = len(labels) - num_positives
    # num_positives is max false negatives
    # num_negatives is max false positives - append these to fitness so we have reliable AuC
    fn_trivial_fitness = (0, num_positives)
    fp_trivial_fitness = (num_negatives, 0)

    print(x_train.shape)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Arguments
    random.seed(25)
    crossover_rate = 0.5
    mutation_rate = 0.2
    tourn_size = 3
    tourn2_size = 1.5
    epsilon = 0.1


    input_types = []
    for i in range(x_train.shape[1]):  # multiplication op doesn't work
        input_types.append(float)
    pset = gp.PrimitiveSetTyped("MAIN", input_types, bool)

    # Essential Primitives
    pset.addPrimitive(is_greater, [float, float], bool)
    pset.addPrimitive(is_equal_to, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)

    pset.addPrimitive(np.logical_not, [bool], bool)
    pset.addPrimitive(np.logical_and, [bool, bool],
                      bool)  # Demorgan's rule says all logic ops can be made with not & and

    pset.addPrimitive(np.negative, [float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)

    # constants
    pset.addTerminal(10.0, float)
    pset.addTerminal(1, bool)
    pset.addTerminal(0, bool)

    # More primitives (for fun/tinkering/reducing verbosity of tree)

    # Logic to float

    # Float to logic

    # Logic to logic
    pset.addPrimitive(operator.xor, [bool, bool], bool)

    # Float to float
    pset.addPrimitive(relu, [float], float)
    # pset.addPrimitive(operator.pow, [float, int], float)
    pset.addPrimitive(math.floor, [float], int)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalSymbReg, pset=pset, data=data, labels=labels)
    # select
    toolbox.register("tournament_select", tools.selTournament, tournsize=tourn_size)
    toolbox.register("NSGA_select", tools.selNSGA2)
    toolbox.register("SPEA_select", tools.selSPEA2)
    toolbox.register("random_select", tools.selRandom)
    toolbox.register("best_select", tools.selBest)
    toolbox.register("worst_select", tools.selWorst)
    toolbox.register("dub_tournament_select", tools.selDoubleTournament, fitness_size=tourn_size, parsimony_size=tourn2_size, fitness_first=True)
    toolbox.register("stochastic_select", tools.selStochasticUniversalSampling)  # randomized with random class
    toolbox.register("dom_tournament_select", tools.selTournamentDCD)
    toolbox.register("lexicase_select", tools.selLexicase)
    toolbox.register("auto_eps_lexicase_select", tools.selAutomaticEpsilonLexicase)
    toolbox.register("eps_lexicase_select", tools.selEpsilonLexicase, epsilon=epsilon)
    # potentially useful
    toolbox.register("sort", tools.sortNondominated, first_front_only=True)

    # array of selection methods and list to contain their pareto fronts
    selects = [toolbox.eps_lexicase_select, toolbox.tournament_select, toolbox.NSGA_select, toolbox.SPEA_select,
               toolbox.random_select, toolbox.best_select, toolbox.worst_select, toolbox.dub_tournament_select,
               toolbox.stochastic_select, toolbox.lexicase_select, toolbox.auto_eps_lexicase_select]
    # not sure how to work dom_tournament_select
    names = ["eps_lexicase_select", "tournament_select", "NSGA_select", "SPEA_select", "random_select", "best_select",
             "worst_select", "dub_tournament_select", "stochastic_select", "lexicase_select",
             "auto_eps_lexicase_select"]
    size = len(selects)
    colors = [((i / size) % 1, (i * 3 / size) % 1, (i * 5 / size) % 1) for i in range(size)]
    all_fronts = []
    all_areas = []


    # crossover
    toolbox.register("mate", gp.cxOnePoint)
    # mutate
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # Main loop
    for select, name in zip(selects, names):
        random.seed(25)
        gen = range(40)
        avg_list = []
        max_list = []
        min_list = []

        pop = toolbox.population(n=300)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Begin the evolution
        print("Starting %s" %name)
        for g in gen:
            if g % 10 == 0:
                print("-- Generation %i of %i --" % (g, len(gen)))

            # Select the next generation individuals
            offspring = select(pop, len(pop))
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
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
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
        # all_fronts.append(toolbox.sort(pop, len(pop))[0])
        hof_pop = generate_min_front(pop)
        hof = np.asarray([ind.fitness.values for ind in hof_pop])
        hof = np.insert(hof, 0, [fp_trivial_fitness, fn_trivial_fitness], 0)
        hof = hof[np.argsort(hof[:, 0])]
        all_fronts.append(hof)
        all_areas.append(area_under_curve(hof))
        print("%s: %f" % (name, all_areas[-1]))

    # graph all the pareto fronts for each selection method
    for front, name, area, color in zip(all_fronts, names, all_areas, colors):
        plt.scatter(front[:, 0], front[:, 1], color=color, label=name)
        plt.plot(front[:, 0], front[:, 1], color=color, drawstyle='steps-post')
        print("%s: %f" % (name, area))
    plt.xlabel("False Positives")
    plt.ylabel("False Negatives")
    plt.title("Pareto Front")
    plt.legend()
    # save information to a file
    filename = "results/Aaron_select.csv"
    header = "Name, Area\n"
    file = open(filename, 'w')
    file.write(header)
    for name, area in zip(names, all_areas):
        file.write("%s,%f\n" % (name, area))
    file.close()
    plt.show()


main()
