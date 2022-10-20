#! usr/bin python3

# filename:     ga_class.py
# desc:         class for the gaPhrase algorithm
# author:       Tristan McGuire
# date:         October 2022

import random


ASCII = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
         53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
         74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
         95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
         113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126]


class Individual(object):
    """
        An individual phrase in the population of possible phrases.
    """
    def __init__(self, target_length: int = 0, chromosome: list = None) -> None:
        self.length = target_length
        if chromosome is not None:
            self.chromosome = chromosome
        else:
            self.chromosome = self.init_chromosome()

        if len(self.chromosome) != self.length:
            raise Exception(f'Chromosome list given does not match target length for Individual.__init__()\n'
                            f'Target: {self.length}\tGiven:{len(self.chromosome)}\n')
        elif self.length < 1:
            raise Exception(f'Target length is 0 - no target length for Individual.__init__()\n')

        self.string = self.output_chromosome()
        self.fitness = 0

    def init_chromosome(self) -> list:
        temp = []
        for x in range(self.length):
            temp.append(random.choice(ASCII))
        return temp

    def output_chromosome(self) -> str:
        temp = ''
        for x in self.chromosome:
            temp += chr(x)
        return temp

    def get_fitness(self, target: str = None) -> None:
        if target is None:
            raise Exception('\nNo target value available for Individual.get_fitness()\n')

        for x in range(len(target)):
            if self.chromosome[x] == ord(target[x]):
                self.fitness += 1

    def mutate(self, mutation_rate: float = 0.1) -> None:
        test = random.random()
        if test <= mutation_rate:
            value = random.choice(self.chromosome)
            index = self.chromosome.index(value)
            value = random.randrange(ASCII[0], ASCII[-1])
            self.chromosome[index] = value


class Population(object):
    """
        a population of Individual possible phrases.
    """
    def __init__(self, popsize: int = 12, target: str = 't3sT', xover_function: str = None) -> None:
        self.popsize = popsize
        self.target = target
        self.target_length = len(target)
        self.population = self.init_population()
        if xover_function is not None:
            self.xover_function_name = xover_function
        else:
            self.xover_function_name = 'keep_fittest'

        self.update_fitness()
        self.sort_by_fitness()

    def init_population(self) -> list:
        """
            Returns a list of Individual() objects.
        """
        temp = []
        for x in range(self.popsize):
            temp.append(Individual(self.target_length))
        return temp

    def update_fitness(self) -> None:
        for x in self.population:
            x.fitness = 0
            x.get_fitness(self.target)

    def sort_by_fitness(self) -> None:
        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def xover_function(self) -> None:
        xover_name = f'xover_{self.xover_function_name}'
        Crossover(self, xover_name)


class GAPhrase:
    """
        The genetic algorithm.
    """
    def __init__(self, target='T3st', popsize=100, mutation_rate=0.1, xover=None, acceptance=1.0):
        self.generation = 0
        self.target = target
        self.target_length = len(self.target)
        self.popsize = popsize
        self.mutation_rate = mutation_rate
        self.xover = xover
        self.acceptance_value = acceptance * self.target_length
        self.pop = Population(popsize=self.popsize, target=self.target, xover_function=self.xover)

    def init(self):
        self.generation = 1
        self.pop.update_fitness()
        self.pop.sort_by_fitness()

    def run(self):
        print(f'\nGAPhrase: Match a target phrase via evolution.\n***********************************************\n')
        self.init()
        letters = self.pop.population[0].output_chromosome()
        fittest = self.pop.population[0].fitness
        print(f'\rGeneration: {self.generation:<5}\tTarget: "{self.target}"  \tFittest: "{letters}"'
              f'  \tFitness: {fittest}', end='')
        while self.pop.population[0].fitness < self.acceptance_value and self.generation < 10000:
            self.generation += 1
            self.pop.xover_function()
            for individual in self.pop.population:
                individual.mutate(self.mutation_rate)
            self.pop.update_fitness()
            self.pop.sort_by_fitness()
            letters = self.pop.population[0].output_chromosome()
            fittest = self.pop.population[0].fitness
            print(f'\rGeneration: {self.generation:<5}\tTarget: "{self.target}"  \tFittest: "{letters}"'
                  f'  \tFitness: {fittest}', end='')


class Crossover(object):
    """
        Crossover functions container.
    """
    def __init__(self, pop: Population = None, xover: str = None) -> None:
        self.pop = pop
        self.xover = getattr(self, xover)
        self.xover()

    def xover_keep_fittest(self) -> None:
        """
            Fitness function uses fittest half of population to create breeding pairs,
            generates a number of children equal to (roughly if population not even number) one
            quarter of the total population size.  Re-evaluates fitness, sorts, and 'kills' (removes)
            least fit quarter of the population.  Popsize remains the same after a generation.
        """
        self.pop.update_fitness()
        self.pop.sort_by_fitness()
        children = []

        if self.pop.popsize % 2 != 0:
            breeding_pairs = int((self.pop.popsize - 1) / 4)
        else:
            breeding_pairs = int(self.pop.popsize / 4)

        for x in range(0, breeding_pairs, 2):
            temp_child = []
            for y in range(self.pop.target_length):
                if y % 2 == 0:
                    temp_child.append(self.pop.population[x].chromosome[y])
                else:
                    temp_child.append(self.pop.population[x+1].chromosome[y])
            children.append(temp_child)

        for child in children:
            self.pop.population.append(Individual(self.pop.target_length, child))

        self.pop.update_fitness()
        self.pop.sort_by_fitness()

        while len(self.pop.population) > self.pop.popsize:
            self.pop.population.pop()

    def xover_comp(self) -> None:
        """
            Fitness function randomly select three possible parents from the population
            and the two fittest reproduce while the least fit 'dies' (is removed).  Population
            size remains the same after a generation.
        """
        random.shuffle(self.pop.population)
        children = []

        for index in range(0, len(self.pop.population), 3):
            if (len(self.pop.population) - index) < 3:
                break
            else:
                temp_child = []
                parents = [self.pop.population[index], self.pop.population[index+1], self.pop.population[index+2]]
                parents.sort(key=lambda x: x.fitness, reverse=True)
                parents[2].fitness = -1
                for x in range(self.pop.target_length):
                    if x % 2 == 0:
                        temp_child.append(parents[0].chromosome[x])
                    else:
                        temp_child.append(parents[1].chromosome[x])
                children.append(temp_child)

        for child in children:
            self.pop.population.append(Individual(self.pop.target_length, child))

        for individual in self.pop.population:
            if individual.fitness != -1:
                individual.get_fitness(self.pop.target)

        self.pop.sort_by_fitness()

        while len(self.pop.population) > self.pop.popsize:
            self.pop.population.pop()


if __name__ == '__main__':
    ga = GAPhrase(target='Tristan McGuire', popsize=200, mutation_rate=0.25, xover='comp', acceptance=1.0)
    ga.run()
    print(f'\nEnd.')
