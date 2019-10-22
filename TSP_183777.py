"""
Author: Pranav Srivastava
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random
from Individual import *
import sys
import time

random.seed(time.clock())

myStudentNum = 183777 # Replace 12345 with your student number
#seed value is dependent on student id
#random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}

        self.readInstance()
        #self.initPopulation()
        
        self.children = []
        self.selectedchildren = []
        self.configuration = {}

    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            print("fitness>>",individual.fitness)
            print("chromosome>>",individual.genes)
            self.population.append(individual)

        self.best = self.population[0].copy()
        #print("best>", self.best.__dict__)
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()

    def initNewPopulation(self):
        """
        Creating new population from the children of last generation
        """
        #remove previous population
        self.population = []

        for i in range(0, len(self.selectedchildren)):
            individual = Individual(self.genSize, self.data)
            individual.setGene(self.selectedchildren[i])
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ",self.best.getFitness())

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        rand1 =  random.randint(0, self.popSize-1)
        rand2 =  random.randint(0, self.popSize-1)

        while rand1 == rand2:
            rand2 =  random.randint(0, self.popSize-1)
        print("random>>>",rand1)

        indA = self.matingPool[ rand1 ]
        indB = self.matingPool[ rand2 ]
        #print(indA, indB)
        return [indA, indB]

    def stochasticUniversalSampling(self, ind):
        """
        Your stochastic universal sampling Selection Implementation
        """
        self.selectedchildren = []
        individuals = []
        for i in range(0, len(ind)):
            individual = Individual(self.genSize, self.data)
            individual.setGene(ind[i])
            individual.computeFitness()
            individuals.append(individual)
            print("individuals>>>", individuals[i].fitness)
        #F is the sum of the fitness values of all chromosomes in the population
        F = 0

        # N is the number of parents to select
        N = self.popSize

        #P is the distance between successive points
        P = 0
        Pointers = []
        Ruler = []
        point = 0
        pickedpop = []

        fractionOfTotal = 0
        #calculate total fitness of the population
        for i in range(0,len(individuals)):
            F = F + individuals[i].fitness

        Ruler.append(fractionOfTotal)
        for i in range(0, len(individuals)):            
            fractionOfTotal = fractionOfTotal + individuals[i].fitness/F
            Ruler.append(fractionOfTotal)
        print("ruler", Ruler)              

        P = 1/N
        print("P___",P)

        point = random.uniform(0,P)


        for i in range(0, N):
            for j in range(0, len(Ruler)):
                if point >= Ruler[j] and point <= Ruler[j+1]:
                    self.selectedchildren.append(individuals[j].genes)
            point = point + P
            print("point>>>",point)
 
        print("picked pop>", self.selectedchildren)
        
    def uniformCrossover(self, indA, indB):

        """
        Your Uniform Crossover Implementation
        """
        randomIndex = []
        childA = [] 
        childB = []

        tempA = [] 
        tempB = []
        #Ranomly select positions that will no longer change

        rand = 0
        for i in range(0, self.genSize):
            rand = random.randint(rand, self.genSize-1)
            if rand not in randomIndex:
                randomIndex.append(rand)

        for i in range(0, len(randomIndex)):
            index = randomIndex[i]
            tempA.insert(index, indA.genes[index])
            tempB.insert(index, indB.genes[index])
        print("tempA, tempB>>>")
        print(tempA,tempB)

        a = 0
        b = 0
        i = 0
        j = 0
        while i < self.genSize and j < self.genSize:
            if a not in randomIndex:
                if indB.genes[i] not in tempA and indB.genes[i] not in childA:
                    childA.insert(a, indB.genes[i])
                    a += 1
                else:
                    i += 1
            else:
                if a in randomIndex:
                    childA.insert(a, indA.genes[a])
                    a += 1          
            if b not in randomIndex:
                if indA.genes[j] not in tempB and indA.genes[j] not in childB:
                    childB.insert(b, indA.genes[j])
                    b += 1
                else:
                    j += 1
            else:
                if b in randomIndex:
                    childB.insert(b, indB.genes[b])
                    b += 1    

        #print("children")
        #print(childA, childB)
        return [childA, childB]

    def randomStrip(self):
        strip = []
        start =  random.randint(0, self.genSize-1)
        end = random.randint(start, self.genSize-1)
        for i in range (start, end+1):
            strip.append(i)
        return strip

    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation
        """
        tempA = []
        tempB = []

        childA = []
        childB = []

        strip = []
        strip = self.randomStrip()
        #print("strip>>>",strip)

        relA = {}
        relB = {}

        for i in range(0, len(strip)):
            index = strip[i]
            tempB.insert(index, indA.genes[index])
            tempA.insert(index, indB.genes[index])
            A = indA.genes[index]
            B = indB.genes[index]
            relB[A] = B
            relA[B] = A

        j = 0
        k = 0
        for i in range(0, self.genSize):
            if i not in strip:
                if indA.genes[i] not in tempA and indA.genes[i] not in childA:
                    childA.insert(i, indA.genes[i])
                else:
                    value = relA[indA.genes[i]]
                    while j < len(relA)+1:
                        if value not in childA and value not in tempA:
                            childA.insert(i, value)
                            break
                        else:
                            value = relA[value]
                        j +=1
                if indB.genes[i] not in tempB and indB.genes[i] not in childB:
                    childB.insert(i, indB.genes[i])
                else:
                    value = relB[indB.genes[i]]
                    while k < len(relB)+1:
                        if value not in childB and value not in tempB:
                            childB.insert(i, value)
                            break
                        else:
                            value = relB[value]
                        k +=1
            elif i in strip:
                childA.insert(i, indB.genes[i])
                childB.insert(i, indA.genes[i])

        #print("children>>>")
        #print(childA, childB)
        return [childA, childB]

    def reciprocalExchangeMutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """

        if random.random() > self.mutationRate:
            return

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind[0].genes[indexA]
        ind[0].genes[indexA] = ind[0].genes[indexB]
        ind[0].genes[indexB] = tmp

        ind[0].computeFitness()
        self.updateBest(ind[0])
        return ind[0].genes

    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        #if random.random() > self.mutationRate:
        #    return

        #start index of the strip
        indexStart = random.randint(0, self.genSize-1)

        #end index of the strip
        indexEnd = random.randint(indexStart, self.genSize-1)

        temp = ind[0].genes[indexStart:indexEnd+1]
        #reverse the strip
        temp.reverse()

        #update the chromose with the reversed strip
        j = 0
        for i in range(indexStart, indexEnd+1):
            ind[0].genes[i] = temp[j]
            j += 1

        return ind[0].genes
        #ind.computeFitness()
        #self.updateBest(ind)

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        #print("child>>>",child)
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            #print("call from mating pool")
            self.matingPool.append( ind_i.copy() )
        print("size of mating pool>>",len(self.matingPool))

    def newGeneration(self):
        #replacing current population with a new one
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        Parent = []
        child = []

        self.mates = []
        crossedChild = []
        children = []
        #for i in range(0, 3):
        #print("length of population>>>",len(self.population))
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            #random selection
            Parent = self.randomSelection()
            #print("parent A: ",Parent[0].genes)
            
            #cross over
            if ga.configuration[config]["crossover"] == "pmx":
                child = self.pmxCrossover(Parent[0],Parent[1])
            elif ga.configuration[config]["crossover"] == "ox":
                child = self.uniformCrossover(Parent[0],Parent[1])
            print("child: ",child)
            
            
            #Mutation
            #print("mutation step>>>")
            for childIndex in range(0,len(child)):
                ind = []
                individual = Individual(self.genSize, self.data)
                individual.setGene(child[childIndex])
                ind.append(individual)

                crossedChild.append(child[childIndex])
                mutatedChild = []
                
                if ga.configuration[config]["mutation"] == "reciex":
                    mutatedChild = self.reciprocalExchangeMutation(ind)
                elif ga.configuration[config]["mutation"] == "inversion":
                    mutatedChild = self.inversionMutation(ind)     

                if mutatedChild not in children and mutatedChild is not None:
                    children.append(mutatedChild)

        if len(children) == 0 or len(children)<len(self.population):
            for i in range(0, len(self.population)):
                if len(children) < len(self.population):
                    children.append(crossedChild[i])


        print("new children>>>", children)
        if ga.configuration[config]["selection"] == "random":
        #this is to select the children randomly
            self.randomChildSelection(children)
        elif ga.configuration[config]["selection"] == "sus":
            self.stochasticUniversalSampling(children)
        self.initNewPopulation()

    def randomChildSelection(self, children):
        rand = 0
        self.selectedchildren = []
        while len(self.selectedchildren) < self.popSize:
            print("fuzzy rand", rand)
            if children[rand] not in self.selectedchildren:
                self.selectedchildren.append(children[rand])
                rand = random.randint(0, self.popSize-1)
            else:
                rand1 = random.randint(1, self.popSize-1)
                rand = rand1
        print("selected children>>>",self.selectedchildren)

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            print("generation>>>",self.iteration)
            self.GAStep()
            self.iteration += 1


        print ("Total iterations: ",self.iteration)
        print ("Best Solution: ____________", self.best.getFitness())

if len(sys.argv) < 2:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)


problem_file = sys.argv[1]

# inputs: File Name, Popuation Size, Mutation Rate, Max Iterations

configchoice = input("plese enter the configuration number you want to run...")
config = int(configchoice)

ga = BasicTSP(sys.argv[1], 4, 0.1, 100)

ga.configuration[1] =  {"initalsol":"random", "crossover":"ox", "mutation":"inversion", "selection":"random"}
ga.configuration[2] =  {"initalsol":"random", "crossover":"pmx", "mutation":"reciex", "selection":"random"}
ga.configuration[3] =  {"initalsol":"random", "crossover":"ox", "mutation":"reciex", "selection":"sus"}
ga.configuration[4] =  {"initalsol":"random", "crossover":"ox", "mutation":"reciex", "selection":"sus"}
ga.configuration[5] =  {"initalsol":"random", "crossover":"pmx", "mutation":"inversion", "selection":"sus"}
ga.configuration[6] =  {"initalsol":"random", "crossover":"ox", "mutation":"inversion", "selection":"sus"}
ga.configuration[7] =  {"initalsol":"heuristic", "crossover":"pmx", "mutation":"reciex", "selection":"sus"}
ga.configuration[8] =  {"initalsol":"heuristic", "crossover":"ox", "mutation":"inversion", "selection":"sus"}

#only initialize the population randomly when picking the specific configuration
if ga.configuration[config]["initalsol"] == "random":
    ga.initPopulation()

ga.search()
print(ga.configuration[config]["initalsol"])
