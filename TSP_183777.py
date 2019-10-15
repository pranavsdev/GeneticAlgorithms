

"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random
from Individual import *
import sys

myStudentNum = 183777 # Replace 12345 with your student number
random.seed(myStudentNum)

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
        self.initPopulation()


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
            #print(self.data[int(id)])
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        #print("self best>>>",self.__dict__)
        print ("Best initial sol: ",self.best.getFitness())

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        #print(indA, indB)
        return [indA, indB]

    def stochasticUniversalSampling(self):
        """
        Your stochastic universal sampling Selection Implementation
        """
        pass

    def uniformCrossover(self, indA, indB):

        """
        Your Uniform Crossover Implementation
        """
        randomIndex = [2,4,6,7]
        childA = [] 
        childB = []

        tempA = [] 
        tempB = []
        #Ranomly select positions that will no longer change
        """ randombits = self.genSize/2
        for i in range(0, randombits):
            randomIndex+ = random.random() """
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
            if b not in randomIndex:
                if indA.genes[j] not in tempB and indA.genes[j] not in childB:
                    childB.insert(b, indA.genes[j])
                    b += 1
                else:
                    j += 1
            else:
                if a in randomIndex:
                    childA.insert(a, indA.genes[a])
                    a += 1                     
                if b in randomIndex:
                    childB.insert(b, indB.genes[b])
                    b += 1    

        print("children")
        print(childA, childB)


    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation
        """
        tempA = []
        tempB = []

        childA = []
        childB = []

        strip = [3, 4, 5]
        relA = {}
        relB = {}

        for i in range(0, len(strip)):
            index = strip[i]
            print("index>>", index)
            tempB.insert(index, indA.genes[index])
            tempA.insert(index, indB.genes[index])
            A = indA.genes[index]
            B = indB.genes[index]
            relB[A] = B
            relA[B] = A

        print("tempA, tempB>>>")
        print(tempA,tempB)

        print("relA>>>",relA)
        print("relB>>>",relB)

        j = 0
        for i in range(0, self.genSize):
            if i not in strip:
                if indA.genes[i] not in tempA and indA.genes[i] not in childA:
                    childA.insert(i, indA.genes[i])
                else:
                    value = relA[indA.genes[i]]
                    while j < len(relA):
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
                    while j < len(relB):
                        if value not in childB and value not in tempB:
                            childB.insert(i, value)
                            break
                        else:
                            value = relB[value]
                        j +=1
            elif i in strip:
                childA.insert(i, indB.genes[i])
                childB.insert(i, indA.genes[i])

        print("children>>>")
        print(childA, childB)
        return [childA, childB]

    def reciprocalExchangeMutation(self, ind):
        """
        Your Reciprocal Exchange Mutation implementation
        """
        pass

    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        print("random>>",random.random())
        print("mutation rate>>>",self.mutationRate)
        if random.random() > self.mutationRate:
            return

        indexStart = random.randint(0, self.genSize-1)
        indexEnd = random.randint(0, self.genSize-1)

        temp = ind[indexStart:indexEnd+1]
        temp.reverse()


        j = 0
        for i in range(indexStart, indexEnd+1):
            ind[i] = temp[j]
            j += 1
        
        print("ind>>>",ind)
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
        print("random>>>",random.random())
        print("mutation rate",self.mutationRate)

        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        """     tmp = ind.genes[indexA]
                ind.genes[indexA] = ind.genes[indexB]
                ind.genes[indexB] = tmp
        """
        tmp = ind[indexA]
        ind[indexA] = ind[indexB]
        ind[indexB] = tmp

        print("ind>>>",ind)
        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append( ind_i.copy() )

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

        for i in range(0, 3):
        #print("length of population>>>",len(self.population))
        #for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            #random selection
            Parent = self.randomSelection()
            print("parent A: ",Parent[0].genes)
            #print(Parent[1].__dict__)
            print("parent B: ",Parent[1].genes)
            
            #cross over
            child = self.pmxCrossover(Parent[0],Parent[1])
            #print("child: ",child)
            
            #Mutation
            print("mutation step>>>")
            self.inversionMutation(child[0])
            
            print ("Total iterations: ",self.iteration)
            print ("Best Solution: ", self.best.getFitness())

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
            self.GAStep()
            self.iteration += 1

"""         print ("Total iterations: ",self.iteration)
        print ("Best Solution: ", self.best.getFitness()) """

if len(sys.argv) < 2:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)


problem_file = sys.argv[1]

# inputs: File Name, Popuation Size, Mutation Rate, Max Iterations

ga = BasicTSP(sys.argv[1], 17, 0.9, 1)
ga.readInstance()
ga.initPopulation()
ga.search()