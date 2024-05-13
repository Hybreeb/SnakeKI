import numpy as np
from abc import abstractmethod
from typing import Optional, Union, Tuple, List
import random

######### GENETISCHER ALGORITHMUS #########

class Individuum(object):
    def __init__(self):
        pass
    ### Chormosome ###
    ## Beinhalten die Werte des Individuums ##

    # Gibt Chromosom zurueck
    @property
    @abstractmethod
    def chromosom(self):
        raise Exception('Abstrakte Methoden muessen definiert werden!')

    @chromosom.setter
    def chromosom(self, val):
        raise Exception('Abstrakte Methoden muessen definiert werden! Setter Error')

    ### Fitness ###
    ## Numerischer Einordnung der Leistung ##

    # Fitness Berechnen
    @abstractmethod
    def berechne_fitness(self):
        raise Exception('Abstrakte Methoden muessen definiert werden!')

    # Gibt Fitness zurueck
    @property
    @abstractmethod
    def fitness(self):
        return self._fitness

    @fitness.setter
    @abstractmethod
    def fitness(self, val):
        raise Exception('Abstrakte Methoden muessen definiert werden! Setter Error')



class Population(object):
    def __init__(self, individuen: List[Individuum]):
        self.individuen = individuen

    #Anzahl der Individuen Ausgeben
    @property
    def anz_individuen(self) -> int:
        return len(self.individuen)

    @anz_individuen.setter
    def anz_individuen(self, val) -> None:
        raise Exception('Setter Error')


    #Durchschnittliche Fitness Berechnen und Ausgeben
    @property
    def durch_fitness(self) -> float:
        return (sum(individuum.fitness for individuum in self.individuen) / float(self.anz_individuen))

    @durch_fitness.setter
    def durch_fitness(self, val) -> None:
        raise Exception('Setter Error')

    # Durchschnittlichen Score Berechnen und Ausgeben

    @property
    def durch_score(self) -> float:
            return (sum(individuum.score for individuum in self.individuen) / float(self.anz_individuen))

    @durch_score.setter
    def durch_score(self, val) -> None:
            raise Exception('Setter Error')

    #Bestes Individuum (Fitness)
    @property
    def beste_fitness_individuum(self) -> Individuum:
        return max(self.individuen, key=lambda individuum: individuum.fitness)

    @beste_fitness_individuum.setter
    def beste_fitness_individuum(self, val) -> None:
        raise Exception('Setter Error')


##### CROSSOVER #####

def simulated_binary_crossover(vater: np.ndarray, mutter: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eta = 100
    rand = np.random.random(vater.shape)
    gamma = np.empty(vater.shape)

    gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))
    gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))

    kind1 = 0.5 * ((1 + gamma) * vater + (1 - gamma) * mutter)
    kind2 = 0.5 * ((1 - gamma) * vater + (1 + gamma) * mutter)

    return kind1, kind2

def single_point_binary_crossover(vater: np.ndarray, mutter: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    kind1 = vater.copy()
    kind2 = mutter.copy()

    reihen, zeilen = mutter.shape
    reihe = np.random.randint(0, reihen)
    zeile = np.random.randint(0, zeilen)


    kind1[:reihe, :] = mutter[:reihe, :]
    kind2[:reihe, :] = vater[:reihe, :]

    kind1[reihe, :zeile + 1] = mutter[reihe, :zeile + 1]
    kind2[reihe, :zeile + 1] = vater[reihe, :zeile + 1]


    return kind1, kind2

def uniform_binary_crossover(vater: np.ndarray, mutter: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    offspring1 = vater.copy()
    offspring2 = mutter.copy()

    mask = np.random.uniform(0, 1, size=offspring1.shape)
    offspring1[mask > 0.5] = mutter[mask > 0.5]
    offspring2[mask > 0.5] = vater[mask > 0.5]

    return offspring1, offspring2

##### MUTATION #####

def gaussian_mutation(chromosom: np.ndarray, mutationsrate: float) -> None:
    mutation_array = np.random.random(chromosom.shape) < mutationsrate
    gaussian_mutation = np.random.normal(size=chromosom.shape)
    gaussian_mutation[mutation_array] /= 5
    chromosom[mutation_array] += gaussian_mutation[mutation_array]




def elite_selektion(population: Population, anz_individuen: int) -> List[Individuum]:

    individuen = sorted(population.individuen, key = lambda individuum: individuum.fitness, reverse=True)
    return individuen[:anz_individuen]

def roulette_selektion(population: Population, anz_individuen: int) -> List[Individuum]:

    selektion = []
    roulette = sum(individuum.fitness for individuum in population.individuen)
    for _ in range(anz_individuen):
        ausw = random.uniform(0, roulette)
        akt = 0
        for individuum in population.individuen:
            akt += individuum.fitness
            if akt > ausw:
                selektion.append(individuum)
                break

    return selektion

