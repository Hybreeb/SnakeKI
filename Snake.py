import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List, Callable, NewType
from fractions import Fraction
import random
from collections import deque
import sys
import os
import json
from GenAlgo import Individuum


# Winkel fuer die Sicht
class Winkel(object):
    __slots__ = ('a', 'b')

    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


# Koordinaten
class Koordinate(object):
    __slots__ = ('x', 'y')

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    # Definiere das Ansprechverhalten der Klasse
    def __eq__(self, alt: Union['Koordinate', Tuple[int, int]]) -> bool:
        if isinstance(alt, tuple) and len(alt) == 2:
            return alt[0] == self.x and alt[1] == self.y
        elif isinstance(alt, Koordinate) and self.x == alt.x and self.y == alt.y:
            return True
        return False

    def __sub__(self, alt: Union['Koordinate', Tuple[int, int]]) -> 'Koordinate':
        if isinstance(alt, tuple) and len(alt) == 2:
            diff_x = self.x - alt[0]
            diff_y = self.y - alt[1]
            return Koordinate(diff_x, diff_y)
        elif isinstance(alt, Koordinate):
            diff_x = self.x - alt.x
            diff_y = self.y - alt.y
            return Koordinate(diff_x, diff_y)
        return None

    def __rsub__(self, alt: Tuple[int, int]):
        diff_x = alt[0] - self.x
        diff_y = alt[1] - self.y
        return Koordinate(diff_x, diff_y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self) -> str:
        return '({}, {})'.format(self.x, self.y)

    def copy(self) -> 'Koordinate':
        x = self.x
        y = self.y
        return Koordinate(x, y)


# Entfernungen
class Entfernung(object):
    __slots__ = ('ent_zur_wand', 'ent_zum_essen', 'ent_zu_self')

    def __init__(self,
                 ent_zur_wand: Union[float, int],
                 ent_zum_essen: Union[float, int],
                 ent_zu_self: Union[float, int]
                 ):
        self.ent_zur_wand = float(ent_zur_wand)
        self.ent_zum_essen = float(ent_zum_essen)
        self.ent_zu_self = float(ent_zu_self)


class Sicht(object):
    __slots__ = ('ort_wand', 'ort_essen', 'ort_self')

    def __init__(self,
                 ort_wand: Koordinate,
                 ort_essen: Optional[Koordinate] = None,
                 ort_self: Optional[Koordinate] = None,
                 ):
        self.ort_wand = ort_wand
        self.ort_essen = ort_essen
        self.ort_self = ort_self


class Snake(Individuum):
    def __init__(self, spielfeldgroesse: Tuple[int, int],
                 chromosom: Optional[Dict[str, List[np.ndarray]]] = None,
                 start_pos: Optional[Koordinate] = None,
                 initial_v: Optional[str] = None,
                 start_richtung: Optional[str] = None,
                 zwischen_schicht_architektur: Optional[List[int]] = [16, 8]):

        self.score = 0
        self._fitness = 0
        self._frames = 0
        self._frames_seit_letztem_essen = 0
        self.moegliche_richtung = ('u', 'd', 'l', 'r')

        self.spielfeldgroesse = spielfeldgroesse
        self.zwischen_schicht_architektur = zwischen_schicht_architektur

        # Startposition initialisieren
        if not start_pos:
            x = random.randint(2, self.spielfeldgroesse[0] - 3)
            y = random.randint(2, self.spielfeldgroesse[1] - 3)

            start_pos = Koordinate(x, y)
        self.start_pos = start_pos

        # Sicht und Entfernung initialisieren
        self._umsicht = Winkel(-1, 0), Winkel(0, 1), Winkel(1, 0), Winkel(0, -1)  # Hoch, Rechts, Runter, Links
        self._entfernung: List[Entfernung] = [None] * len(self._umsicht)
        self._sicht: List[Sicht] = [None] * len(self._umsicht)

        # Neuronales Netzwerk initialisieren
        anz_inputs = 16
        self.entfernung_als_array: np.ndarray = np.zeros((anz_inputs, 1)) # in diesem mehrdimensionalen Array werden später die Werte der Neuronen weitergegeben
        self.netzwerk_architektur = [anz_inputs]  # Inputs
        self.netzwerk_architektur.extend(self.zwischen_schicht_architektur)  # Zwischenschicht
        self.netzwerk_architektur.append(4)  # 4 Outputs ['u', 'd', 'l', 'r']
        self.netzwerk = NeuronalesNetzwerk(self.netzwerk_architektur)

        if chromosom:
            self.netzwerk.parameter = chromosom

        # Naechstes Essen generieren
        essen_seed = np.random.randint(-1000000000, 1000000000)
        self.essen_seed = essen_seed
        self.rand_essen = random.Random(self.essen_seed)
        self.ort_essen = None

        if start_richtung:
            start_richtung = start_richtung[0]
        else:
            start_richtung = self.moegliche_richtung[random.randint(0, 3)]

        self.start_richtung = start_richtung
        self.init_snake(self.start_richtung)
        self.initial_v = initial_v
        self.init_v(self.start_richtung, self.initial_v)
        self.generiere_essen()

    def berechne_fitness(self):
        # Fitness:
        # - Gefundenes Essen besonders belohnen
        # - Schritte bestrafen ??
        # - Fitness > 0
        a = self.score
        x = self._frames
        self._fitness = (100 * (2 ** a) + x) - 0.5 * x ** 1.15
        #self._fitness= (self._frames) + ((2 ** self.score)) + (self.score) * 500
        #self._fitness= ((self._frames) + ((2 ** self.score)) + (self.score) * 500) - (self._frames_seit_letztem_essen) * (self.score)
        # self._fitness = (((2 ** self.score) + (self.score * 100)) + (self._frames)) - (self._frames) * (self.score ** 1.1)
        #self._fitness = (self._frames) + ((2 ** self.score) + (self.score ** 2.1) * 500) - (((.25 * self._frames) ** 1.3) * (self.score ** 1.2))
        # self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)) * (self.score))

        # Fitness muss mindestens 0.1 Betragen
        self._fitness = max(self._fitness, .1)

    def umsehen(self):

        # Umsehen

        for i, winkel in enumerate(self._umsicht):
            entfernung, sicht = self.schaue_in_richtung(winkel)
            self._entfernung[i] = entfernung
            self._sicht[i] = sicht

        self._entfernung_als_input_array()

    def schaue_in_richtung(self, winkel: Winkel) -> Tuple[Entfernung, Sicht]:
        ent_zur_wand = None
        ent_zum_essen = np.inf
        ent_zu_self = np.inf

        ort_wand = None
        ort_essen = None
        ort_self = None

        position = self.snake_array[0].copy()
        distanz = 1.0
        ges_distanz = 0.0

        # Verhindert ungueltige Startpositionen
        position.x += winkel.b
        position.y += winkel.a
        ges_distanz += distanz
        self_gefunden = False
        essen_gefunden = False

        # Wurde etwas gefunden?
        while self._im_spielfeld(position):
            if not self_gefunden and self._schlange_gefunden(position):
                ent_zu_self = ges_distanz
                ort_self = position.copy()
                self_gefunden = True
            if not essen_gefunden and self._essen_gefunden(position):
                ent_zum_essen = ges_distanz
                ort_essen = position.copy()
                essen_gefunden = True

            ort_wand = position
            position.x += winkel.b
            position.y += winkel.a
            ges_distanz += distanz
        assert (ges_distanz != 0.0)

        ent_zur_wand = 1.0 / ges_distanz
        ent_zum_essen = 1.0 / ent_zum_essen
        ent_zu_self = 1.0 / ent_zu_self
        entfernung = Entfernung(ent_zur_wand, ent_zum_essen, ent_zu_self)
        sicht = Sicht(ort_wand, ort_essen, ort_self)
        return (entfernung, sicht)

    def _entfernung_als_input_array(self) -> None:
        # Definiere Input-Schicht als Array
        for va_index, v_index in zip(range(0, len(self._entfernung) * 3, 3), range(len(self._entfernung))):
            entfernung = self._entfernung[v_index]
            self.entfernung_als_array[va_index, 0] = entfernung.ent_zur_wand
            self.entfernung_als_array[va_index + 1, 0] = entfernung.ent_zum_essen
            self.entfernung_als_array[va_index + 2, 0] = entfernung.ent_zu_self

        # 3 Entfernungen pro Richtung
        i = len(self._entfernung) * 3

        ## Fuege 4 Bewegungsinputs des Kopfes hinzu (u, d, r, l)
        # Immer nur eine Bewegungsrichtung moeglich (One Hot)
        richtung = self.richtung[0]
        richtung_one_hot = np.zeros((len(self.moegliche_richtung), 1))
        richtung_one_hot[self.moegliche_richtung.index(richtung), 0] = 1
        self.entfernung_als_array[i: i + len(self.moegliche_richtung)] = richtung_one_hot

        i += len(self.moegliche_richtung)
        """## Fuege 4 Bewegungsinputs des Schwanzes (letzte Kachel)
        # Immer nur eine Schwanzrichtung moeglich TODO: SCHLAFEN GEHEN
        schwanz_richtung_one_hot = np.zeros((len(self.moegliche_richtung), 1))
        schwanz_richtung_one_hot[self.moegliche_richtung.index(self.schwanz_richtung), 0] = 1
        self.entfernung_als_array[i: i + len(self.moegliche_richtung)] = schwanz_richtung_one_hot"""

        # Ueberpruefen ob die Snake im gueltigen Bereich ist

    def _im_spielfeld(self, position: Koordinate) -> bool:
        return position.x >= 0 and position.y >= 0 and \
               position.x < self.spielfeldgroesse[0] and \
               position.y < self.spielfeldgroesse[1]

    def generiere_essen(self) -> None:
        width = self.spielfeldgroesse[0]
        height = self.spielfeldgroesse[1]

        # Wenn keine weiteres Essen generiert werden kann ist hat die Snake gewonnen
        freierplatz = [divmod(i, height) for i in range(width * height) if divmod(i, height) not in self._ort_koerper]
        if freierplatz:
            ort = self.rand_essen.choice(freierplatz)
            self.ort_essen = Koordinate(ort[0], ort[1])
        else:
            print("Eine Snake hat die Aufgabe erfuellt!")
            pass

    def init_snake(self, start_richtung: str) -> None:
        ## Initialisere die Schlange##

        kopf = self.start_pos
        if start_richtung == 'u':
            snake = [kopf, Koordinate(kopf.x, kopf.y + 1), Koordinate(kopf.x, kopf.y + 2),
                     Koordinate(kopf.x, kopf.y + 3)]
        elif start_richtung == 'd':
            snake = [kopf, Koordinate(kopf.x, kopf.y - 1), Koordinate(kopf.x, kopf.y - 2),
                     Koordinate(kopf.x, kopf.y - 3)]
        elif start_richtung == 'l':
            snake = [kopf, Koordinate(kopf.x + 1, kopf.y), Koordinate(kopf.x + 2, kopf.y),
                     Koordinate(kopf.x + 3, kopf.y)]
        elif start_richtung == 'r':
            snake = [kopf, Koordinate(kopf.x - 1, kopf.y), Koordinate(kopf.x - 2, kopf.y),
                     Koordinate(kopf.x - 3, kopf.y)]

        self.snake_array = deque(snake)
        self._ort_koerper = set(snake)
        self.am_leben = True

    def update(self):
        if self.am_leben:
            self._frames += 1
            self.umsehen()
            self.netzwerk.weiterleiten(self.entfernung_als_array)
            self.richtung = self.moegliche_richtung[np.argmax(self.netzwerk.out)] # der Index des groesten Datensatz
            return True
        else:
            return False

    def bewegen(self, spielfeldgroesse: Tuple[int, int], ) -> bool:
        if not self.am_leben:
            return False

        richtung = self.richtung[0]

        ## Output Neuronen werden Ausgewertet ##
        head = self.snake_array[0]
        if richtung == 'u':
            next_pos = Koordinate(head.x, head.y - 1)
        elif richtung == 'd':
            next_pos = Koordinate(head.x, head.y + 1)
        elif richtung == 'r':
            next_pos = Koordinate(head.x + 1, head.y)
        elif richtung == 'l':
            next_pos = Koordinate(head.x - 1, head.y)

        # Kollisionsueberpruefung und Bewegungsanimation
        if self._darf_betreten(next_pos):
            if next_pos == self.snake_array[-1]:
                self.snake_array.pop()
                self.snake_array.appendleft(next_pos)

            elif next_pos == self.ort_essen:
                self.score += 1
                self._frames_seit_letztem_essen = 0

                self.snake_array.appendleft(next_pos)
                self._ort_koerper.update({next_pos})
                self.generiere_essen()
            else:
                self.snake_array.appendleft(next_pos)
                self._ort_koerper.update({next_pos})
                schwanz = self.snake_array.pop()
                self._ort_koerper.symmetric_difference_update({schwanz})

            """# Initialisierung der Schwanzrichtung Inputs
            p2 = self.snake_array[-2]
            p1 = self.snake_array[-1]
            diff = p2 - p1
            if diff.x < 0:
                self.schwanz_richtung = 'l'
            elif diff.x > 0:
                self.schwanz_richtung = 'r'
            elif diff.y > 0:
                self.schwanz_richtung = 'd'
            elif diff.y < 0:
                self.schwanz_richtung = 'u'"""

            self._frames_seit_letztem_essen += 1

            # Maximale Schritte zwischen Essen
            max_frames = (spielfeldgroesse[0] * spielfeldgroesse[1]) * 0.7
            if self._frames_seit_letztem_essen > max_frames:
                # Snake verhungert
                self.am_leben = False
                return False

            return True
        else:
            # Wand beruehrt, Snake tot.
            self.am_leben = False
            return False

    def _essen_gefunden(self, position: Koordinate) -> bool:
        return position == self.ort_essen

    def _schlange_gefunden(self, position: Koordinate) -> bool:
        return position in self._ort_koerper

    def _darf_betreten(self, position: Koordinate) -> bool:
        if (position.x < 0) or (position.x > self.spielfeldgroesse[0] - 1):
            return False
        if (position.y < 0) or (position.y > self.spielfeldgroesse[1] - 1):
            return False
        if position == self.snake_array[-1]:
            return True
        elif position in self._ort_koerper:
            return False

        else:
            return True

    def init_v(self, start_richtung, initial_v: Optional[str] = None) -> None:
        # Initialisiere Startrichtung
        if initial_v:
            self.richtung = initial_v[0]
        else:
            self.richtung = start_richtung
        #self.schwanz_richtung = self.richtung


def save_snake(population_ordner: str, individuum_name: str, snake: Snake, settings: Dict[str, Any]) -> None:
    # Schlange Speichern, siehe ReadMe
    if not os.path.exists(population_ordner):
        os.makedirs(population_ordner)

    # Speicher Einstellungen
    if 'einstellungen.json' not in os.listdir(population_ordner):
        d = os.path.join(population_ordner, 'einstellungen.json')
        with open(d, 'w', encoding='utf-8') as out:
            json.dump(settings, out, sort_keys=True, indent=4)

    individuum_dir = os.path.join(population_ordner, individuum_name)
    if not os.path.exists((individuum_dir)):
        os.makedirs(individuum_dir)

    # Speicher Chromosonen aufgeschluesselt in weights und bias
    L = len(snake.netzwerk.neuronen)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = snake.netzwerk.parameter[w_name]
        bias = snake.netzwerk.parameter[b_name]

        np.save(os.path.join(individuum_dir, w_name), weights)
        np.save(os.path.join(individuum_dir, b_name), bias)


def load_snake(population_ordner: str, individuum_name: str,
               einstellungen: Optional[Union[Dict[str, Any], str]] = None) -> Snake:
    if not einstellungen:
        d = os.path.join(population_ordner, 'einstellungen.json')

        with open(d, 'r', encoding='utf-8') as fp:
            einstellungen = json.load(fp)

    elif isinstance(einstellungen, dict):
        einstellungen = einstellungen

    elif isinstance(einstellungen, str):
        dateipath = einstellungen
        with open(dateipath, 'r', encoding='utf-8') as fp:
            einstellungen = json.load(fp)

    parameter = {}
    for dname in os.listdir(os.path.join(population_ordner, individuum_name)):
        ext = dname.rsplit('.npy', 1)
        if len(ext) == 2:
            param = ext[0]
            parameter[param] = np.load(os.path.join(population_ordner, individuum_name, dname))
        else:
            continue

    snake = Snake(einstellungen['spielfeldgroesse'], chromosom=parameter,
                  zwischen_schicht_architektur=[16, 8])
    return snake


class NeuronalesNetzwerk(object):
    def __init__(self, neuronen: List[int]):
        self.parameter = {}
        self.neuronen = neuronen
        self.rand = np.random.RandomState()

        # Initialisiere Weights und Bias
        for l in range(1, len(self.neuronen)):
            self.parameter['W' + str(l)] = np.random.uniform(-1, 1, size=(self.neuronen[l], self.neuronen[l - 1]))
            self.parameter['b' + str(l)] = np.random.uniform(-1, 1, size=(self.neuronen[l], 1))

            self.parameter['A' + str(l)] = None

    # Leite Input durch Neuronenschichten
    def weiterleiten(self, X: np.ndarray) -> np.ndarray:
        a_vor = X
        s1 = len(self.neuronen) - 1

        # Aktivierungsfunktionen #
        Aktivierungsfunktion = NewType('Aktivierungsfunktion',
                                       Callable[[np.ndarray], np.ndarray])  # TODO -> Einstellungen

        sigmoid = Aktivierungsfunktion(lambda X: 1.0 / (1.0 + np.exp(-X)))
        tanh = Aktivierungsfunktion(lambda X: np.tanh(X))
        relu = Aktivierungsfunktion(lambda X: np.maximum(0, X))
        leaky_relu = Aktivierungsfunktion(lambda X: np.where(X > 0, X, X * 0.01))
        linear = Aktivierungsfunktion(lambda X: X)

        # Weiterleitung innerhalb der Zwischenschichten per relu Aktivierungsfunktion
        for s2 in range(1, s1):
            W = self.parameter['W' + str(s2)]
            b = self.parameter['b' + str(s2)]
            Z = np.dot(W, a_vor) + b
            a_vor = relu(Z)  # Aktivierungsfunktion relu aktiviert nur passende Werte, gibt unpassende Werte mit 0 zurück
            self.parameter['A' + str(s2)] = a_vor

        # Output Weiterleiten per sigmoid Aktivierungsfunktion
        W = self.parameter['W' + str(s1)]
        b = self.parameter['b' + str(s1)]
        Z = np.dot(W, a_vor) + b
        out = sigmoid(Z)  # Aktivierungsfunktion sigmoid aktiviert nur passende Werte, gibt unpassende Werte mit 0 zurück
        self.parameter['A' + str(s1)] = out


        self.out = out
        return out










