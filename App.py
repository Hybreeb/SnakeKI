from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
import sys
from typing import List
from Snake import *
import numpy as np
from Einstellungen import einstellungen
from GenAlgo import Population
from GenAlgo import elite_selektion, roulette_selektion
from GenAlgo import gaussian_mutation
from GenAlgo import simulated_binary_crossover
from GenAlgo import single_point_binary_crossover
from GenAlgo import uniform_binary_crossover
from math import sqrt
from decimal import Decimal
import random
import csv


class Spielfeld(QtWidgets.QMainWindow):
    def __init__(self, einstellungen):
        gui = einstellungen['gui']
        super().__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(240, 240, 240))
        self.setPalette(palette)
        self.einstellungen = einstellungen
        self._mutationsrate = self.einstellungen['mutationsrate']

        self._next_gen_groesse = self.einstellungen['anz_eltern'] + self.einstellungen['anz_kinder']

        self.spielfeldgroesse = einstellungen['spielfeldgroesse']
        aufloesung = 1000 / einstellungen['spielfeldgroesse'][0], 1000 / einstellungen['spielfeldgroesse'][1]
        self.border = 10
        self.spielfeldbreite = aufloesung[0] * self.spielfeldgroesse[0]
        self.spielfeldhoehe = aufloesung[1] * self.spielfeldgroesse[1]
        self.top = 0
        self.left = 0
        self.width = self.spielfeldbreite + self.border + self.border
        self.height = self.spielfeldhoehe + self.border + self.border

        individuen: List[Individuum] = []

        if einstellungen['replay'] == True:
            for _ in range(self.einstellungen['anz_eltern']):
                loadname = einstellungen['loadname']
                loadgenname = einstellungen['loadgenname']
                individuum = load_snake(loadgenname, loadname, einstellungen)
                individuen.append((individuum))
        elif einstellungen['replay'] == False:
            for _ in range(self.einstellungen['anz_eltern']):
                individuum = Snake(self.spielfeldgroesse, zwischen_schicht_architektur=[16, 8])
                individuen.append(individuum)

        self.top_fitness = 0
        self.top_score = 0
        self.akt_individuum = 0
        self.population = Population(individuen)
        self.snake = self.population.individuen[self.akt_individuum]
        self.akt_gen = 0

        self.init_fenster()

        if gui == True:
            ticks = 200
        elif gui == False:
            ticks = 1 / 100000  #Ticks kleiner = schnelleres Training, Bottlenecked von CPU

        # Timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)

        self.timer.start(ticks)

        # Einstellungen 'gui'
        if gui:
            self.show()

    def init_fenster(self):
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('K. I.   S N A K E')
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Initialisiere Neuronales Netzwerk Container
        self.nn_widget_container = NeuronalesNetzwerkWidget(self.centralWidget, self.snake)
        self.nn_widget_container.setGeometry(QtCore.QRect(self.border, self.border, self.spielfeldbreite + self.border,
                                                          self.spielfeldhoehe + self.border))
        self.nn_widget_container.setObjectName('nn_widget_container')

        # Initialisiere Snake Container
        self.snake_widget_container = SnakeWidget(self.centralWidget, self.spielfeldgroesse, self.snake)
        self.snake_widget_container.setGeometry(
            QtCore.QRect(self.border, self.border, self.spielfeldbreite, self.spielfeldhoehe))
        self.snake_widget_container.setObjectName('snake_widget_container')

        # Initialisiere Stats Container
        self.ga_container = Stats(self.centralWidget, self.einstellungen)
        self.ga_container.setGeometry(
            QtCore.QRect(self.border, self.border, self.spielfeldbreite, self.spielfeldhoehe))
        self.ga_container.setObjectName('stats')

    def update(self) -> None:
        self.snake_widget_container.update()
        self.nn_widget_container.update()
        # Aktuelle snake ist am leben
        if self.snake.am_leben:
            self.snake.bewegen(self.spielfeldgroesse)
            if self.snake.score > self.top_score:
                self.top_score = self.snake.score
                self.ga_container.top_score_label.setText(str(self.snake.score))
        # Aktuelle snake ist tot
        else:
            self.snake.berechne_fitness()
            fitness = self.snake.fitness
            score = self.snake.score
            generation = self.akt_gen
            akt_id = self.akt_individuum

            if fitness > self.top_fitness:
                self.top_fitness = fitness

            # Ausgabe aktuelle Snake an Konsole
            iausgabe = einstellungen['ausgabe_individuum']
            if iausgabe:
                print("Gen:", generation, "ID:", self.akt_individuum, "Score:", score, "Fitness:", fitness)

            # Speichern aktuelle Snake
            savegenname = einstellungen['savegenname']
            savescore = einstellungen['savescore']
            savefitness = einstellungen['savefitness']
            savegeneration = einstellungen['savegeneration']

            if score >= savescore and fitness >= savefitness and generation >= savegeneration:
                savename = "G" + str(generation) + "_ID" + str(akt_id) + "_Score" + str(score) + "_Fitness" + str(
                    fitness)
                save_snake(savegenname, str(savename), self.snake, self.einstellungen)

            # Naechste Snake im Pool
            self.akt_individuum += 1
            # Wenn keine mehr vorhanden dann
            # Naechste Generation
            if (self.akt_gen > 0 and self.akt_individuum == self._next_gen_groesse) or (
                    self.akt_gen == 0 and self.akt_individuum == einstellungen['anz_eltern']):

                # Ausgabe in Konsole

                print("--->")
                print(" Generation {}:".format(self.akt_gen))
                print(" Beste Fitness: ", self.population.beste_fitness_individuum.fitness,
                      "\n Bester Score: ", self.population.beste_fitness_individuum.score,
                      "\n Durchschnittliche Fitness: ", self.population.durch_fitness)
                print("--->")

                d_aufzeichnen = einstellungen['daten_aufzeichnen']
                # TODO: Ausgabe in 1 Datei
                if d_aufzeichnen:
                    # Ausgabe in Dateien
                    d_beste_fitness = open("beste_fitness.txt", "a")
                    d_beste_fitness.write(str(self.population.beste_fitness_individuum.fitness) + "\n")
                    d_beste_fitness.close()

                    d_bester_score = open("bester_score.txt", "a")
                    d_bester_score.write(str(self.population.beste_fitness_individuum.score) + "\n")
                    d_bester_score.close()

                    d_durch_fitness = open("durch_fitness.txt", "a")
                    d_durch_fitness.write(str(self.population.durch_fitness) + "\n")
                    d_durch_fitness.close()

                    d_durch_score = open("durch_score.txt", "a")
                    d_durch_score.write(str(self.population.durch_score) + "\n")
                    d_durch_score.close()
                self.naechste_generation()
            else:
                akt_population = self.einstellungen['anz_eltern'] if self.akt_gen == 0 else self._next_gen_groesse
                self.ga_container.akt_individual_label.setText('{}/{}'.format(self.akt_individuum + 1, akt_population))

            self.snake = self.population.individuen[self.akt_individuum]
            self.snake_widget_container.snake = self.snake
            self.nn_widget_container.snake = self.snake

    def naechste_generation(self):
        self.generation_erhoehen()
        # Setze auf 0
        self.akt_individuum = 0
        # Berechne die Fitness des Pools
        for individuum in self.population.individuen:
            individuum.berechne_fitness()
        # Waehle beste Individuen
        self.population.individuen = elite_selektion(self.population, self.einstellungen['anz_eltern'])
        random.shuffle(self.population.individuen)

        naechste_population: List[Snake] = []

        for individuum in self.population.individuen:
            parameter = individuum.netzwerk.parameter
            spielfeldgroesse = individuum.spielfeldgroesse
            zwischen_schicht_architektur = individuum.zwischen_schicht_architektur

            s = Snake(spielfeldgroesse, chromosom=parameter, zwischen_schicht_architektur=zwischen_schicht_architektur)
            naechste_population.append(s)

        # Paare Snakes bis genug Kinder vorhanden
        while len(naechste_population) < self._next_gen_groesse:
            vati, mutti = roulette_selektion(self.population, 2)

            L = len(vati.netzwerk.neuronen)
            k1_parameter = {}
            k2_parameter = {}

            for l in range(1, L):
                vati_W_l = vati.netzwerk.parameter['W' + str(l)]
                mutti_W_l = mutti.netzwerk.parameter['W' + str(l)]
                vati_b_l = vati.netzwerk.parameter['b' + str(l)]
                mutti_b_l = mutti.netzwerk.parameter['b' + str(l)]

                # Crossover
                k1_W_l, k2_W_l, k1_b_l, k2_b_l = self._crossover(vati_W_l, mutti_W_l, vati_b_l, mutti_b_l)

                # Mutation
                self._mutation(k1_W_l, k2_W_l, k1_b_l, k2_b_l)

                # Weise Kinder zu
                k1_parameter['W' + str(l)] = k1_W_l
                k2_parameter['W' + str(l)] = k2_W_l
                k1_parameter['b' + str(l)] = k1_b_l
                k2_parameter['b' + str(l)] = k2_b_l

                # Begrenze auf [-1, 1]
                np.clip(k1_parameter['W' + str(l)], -1, 1, out=k1_parameter['W' + str(l)])
                np.clip(k2_parameter['W' + str(l)], -1, 1, out=k2_parameter['W' + str(l)])
                np.clip(k1_parameter['b' + str(l)], -1, 1, out=k1_parameter['b' + str(l)])
                np.clip(k2_parameter['b' + str(l)], -1, 1, out=k2_parameter['b' + str(l)])

            # Erstelle Kinder und fuege hinzu
            k1 = Snake(vati.spielfeldgroesse, chromosom=k1_parameter,
                       zwischen_schicht_architektur=vati.zwischen_schicht_architektur)
            k2 = Snake(mutti.spielfeldgroesse, chromosom=k2_parameter,
                       zwischen_schicht_architektur=mutti.zwischen_schicht_architektur)
            naechste_population.extend([k1, k2])

        random.shuffle(naechste_population)
        self.population.individuen = naechste_population

    def generation_erhoehen(self):
        self.akt_gen += 1
        self.ga_container.akt_generation_label.setText(str(self.akt_gen + 1))

    def _crossover(self, vater_weights: np.ndarray, mutter_weights: np.ndarray,
                   vater_bias: np.ndarray, mutter_bias: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        child1_weights, child2_weights = None, None
        child1_bias, child2_bias = None, None


        if self.einstellungen['crossover'] == 1:
            # One Point Crossover
            child1_weights, child2_weights = single_point_binary_crossover(vater_weights, mutter_weights)
            child1_bias, child2_bias = single_point_binary_crossover(vater_bias, mutter_bias)
        elif self.einstellungen['crossover'] == 2:
            # Uniform Crossover
            child1_weights, child2_weights = uniform_binary_crossover(vater_weights, mutter_weights)
            child1_bias, child2_bias = uniform_binary_crossover(vater_bias, mutter_bias)
        else:
            print("Einstellungen: Crossover Fehler ")

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:

        # Mutationsrate faellt ab
        mutationsrate = self._mutationsrate
        if self.einstellungen['mutationsrate_nimmt_ab'] == True:
            mutationsrate = mutationsrate / sqrt(self.akt_gen + 1)

        # Mutation von Weight und Bias
        gaussian_mutation(child1_weights, mutationsrate)
        gaussian_mutation(child2_weights, mutationsrate)

        gaussian_mutation(child1_bias, mutationsrate)
        gaussian_mutation(child2_bias, mutationsrate)


class Stats(QtWidgets.QWidget):
    def __init__(self, eltern, settings):
        super().__init__(eltern)
        font = QtGui.QFont('Monospace', 16, QtGui.QFont.Bold)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)
        ursprung = Qt.AlignLeft | Qt.AlignTop
        lcol = 0
        scol = 1
        row = 0

        ### Stats Anzeigen
        # Generation
        self._erstelle_lwidget_raster('Generation: ', font, grid, row, lcol, ursprung)
        self.akt_generation_label = self._erstelle_lwidget('1', font)
        grid.addWidget(self.akt_generation_label, row, scol, ursprung)
        lcol = 2
        scol = 3

        # Aktuelle Snake
        self._erstelle_lwidget_raster('Individuum: ', font, grid, row, lcol, ursprung)
        self.akt_individual_label = self._erstelle_lwidget('1/{}'.format(settings['anz_eltern']), font)
        grid.addWidget(self.akt_individual_label, row, scol, ursprung)
        lcol = 4
        scol = 5

        # Bester Score
        self._erstelle_lwidget_raster('Bester Score: ', font, grid, row, lcol, ursprung)
        self.top_score_label = self._erstelle_lwidget('0', font)
        grid.addWidget(self.top_score_label, row, scol, ursprung)

        # Layout

        grid.setSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 1)

        self.setLayout(grid)

        self.show()

    def _erstelle_lwidget(self, string_label: str, font: QtGui.QFont) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel()
        label.setText(string_label)
        label.setFont(font)
        label.setContentsMargins(0, 0, 0, 0)
        return label

    def _erstelle_lwidget_raster(self, string_label: str, font: QtGui.QFont,
                                 grid: QtWidgets.QGridLayout, row: int, col: int,
                                 alignment: Qt.Alignment) -> None:
        label = QtWidgets.QLabel()
        label.setText(string_label)
        label.setFont(font)
        label.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(label, row, col, alignment)


class SnakeWidget(QtWidgets.QWidget):
    def __init__(self, eltern, spielfeldgroesse=(50, 50), snake=None):
        super().__init__(eltern)
        self.spielfeldgroesse = spielfeldgroesse
        if snake:
            self.snake = snake
        self.setFocus()
        self.show()

    def update(self):
        if self.snake.am_leben:
            self.snake.update()
            self.repaint()

    # Zeichne Spielfeld
    def zeichne(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(Qt.black))
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        painter.setPen(QtCore.Qt.black)
        painter.drawLine(0, 0, width, 0)
        painter.drawLine(width, 0, width, height)
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)
        aufloesung = 1000 / einstellungen['spielfeldgroesse'][0], 1000 / einstellungen['spielfeldgroesse'][1]

        ort_essen = self.snake.ort_essen
        if ort_essen:
            painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
            painter.setPen(QtGui.QPen(Qt.black))
            painter.setBrush(QtGui.QBrush(Qt.lightGray))

            painter.drawRect(ort_essen.x * aufloesung[0],
                             ort_essen.y * aufloesung[1],
                             aufloesung[0],
                             aufloesung[1])

        for koordinate in self.snake.snake_array:
            if koordinate == self.snake.snake_array[0]:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 160, 10)))
                painter.drawRect(koordinate.x * aufloesung[0],
                                 koordinate.y * aufloesung[1],
                                 aufloesung[0],
                                 aufloesung[1])
                painter.setBrush(QtGui.QBrush(QtGui.QColor(15, 15, 15)))
                painter.drawRect(koordinate.x * aufloesung[0] + (aufloesung[0] * 0.25),
                                 koordinate.y * aufloesung[1] + (aufloesung[1] * 0.2),
                                 aufloesung[0] * 0.1,
                                 aufloesung[1] * 0.1)
                painter.drawRect(koordinate.x * aufloesung[0] + (aufloesung[0] * 0.75),
                                 koordinate.y * aufloesung[1] + (aufloesung[1] * 0.2),
                                 aufloesung[0] * 0.1,
                                 aufloesung[1] * 0.1)
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(10, 120, 10)))
                painter.drawRect(koordinate.x * aufloesung[0],
                                 koordinate.y * aufloesung[1],
                                 aufloesung[0],
                                 aufloesung[1])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)
        self.zeichne(painter)
        painter.end()


class NeuronalesNetzwerkWidget(QtWidgets.QWidget):
    def __init__(self, eltern, snake: Snake):
        super().__init__(eltern)
        self.snake = snake
        self.h_offset_schichten = 50
        self.v_offset_neuronen = 10
        self.max_schicht = max(self.snake.netzwerk.neuronen)
        self.neuron_pos = {}
        self.show()

    # Zeichne Neuronales Netzwerk
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)
        self.zeichne_netzwerk(painter)
        painter.end()

    def update(self) -> None:
        self.repaint()

    def zeichne_netzwerk(self, painter: QtGui.QPainter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        radius = 14
        height = self.frameGeometry().height()
        width = self.frameGeometry().width()
        neuronen = self.snake.netzwerk.neuronen
        nnsaturation = 50
        standard_offset = 30
        h_offset = standard_offset
        inputs = self.snake.entfernung_als_array
        out = self.snake.netzwerk.weiterleiten(inputs)
        max_out = np.argmax(out)

        # Zeichne Neuronen
        for schicht, anz_neuronen in enumerate(neuronen):
            v_offset = (height - ((2 * radius + 18) * anz_neuronen)) / 2
            aktivierungen = None
            if schicht > 0:
                aktivierungen = self.snake.netzwerk.parameter['A' + str(schicht)]

            for neuron in range(anz_neuronen):
                x = h_offset
                y = neuron * (radius * 2 + 18) + v_offset
                t = (schicht, neuron)
                if t not in self.neuron_pos:
                    self.neuron_pos[t] = (x, y + radius)

                painter.setBrush(QtGui.QBrush(Qt.black, Qt.NoBrush))

                # Input Schicht

                if schicht == 0:
                    if inputs[neuron, 0] > 0:
                        painter.setBrush(QtGui.QColor(5, 120, 5, nnsaturation))
                    else:
                        painter.setBrush(QtGui.QColor(0, 0, 0, 0))

                # Zwischen Schichten

                elif schicht > 0 and schicht < len(neuronen) - 1:
                    try:
                        saturation = max(min(aktivierungen[neuron, 0], 1.0), 0.0)
                    except:

                        import sys
                        sys.exit(-1)
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 66, saturation * 100)))

                # Output Schicht
                elif schicht == len(neuronen) - 1:
                    if neuron == max_out:
                        painter.setBrush(QtGui.QColor(5, 120, 5, nnsaturation))
                    else:
                        painter.setBrush(QtGui.QColor(0, 0, 0, 0))

                painter.drawEllipse(x, y, radius * 2, radius * 2)
            h_offset += 300

        # Zeichne Weights

        for l in range(1, len(neuronen)):
            weights = self.snake.netzwerk.parameter['W' + str(l)]
            vor_neuronen = weights.shape[1]
            akt_neuronen = weights.shape[0]
            for vor_neuron in range(vor_neuronen):

                for akt_neuron in range(akt_neuronen):
                    # Bei Aktivierung aufleuchten
                    if weights[akt_neuron, vor_neuron] > 0:
                        painter.setPen(QtGui.QColor(140, 0, 0, nnsaturation))
                    else:
                        painter.setPen(QtGui.QColor(100, 100, 100, nnsaturation * 0.7))

                    start = self.neuron_pos[(l - 1, vor_neuron)]
                    end = self.neuron_pos[(l, akt_neuron)]
                    painter.drawLine(start[0] + radius + radius, start[1], end[0], end[1])


## Hier startet die APP ##

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    fenster = Spielfeld(einstellungen)
    sys.exit(app.exec_())