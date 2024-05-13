import numpy as np


einstellungen = {
    'gui':                          False,                           # GUI aktivieren, verlangsamt das Training kritisch
    'spielfeldgroesse':             (10, 10),                       # Fuer erhoehte Schwierigkeit spielfeldgroesse veraendern (10-100)   Bsp: (50, 50)

    #STATS'
    'daten_aufzeichnen':            True,                            # Gibt Daten in .txt aus, verlangsamt Training deutlich
    'ausgabe_individuum':           False,                             # Gibt Informationen zum aktuellen Individuum in Konsole aus, verlangsamt Training deutlich

    ##SAVE##
    # Snake wird unter ./savegenname/loadname gespeichert
    'speichern':                    True,
    'savegenname': '16Inp',                      # Speicherverzeichniss, wird erstellt

    # Speichern wenn *alle* Savekriterien erfuellt sind
    # Savekriterien:
    'savescore': 70,                                                # Speichert nur Snakes mit score > savescore &
    'savefitness': 0,                                               # Speichert nur Snakes mit fitness > savefitness &
    'savegeneration': 0,                                            # Speichert nur Snakes mit generation > savegeneration

    # LOAD#
    'replay':                       False,                          # Eine Snake soll geladen werden -> Einstellungen unter #LOAD# beachten
    'loadgenname': 'best',                                          # Vorhandenes Speicherverzeichniss aus dem eine Snake geladen werden soll
    'loadname': 'G250_ID:15_Score:38_Fitness:274878878587.32886',
    # Name der Snake (Verzeichnisname der Snake innerhalb des Speicherverzeichniss)




    'crossover':                    1,                              # Crossover 1 = One Point Crossover; 2 = Uniform Crossover

    'anz_eltern':                   300,                            # Anzahl der Eltern in neuem Generationspool
    'anz_kinder':                   900,                            # Anzahl der Kinder in neuem Generationspool

    'mutationsrate':                0.1,                            # Mutationsrate (muss Wert zwischen 0.00 - 1.00 haben, hohe Werte sind nicht Sinnvoll)
    'mutationsrate_nimmt_ab':       False,                          # bei True mutationsrate = mutationsrate / sqrt(self.akt_gen + 1)


}