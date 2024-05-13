# Aufbau einer künstlichen Intelligenz

Dieses Repository enthält eine implementierte künstliche Intelligenz, die das Computerspiel Snake spielt. Die künstliche Intelligenz wurde mithilfe eines neuronalen Netzwerks trainiert, das mit einem genetischen Algorithmus optimiert wurde.

## Applikationsaufbau

### Ressourcen

- **Python**: Python wurde als Hauptprogrammiersprache für dieses Projekt verwendet. Python bietet eine einfache Lesbarkeit und Übersichtlichkeit, was es ideal für die Entwicklung von Machine Learning und Data Science Anwendungen macht.
  
- **Numpy**: Numpy wurde für verbesserte Interaktionen und Berechnungen mit Arrays, Listen und Matrizen verwendet. Aufgrund seiner mathematischen Funktionen ist Numpy besonders beliebt für Machine Learning und Data Science Anwendungen.
  
- **PyQt**: PyQt wurde für die Umsetzung grafischer Benutzeroberflächen (GUIs) verwendet. Es basiert auf dem Qt-Framework der Programmiersprache C++ und ermöglicht die einfache Erstellung dynamischer GUIs in Python.

### Grafisches Benutzer Interface (GUI)

Die GUI besteht aus drei Ebenen: das neuronale Netzwerk, das simulierte Umfeld und die Anzeige der aktuellen Statistik. Das simulierte Umfeld basiert auf dem Computerspiel Snake.

### Simuliertes Umfeld

Das simulierte Umfeld ist das Computerspiel Snake, das seit 1997 auf Nokia Mobiltelefonen verbreitet ist. Snake bietet eine realitätsnahe Problemstellung und eine stetig anwachsende Schwierigkeit im Spielverlauf.

### Neuronales Netzwerk

Das neuronale Netzwerk dient der digitalen Entscheidungsfindung. Es besteht aus künstlichen Neuronen und neuronalen Verbindungen, die in mehreren Schichten angeordnet sind. Die Architektur des neuronalen Netzwerks wurde an die Problemstellung angepasst.

### Genetischer Algorithmus

Der genetische Algorithmus orientiert sich an evolutionären Erfolgsstrategien und wurde verwendet, um das neuronale Netzwerk zu optimieren. Er simuliert die natürliche Selektion, Kreuzung und Mutation, um die Performance des neuronalen Netzwerks zu verbessern.

### Einstellungen

Unter Einstellungen.py können verschiedene Parameter angepasst werden, um die Ausführung der Applikation zu steuern.

### Speichern und Laden

Individuen können gespeichert und geladen werden, um den Trainingsprozess zu unterbrechen und später fortzusetzen.

**Hinweis**: Die GUI und das Speichern von Statistiken und Schlangen können die Performance der Applikation beeinträchtigen. Die GUI kann den Lernprozess zusätzlich verzögern.

Für weitere Details und Informationen zur Implementierung siehe die entsprechenden Dokumentationen im Projekt.
