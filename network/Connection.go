package network

import UUID "github.com/google/uuid"

/* Connection stellt eine Verbindung
zu einem Neuron dar. Sowohl die
Gewichtung als auch für bestimmte
Trainingsverfahren benötgte Werte die
der Gradient-Wert oder die letzte
Gewichtungs-Veränderung werden verwaltet.*/
type Connection struct {
	/* Neuron ist ein Pointer auf
	das verbundene Neuron.*/
	*Neuron `json:"-"`

	/* ConnectedNeuronID ist die ID des
	verbundenen Neurons*/
	ConnectedNeuronID UUID.UUID

	/* Weight ist der Wert der Gewichtung
	der Verbindung.*/
	Weight float64

	/* WeightChange speichert die letzte
	Änderung der Gewichtung und wird
	beispielsweise beim Backpropagation-
	Verfahren verwendet.*/
	WeightChange float64
	/* Gradient ist ein Ableitungswert der
	Gewicht-Fehler-Funktion dieser Verindung.*/
	Gradient float64
}
