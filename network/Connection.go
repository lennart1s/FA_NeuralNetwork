package network

/* Connection stellt eine Verbindung
zu einem Neuron dar. Sowohl die
Gewichtung als auch für bestimmte
Trainingsverfahren benötgte Werte die
der Gradient-Wert oder die letzte 
Gewichtungs-Veränderung werden verwaltet.*/
type Connection struct {
	/* Neuron ist ein Pointer auf
	das Verbundene Neuron.*/
	*Neuron //`json:"-"`
 
	//ConnectedNeuronId UUID.UUID

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
	Gradient     float64
}
