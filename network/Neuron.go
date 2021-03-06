package network

import (
	UUID "github.com/google/uuid"
)

/*Neuron stellt ein einzelnes Neuron
in einem Netzwerk dar. Die eingehenden
Verbindungen mit anderen Neuronen wie
Werte wie die letzte Eingabe und die
letzte Ausgabe werden verwaltet.*/
type Neuron struct {
	/*Id ist die einamlige ID von
	einem Neuron.*/
	ID UUID.UUID

	/*Type beschreibt die Art des
	Neurons und wie es folglich
	Verwendet und berechnet wird.
	Unterschieden wird zwischen den
	vier Typen: INPUT, HIDDEN, BIAS
	und OUTPUT.*/
	Type int

	/*Conns ist das Slice in dem die
	eingehenden Verbindungen von anderen
	Neuron gespeichert werden.*/
	Conns []*Connection

	/*Input speichert die letzte Gesamt-
	eingabe. Diese ergibt sich aus der Summe
	der Produkte aus Ausgabe und Gewichtung
	aller eingehenden Werte.*/
	Input float64
	/*Output speichert den letzten Ausgabewert.
	Dieser brechnet sich aus der verwendeten
	Aktivierungsfunktion in abhängigkeit von der
	Gesamteingabe.*/
	Output float64

	/*Processed kann als Anhaltpunkt bei sämtlichen
	Prozessen verwendet werden ob ein bestimmter
	Schritt bereits abgeschlossen wurde.
	Dieses Attribut wird weder vom Neuron selbst
	noch vom Netzwerk verwendet.*/
	Processed bool

	/*CalculatedGradients ist ein Indikator ob
	in der momentanen Trainings-Iteration schon
	die Gradienten der Verbindungen berechnet
	wurden.*/
	CalculatedGradients bool

	/*ChangedWeights ist ein Indikator ob in der
	momentanen Trainings-Iteration die Gweichtungen
	schon angepasst wurden.*/
	ChangedWeights bool

	/*Delta speichert den zuletzt berechneten
	Deltawert. Dieser wird zum Trainieren eines
	Netzwerkes mit Methoden, welche auf der
	Delta-Regel beruhen, verwendet.*/
	Delta float64
	/*PrevLayerWeightedDelta ist die Summe
	der Produkte aus allen Delta-Werten von
	Neuronen, die mit diesem Neuron verbunden
	sind, und der Gewichtung dieser Verbindung.*/
	PrevLayerWeightedDelta float64
}

/*NewNeuron erstell ein neues Neuron vom
gegebenen Typ mit einer einzigartigen ID.*/
func NewNeuron(Type int) Neuron {
	id, _ := UUID.NewUUID()
	n := Neuron{Type: Type, ID: id}
	return n
}

/*ConnectTo verbindet das Neuron
mit einem gegebenen anderen Neuron.*/
func (n *Neuron) ConnectTo(o *Neuron) {
	n.Conns = append(n.Conns, &Connection{Neuron: o, ConnectedNeuronID: o.ID})
}

/*Die verschiedenen Neuron-Typen*/
const (
	INPUT  = 0
	HIDDEN = 1
	BIAS   = 2
	OUTPUT = 3
)

/*Process berechnet die Ausgabe des
Neurons mit einer gegebenen Aktivierungsfunktion.*/
func (n *Neuron) Process(activFunc FloatFunction) {
	if n.Type == INPUT {
		n.Output = n.Input
	} else if n.Type == BIAS {
		n.Output = 1
	} else {
		n.Input = 0
		for _, conn := range n.Conns {
			// Wenn nicht schon passiert, berechne alle eingehenden Ausgaben anderer Neuronen
			if !conn.Neuron.Processed {
				conn.Neuron.Process(activFunc)
			}
			n.Input += conn.Weight * conn.Neuron.Output
		}

		n.Output = activFunc(n.Input)
	}
	n.Processed = true
}

/*UnsetProcessed setzt für das Neuron und alle
Neuronen eingehender Verbindungen den
Processed-Indikator auf false.*/
func (n *Neuron) UnsetProcessed() {
	n.Processed = false
	for _, c := range n.Conns {
		c.Neuron.UnsetProcessed()
	}
}

/*UnsetCalculatedGradients setzt für das Neuron und
alle Neuronen eingehender Verbindungen den
Calculated-Gradients-Indikator auf false.
Wenn zeroValues == true werden alle Gradient-Werte auf
0 gesetzt.*/
func (n *Neuron) UnsetCalculatedGradients(zeroValues bool) {
	n.CalculatedGradients = false
	for _, con := range n.Conns {
		if zeroValues {
			con.Gradient = 0
		}
		con.Neuron.UnsetCalculatedGradients(zeroValues)
	}
}

/*UnsetChangedWeights setzt für das Neuron und alle
Neuronen eingehender Verbindungen den
ChangedWeights-Indikator auf false.*/
func (n *Neuron) UnsetChangedWeights() {
	n.ChangedWeights = false
	for _, con := range n.Conns {
		con.Neuron.UnsetChangedWeights()
	}
}

/*ZeroPrevLayerWeightedDelta setzt für das Neuron
und alle Neuronen eingehender Verbindungen den
PrevLayerWeightedDelta-Wert auf 0.*/
func (n *Neuron) ZeroPrevLayerWeightedDelta() {
	n.PrevLayerWeightedDelta = 0
	for _, con := range n.Conns {
		con.Neuron.ZeroPrevLayerWeightedDelta()
	}
}
