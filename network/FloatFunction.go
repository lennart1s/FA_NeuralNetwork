package network

import "math"

/*FloatFunction bescheibt eine Funktion,#
welche einen float64-Parameter hat und
einen float64 zurückgibt.
Sie wird verwendet um die Aktivierungsfunktion
oder deren Ableitung für ein neuronales Netz
darzustellen.*/
type FloatFunction func(float64) float64

// Beispiele:

/*Sigmoid ist der Name einer häufig genutzen
Aktivierungsfunktion für neuronale Netwerke.
Sie beschreibt eine Art S-Form und gibt
ausschließlich Werte zwischen 0 und 1 zurück.*/
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

/*SigmoidDeriv ist die Ableitung der Sigmoid-
Fuktion, welche für das Trainieren eines Netzwerkes
benötigt werden kann.*/
func SigmoidDeriv(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}
