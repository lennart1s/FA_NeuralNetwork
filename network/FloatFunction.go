package network

import "math"

// FloatFunction : Eine Typ-Definition für funktionen welche einen Float-Wert als Parameter benötigen
// und einen Wert in abhängigkeit des Parameters ausgeben.
type FloatFunction func(float64) float64

/*func GetDefaults() map[string]FloatFunction {
	var m map[string]FloatFunction

	m["Sigmoid"] = Sigmoid
	m["SigmoidDeriv"] = SigmoidDeriv

	return m
}*/

// Beispiele:

// Sigmoid : Die Sigmoid-funktion; gibt ausschließlich Werte zwischen 0 und 1 zurück;
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x)) // eine häufig genutze 'Aktivierungsfunktion' für die Berechnung mit einem KNN
}

// SigmoidDeriv : Die Ableitung der Sigmoidfunkion; wird zum Trainieren des Netzwerkes benötigt
func SigmoidDeriv(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}
