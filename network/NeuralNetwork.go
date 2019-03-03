package network

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"math/rand"
	"os"
	"time"
)

/*Neural Netowork ist eine Strultur aus Neuronen
und Verbindungen zwischen diesen. Außerdem werden
Informationen wie die Aktivierungsfunktion, deren
Ableitung und auch, für spezifische Trainings-
Algorithmen benötigte, Daten verwaltet.*/
type NeuralNetwork struct {
	/*Inputs ist ein Slice, welches Pointer
	zu allen Eingabe-Neuronen des Netwerkes
	speichert um einen einfacheren Zugriff
	zu gewährleisten.*/
	Inputs []*Neuron `json:"-"`
	/*Outputs ist ein Slice, welches Pointer
	zu allen Ausgabe-Neuronen des Netwerkes
	speichert um einen einfacherern Zugriff
	zu gewährleisten.*/
	Outputs []*Neuron `json:"-"`

	/*ActivFunc ist die Aktivierungsfunktion,
	welche beim Berechnen mithilfe des Netzes
	verwendet wird.
	Sie wird als FloatFunction gespeichert.*/
	ActivFunc FloatFunction `json:"-"`
	/*ActivDeriv ist die Ableitung der
	Aktivierungsfunktion, welche beim Trainieren
	des Netzes mit der Gradient-Descent-Methode
	verwendet wird.
	Sie wird als FloatFunction gespeichert.*/
	ActivDeriv FloatFunction `json:"-"`

	/*Neurons ist ein Slice zur speicherung aller
	Neuronen im Netzwerk. Dieses Slice wird zum
	Speichern und Laden des Netzwerkes verwendet*/
	Neurons []*Neuron

	/*NumLayers speichert die Anzahl der Schichten
	in einem schicht-basiertem Netzwerk.*/
	NumLayers int

	/*BackPropRuns speichert die Anzahl an Durchläufen
	der Backpropagation-Trainings-Methode.*/
	BackPropRuns int
	/*Fitness speichert den Fitness-Wert des Netzwerkes.
	Dieser wird beispielsweise bei evolutionären
	Trainings-Methoden genutzt.*/
	Fitness float64
}

/*Run berechnet die Ausgabe des Netzwerkes mit
gegebene Eingaben.*/
func (nn *NeuralNetwork) Run(input []float64) []float64 {
	for _, out := range nn.Outputs {
		out.UnsetProcessed()
	}
	for i := 0; i < len(nn.Inputs); i++ {
		nn.Inputs[i].Input = input[i]
	}
	var output []float64
	for _, out := range nn.Outputs {
		out.Process(nn.ActivFunc)
		output = append(output, out.Output)
	}
	return output
}

/*CreateLayered erstellt ein vollständig verbundenes
Feed-Forward-NN. Die erste Layer-größe beschriebt die
Anzahl anEingabe-Neuronen und die letze die der
Ausgabe-Neuronen.*/
func (nn *NeuralNetwork) CreateLayered(layerSizes []int, useBiases bool) {
	nn.Outputs = nn.Outputs[:0]
	nn.ActivFunc = Sigmoid
	nn.ActivDeriv = SigmoidDeriv

	nn.NumLayers = len(layerSizes)
	for l := len(layerSizes) - 1; l >= 0; l-- {
		parentLayer, _ := nn.GetLayer(l + 1)
		for i := 0; i < layerSizes[l]; i++ {
			n := NewNeuron(HIDDEN)

			// Konfiguriere Neuron
			if l == len(layerSizes)-1 {
				n.Type = OUTPUT
				nn.Outputs = append(nn.Outputs, &n)
			} else {
				if l == 0 {
					n.Type = INPUT
					nn.Inputs = append(nn.Inputs, &n)
				}
				// Verbinde zu allen Neuronen im nächst höheren Layer
				for _, p := range parentLayer {
					if p.Type != BIAS {
						p.ConnectTo(&n)
					}
				}
			}
		}
		// Bias-Neuronen
		if useBiases && l != len(layerSizes)-1 {
			b := NewNeuron(BIAS)
			for _, p := range parentLayer {
				if p.Type != BIAS {
					p.ConnectTo(&b)
				}
			}
		}
	}
}

/*RandomizeWeights weist allen Verbindungen des
Netzwerkes zufällige Gewichtungen zwischen einem
minimal und einem maximal wert zu.*/
func (nn *NeuralNetwork) RandomizeWeights(min float64, max float64) {
	rand.Seed(int64(time.Now().Nanosecond() * time.Now().Minute()))
	// Für alle Schichten außer der Eingabe-Schicht(da keine Eingehenden Verbindungen)
	_, connectedLayers := nn.GetLayer(0)
	for _, l := range connectedLayers {
		for _, n := range l {
			for _, conn := range n.Conns {
				conn.Weight = (rand.Float64() * (max - min)) + min
			}
		}
	}
}

/*GetLayer gibt die Neuronen einer gegebenen Schicht
zurück. Außerdem werden alle tieferen Schichten zurückgegeben.*/
func (nn *NeuralNetwork) GetLayer(l int) ([]*Neuron, [][]*Neuron) {
	var allParents [][]*Neuron
	var layer []*Neuron

	if l == nn.NumLayers-1 {
		return nn.Outputs, allParents
	} else if l < nn.NumLayers && l >= 0 {
		parentLayer, grands := nn.GetLayer(l + 1)
		allParents = append([][]*Neuron{parentLayer}, grands...)
		// für alle Verbindungen zu dem gewünschten Layer
		for _, parent := range parentLayer {
		addConnections:
			for _, con := range parent.Conns {
				for _, layerNeuron := range layer {
					// wenn das verbundene Neuron schon in Liste:
					if layerNeuron.ID == con.Neuron.ID {
						// gehe zum nächsten neuron
						continue addConnections
					}
				}
				// füge Neuron zur Liste hinzu
				layer = append(layer, con.Neuron)
			}
		}
	}

	return layer, allParents
}

/*GetLayers gibt alle Schichten des
Netzwerkes wieder.*/
func (nn *NeuralNetwork) GetLayers() [][]*Neuron {
	input, others := nn.GetLayer(0)
	return append([][]*Neuron{input}, others...)
}

/*SaveTo speichert ein Netzwerk in
eine Textdatei (JSON-Format).*/
func (nn *NeuralNetwork) SaveTo(path string) {
	// alle Neuronen in einer Liste speichern, Redundanz vermeiden
	nn.Neurons = nn.Neurons[:0]
	for _, layer := range nn.GetLayers() {
		nn.Neurons = append(nn.Neurons, layer...)
	}

	// schrieben der Textdatei
	bytes, err := json.MarshalIndent(nn, "", "\t")
	check(err)

	file, err := os.Create(path)
	check(err)
	defer file.Close()

	_, err = file.Write(bytes)
	check(err)
}

/*LoadFrom lädt ein Netzwerk aus einer
Textdatei (JSON-Format).*/
func (nn *NeuralNetwork) LoadFrom(path string) error {
	// Textdatei lesen
	bytes, err := ioutil.ReadFile(path)

	err = json.Unmarshal(bytes, &nn)

	if err != nil {
		return errors.New("Error while reading file '" + path + "'!")
	}

	nn.ActivFunc = Sigmoid       //DEFAULT
	nn.ActivDeriv = SigmoidDeriv //DEFAULT

	for _, neuron := range nn.Neurons {
		// Input- und Output-Neuronen in Listen zuweisen, erleichterter Zugriff
		if neuron.Type == OUTPUT {
			nn.Outputs = append(nn.Outputs, neuron)
		} else if neuron.Type == INPUT {
			nn.Inputs = append(nn.Inputs, neuron)
		}
		// Pointer-Referenzen mit Hilfe der gespeicherten IDs wieder herstellen
		for _, conn := range neuron.Conns {
		search:
			for _, other := range nn.Neurons {
				if other.ID == conn.ConnectedNeuronID {
					conn.Neuron = other
					break search
				}
			}
		}
	}
	nn.Neurons = nn.Neurons[:0]

	return nil
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
