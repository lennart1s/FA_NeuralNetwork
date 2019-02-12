package network

import (
	"encoding/json"
	"io/ioutil"
	"math/rand"
	"os"
	"time"
)

/* Neural Netowork ist eine Strultur aus Neuronen
und Verbindungen zwischen diesen. Außerdem werden
Informationen wie die Aktivierungsfunktion, deren
Ableitung und auch, für spezifische Trainings-
Algorithmen benötigte, Daten verwaltet.*/
type NeuralNetwork struct {
	/* Inputs ist ein Slice, welches Pointer
	zu allen Eingabe-Neuronen des Netwerkes
	speichert um einen einfacheren Zugriff
	zu gewährleisten.*/
	Inputs []*Neuron
	/* Outputs ist ein Slice, welches Pointer
	zu allen Ausgabe-Neuronen des Netwerkes
	speichert um einen einfacherern Zugriff
	zu gewährleisten.*/
	Outputs []*Neuron

	/* ActivFunc ist die Aktivierungsfunktion,
	welche beim Berechnen mithilfe des Netzes
	verwendet wird.
	Sie wird als FloatFunction gespeichert.*/
	ActivFunc FloatFunction `json:"-"`
	/* ActivDeriv ist die Ableitung der
	Aktivierungsfunktion, welche beim Trainieren
	des Netzes mit der Gradient-Descent-Methode
	verwendet wird.
	Sie wird als FloatFunction gespeichert.*/
	ActivDeriv FloatFunction `json:"-"`

	NumLayers int

	BackPropRuns int
	Fitness      float64
}

func (nn *NeuralNetwork) Run(input []float64) []float64 {
	for _, out := range nn.Outputs {
		out.SetUnprocessed()
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

func (nn *NeuralNetwork) CreateLayered(layerSizes []int, useBiases bool) {
	nn.Outputs = nn.Outputs[:0]
	nn.ActivFunc = Sigmoid
	nn.ActivDeriv = SigmoidDeriv

	nn.NumLayers = len(layerSizes)
	for l := len(layerSizes) - 1; l >= 0; l-- {
		parentLayer, _ := nn.GetLayer(l + 1)
		for i := 0; i < layerSizes[l]; i++ {
			n := NewNeuron(HIDDEN)

			if l == len(layerSizes)-1 {
				n.Type = OUTPUT
				nn.Outputs = append(nn.Outputs, &n)
			} else {
				if l == 0 {
					n.Type = INPUT
					nn.Inputs = append(nn.Inputs, &n)
				}
				for _, p := range parentLayer {
					p.ConnectTo(&n)
				}
			}
		}
		if useBiases && l != len(layerSizes)-1 {
			b := NewNeuron(BIAS)
			for _, p := range parentLayer {
				p.ConnectTo(&b)
			}
		}
	}
}

func (nn *NeuralNetwork) RandomizeWeights(min float64, max float64) {
	rand.Seed(int64(time.Now().Nanosecond() * time.Now().Minute()))
	_, connectedLayers := nn.GetLayer(0)
	for _, l := range connectedLayers {
		for _, n := range l {
			for c := 0; c < len(n.Conns); c++ {
				n.Conns[c].Weight = (rand.Float64() * (max - min)) + min
			}
		}
	}
}

func (nn *NeuralNetwork) GetLayer(l int) ([]*Neuron, [][]*Neuron) {
	var allParents [][]*Neuron
	var layer []*Neuron

	if l == nn.NumLayers-1 {
		for i := 0; i < len(nn.Outputs); i++ {
			layer = append(layer, nn.Outputs[i])
		}
	} else if l < nn.NumLayers && l >= 0 {
		parentLayer, grands := nn.GetLayer(l + 1)
		allParents = append([][]*Neuron{parentLayer}, grands...)
		for _, parent := range parentLayer {
		addConnections:
			for _, con := range parent.Conns {
				for _, layerNeuron := range layer {
					if layerNeuron.ID == con.ID {
						continue addConnections
					}
				}
				layer = append(layer, con.Neuron)
			}
		}
	} /*else {
		return nil, nil
	}*/

	return layer, allParents
}

func (nn *NeuralNetwork) GetLayers() [][]*Neuron {
	input, others := nn.GetLayer(0)
	return append([][]*Neuron{input}, others...)
}

////////////////////////

// Deprecated
func (nn *NeuralNetwork) SaveTo(path string) {
	bytes, err := json.MarshalIndent(nn, "", "\t")
	check(err)

	file, err := os.Create(path)
	check(err)
	defer file.Close()

	_, err = file.Write(bytes)
	check(err)
}

// Deprecated
func (nn *NeuralNetwork) LoadFrom(path string) {
	bytes, err := ioutil.ReadFile(path)
	check(err)

	err = json.Unmarshal(bytes, &nn)
	check(err)

	nn.ActivFunc = Sigmoid
	nn.ActivDeriv = SigmoidDeriv
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
