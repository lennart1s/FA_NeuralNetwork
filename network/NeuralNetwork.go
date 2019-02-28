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
					if p.Type != BIAS {
						p.ConnectTo(&n)
					}
				}
			}
		}
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

func (nn *NeuralNetwork) RandomizeWeights(min float64, max float64) {
	rand.Seed(int64(time.Now().Nanosecond() * time.Now().Minute()))
	_, connectedLayers := nn.GetLayer(0)
	for _, l := range connectedLayers {
		for _, n := range l {
			for _, conn := range n.Conns {
				conn.Weight = (rand.Float64() * (max - min)) + min
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
					if layerNeuron.ID == con.Neuron.ID {
						continue addConnections
					}
				}
				layer = append(layer, con.Neuron)
			}
		}
	}

	return layer, allParents
}

func (nn *NeuralNetwork) GetLayers() [][]*Neuron {
	input, others := nn.GetLayer(0)
	return append([][]*Neuron{input}, others...)
}

////////////////////////

func (nn *NeuralNetwork) SaveTo(path string) {
	nn.Neurons = nn.Neurons[:0]
	for _, layer := range nn.GetLayers() {
		nn.Neurons = append(nn.Neurons, layer...)
	}

	bytes, err := json.MarshalIndent(nn, "", "\t")
	check(err)

	file, err := os.Create(path)
	check(err)
	defer file.Close()

	_, err = file.Write(bytes)
	check(err)
}

func (nn *NeuralNetwork) LoadFrom(path string) error {
	bytes, err := ioutil.ReadFile(path)
	//check(err)

	err = json.Unmarshal(bytes, &nn)
	//check(err)

	if err != nil {
		return errors.New("Error while reading file '" + path + "'!")
	}

	nn.ActivFunc = Sigmoid       //DEFAULT
	nn.ActivDeriv = SigmoidDeriv //DEFAULT

	for _, neuron := range nn.Neurons {
		if neuron.Type == OUTPUT {
			nn.Outputs = append(nn.Outputs, neuron)
		} else if neuron.Type == INPUT {
			nn.Inputs = append(nn.Inputs, neuron)
		}
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
