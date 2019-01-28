package neuralnetwork

import (
	"encoding/json"
	"io/ioutil"
	"math/rand"
	"os"
	"time"
)

type NeuralNetwork struct {
	Neurons []Neuron

	ActivFunc  FloatFunction `json:"-"`
	ActivDeriv FloatFunction `json:"-"`

	IsLayered bool
	Layers    int

	Fitness float64
}

func (nn *NeuralNetwork) Run(input []float64) []float64 {
	var inputCount int
	for n := 0; n < len(nn.Neurons); n++ {
		nn.Neurons[n].Processed = false
		if inputCount < len(input) && nn.Neurons[n].Type == INPUT {
			nn.Neurons[n].Input = input[inputCount]
			inputCount++
		}
	}

	var output []float64
	for n := 0; n < len(nn.Neurons); n++ {
		if nn.Neurons[n].Type == OUTPUT {
			nn.Neurons[n].Process(nn.ActivFunc)
			output = append(output, nn.Neurons[n].Output)
		}
	}
	return output
}

func (nn *NeuralNetwork) Create(settings NetworkSettings) {
	nn.Neurons = nn.Neurons[:0]
	nn.ActivFunc = settings.ActivFunc
	nn.ActivDeriv = settings.ActivDeriv

	if settings.Layered {
		nn.createLayered(settings)
	}
}

func (nn *NeuralNetwork) createLayered(settings NetworkSettings) {
	nn.IsLayered = true
	nn.Layers = len(settings.LayerSizes) + 2
	layerSizes := append([]int{settings.Inputs}, append(settings.LayerSizes, settings.Outputs)...)
	for l := 0; l < len(layerSizes); l++ {
		for i := 0; i < layerSizes[l]; i++ {
			n := NewNeuron(HIDDEN)
			n.Layer = l

			if l == 0 {
				n.Type = INPUT
				nn.Neurons = append(nn.Neurons, n)
				continue
			} else if l == len(layerSizes)-1 {
				n.Type = OUTPUT
			}

			/*for o := 0; o < len(nn.Neurons); o++ {
				if nn.Neurons[o].Layer == n.Layer-1 {
					n.ConnectTo(&nn.Neurons[o])
				}
			}*/

			nn.Neurons = append(nn.Neurons, n)

			/*for c := 0; c < len(nn.Neurons[len(nn.Neurons)-1].Conns); c++ {
			search:
				for o := 0; o < len(nn.Neurons); o++ {
					if nn.Neurons[o].Id == nn.Neurons[len(nn.Neurons)-1].Conns[c].UpperNeuronID {
						nn.Neurons[len(nn.Neurons)-1].Conns[c].UpperNeuron = &nn.Neurons[o]
						break search
					}
				}

			}*/
		}
		if settings.UseBiases && l != len(layerSizes)-1 {
			b := NewNeuron(BIAS)
			b.Layer = l
			nn.Neurons = append(nn.Neurons, b)
		}
	}

	for n := 0; n < len(nn.Neurons); n++ {
		for c := 0; c < len(nn.Neurons[n].Conns); c++ {
			for o := 0; o < len(nn.Neurons); o++ {
				if nn.Neurons[o].Layer == nn.Neurons[n].Layer-1 {
					//nn.Neurons[n].Conns[c].UpperNeuron = &nn.Neurons[o]
					nn.Neurons[n].ConnectTo(&nn.Neurons[o])
					//break search
				}
			}

		}
	}
}

func (nn *NeuralNetwork) RandomizeWeights(min float64, max float64) {
	rand.Seed(int64(time.Now().Nanosecond() * time.Now().Minute()))
	for n := 0; n < len(nn.Neurons); n++ {
		for c := 0; c < len(nn.Neurons[n].Conns); c++ {
			nn.Neurons[n].Conns[c].Weight = (rand.Float64() * (max - min)) + min
		}
	}
}

type NetworkSettings struct {
	Inputs  int
	Outputs int

	UseBiases bool

	Layered    bool
	LayerSizes []int

	ActivFunc  FloatFunction
	ActivDeriv FloatFunction
}

func (nn *NeuralNetwork) RedoConnectionIndices() {
	for n := 0; n < len(nn.Neurons); n++ {
		for c := 0; c < len(nn.Neurons[n].Conns); c++ {
			for o := 0; o < len(nn.Neurons); o++ {
				if nn.Neurons[o].Layer == nn.Neurons[n].Layer-1 {
					nn.Neurons[n].Conns[c].UpperNeuron = &nn.Neurons[o]
					nn.Neurons[n].ConnectTo(&nn.Neurons[o])
					//break search
				}
			}

		}
	}
}

func (nn *NeuralNetwork) SaveTo(path string) {
	bytes, err := json.MarshalIndent(nn, "", "\t")
	check(err)

	file, err := os.Create(path)
	check(err)
	defer file.Close()

	_, err = file.Write(bytes)
	check(err)
}

func (nn *NeuralNetwork) LoadFrom(path string, activFunc FloatFunction, activDeriv FloatFunction) {
	bytes, err := ioutil.ReadFile(path)
	check(err)

	err = json.Unmarshal(bytes, &nn)
	check(err)

	nn.ActivFunc = activFunc
	nn.ActivDeriv = activDeriv

	for n := 0; n < len(nn.Neurons); n++ {
		for c := 0; c < len(nn.Neurons[n].Conns); c++ {
		search:
			for o := 0; o < len(nn.Neurons); o++ {
				if nn.Neurons[o].Id == nn.Neurons[n].Conns[c].UpperNeuronID {
					nn.Neurons[n].Conns[c].UpperNeuron = &nn.Neurons[o]
					break search
				}
			}

		}
	}
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
