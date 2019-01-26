package neuralnetwork

import (
	"math/rand"
	"time"
)

type NeuralNetwork struct {
	Neurons []Neuron

	ActivFunc  FloatFunction `json:"-"`
	ActivDeriv FloatFunction `json:"-"`

	IsLayered bool

	Fitness float64
}

func (nn *NeuralNetwork) Run(input []float64) []float64 {
	for n := 0; n < len(nn.Neurons); n++ {
		nn.Neurons[n].Processed = false
		var inputCount int
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
	layerSizes := append([]int{settings.Inputs}, append(settings.LayerSizes, settings.Outputs)...)
	for l := 0; l < len(layerSizes); l++ {
		for i := 0; i < layerSizes[l]; i++ {
			n := Neuron{Type: HIDDEN, Layer: l}

			if l == 0 {
				n.Type = INPUT
				continue
			} else if l == len(layerSizes)-1 {
				n.Type = OUTPUT
			}

			for o := 0; o < len(nn.Neurons); o++ {
				if nn.Neurons[o].Layer == n.Layer-1 {
					n.Conns = append(n.Conns, Connection{UpperNeuron: &nn.Neurons[o]})
				}
			}

			nn.Neurons = append(nn.Neurons, n)
		}
		if settings.UseBiases {
			b := Neuron{Type: BIAS, Layer: l}
			nn.Neurons = append(nn.Neurons, b)
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
