package training

import (
	NN "FA_NeuralNetwork/network"
)

func Backpropagation(nn *NN.NeuralNetwork, td TrainingData) {
	layers := nn.GetLayers()

	for _, layer := range layers {
		for _, neuron := range layer {
			for c := 0; c < len(neuron.Conns); c++ {
				neuron.Conns[c].Gradient = 0
			}
		}
	}

	for i := 0; i < len(td.Inputs); i++ {
		actualOut := nn.Run(td.Inputs[i])

		for _, layer := range layers {
			for _, neuron := range layer {
				neuron.PrevLayerWeightedDelta = 0
			}
		}

		var outputIndex int
		for l := len(layers) - 1; l > 0; l-- {
			for _, neuron := range layers[l] {
				if neuron.Type == NN.OUTPUT {
					err := actualOut[outputIndex] - td.Ideals[i][outputIndex]
					neuron.Delta = -err * nn.ActivDeriv(neuron.Input)
					outputIndex++
				}
				if neuron.Type == NN.HIDDEN {
					neuron.Delta = nn.ActivDeriv(neuron.Input) * neuron.PrevLayerWeightedDelta
				}
				for _, con := range neuron.Conns {
					con.PrevLayerWeightedDelta += neuron.Delta * con.Weight
				}
			}
		}

		for _, layer := range layers {
			for _, neuron := range layer {
				for c := 0; c < len(neuron.Conns); c++ {
					neuron.Conns[c].Gradient += neuron.Delta * neuron.Conns[c].Output
				}
			}
		}

	}

	for _, layer := range layers {
		for _, neuron := range layer {
			for c := 0; c < len(neuron.Conns); c++ {
				con := &neuron.Conns[c]
				con.WeightChange = td.LearningRate*con.Gradient + td.Momentum*con.WeightChange
				con.Weight += con.WeightChange
			}
		}
	}

}
