package training

import (
	NN "FA_NeuralNetwork/neuralnetwork"
	"fmt"
)

func Backpropagation(nn *NN.NeuralNetwork, td TrainingData) {
	if !nn.IsLayered {
		fmt.Println("Backpropagation can only be used with layered NeuralNetworks!")
		return
	}

	layers := getLayers(nn)

	for i := 0; i < len(td.Inputs) && i < len(td.Ideals); i++ {
		actualOut := nn.Run(td.Inputs[i])

		// Delta-calculation
		var outputCount int
		for l := len(layers) - 1; l >= 0; l++ {
			for _, n := range layers[l] {
				if n.Type == NN.OUTPUT {
					err := actualOut[outputCount] - td.Ideals[i][outputCount]
					n.Delta = -err * nn.ActivDeriv(n.Input)
					outputCount++
					for _, c := range n.Conns {
						c.UpperNeuron.PrevLayerWeightedDelta += n.Delta * c.Weight
					}
				} else if n.Type == NN.HIDDEN {
					n.Delta = nn.ActivDeriv(n.Input) * n.PrevLayerWeightedDelta
					for _, c := range n.Conns {
						c.UpperNeuron.PrevLayerWeightedDelta += n.Delta * c.Weight
					}
				}
			}
		}

		// Gradient-calculation

	}

}

func getLayers(nn *NN.NeuralNetwork) [][]*NN.Neuron {
	layers := make([][]*NN.Neuron, nn.Layers)
	for n := 0; n < len(nn.Neurons); n++ {
		layers[nn.Neurons[n].Layer] = append(layers[nn.Neurons[n].Layer], &nn.Neurons[n])
	}
	return layers
}

func countConnections(nn *NN.NeuralNetwork) int {
	var count int
	for _, n := range nn.Neurons {
		count += len(n.Conns)
	}
	return count
}
