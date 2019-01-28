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
	for n := 0; n < len(nn.Neurons); n++ {
		for c := 0; c < len(nn.Neurons[n].Conns); c++ {
			nn.Neurons[n].Conns[c].Gradient = 0
		}
	}

	for i := 0; i < len(td.Inputs); i++ {
		actualOut := nn.Run(td.Inputs[i])

		// Delta-calculation
		for n := 0; n < len(nn.Neurons); n++ {
			nn.Neurons[n].PrevLayerWeightedDelta = 0
		}
		var outputCount int
		for l := len(layers) - 1; l >= 0; l-- {
			for _, n := range layers[l] {
				if n.Type == NN.OUTPUT {
					err := actualOut[outputCount] - td.Ideals[i][outputCount]
					n.Delta = -err * nn.ActivDeriv(n.Input)
					outputCount++ **DOING SOME (PAUSE***
				} else if n.Type == NN.HIDDEN {
					n.Delta = nn.ActivDeriv(n.Input) * n.PrevLayerWeightedDelta
				}
				for _, c := range n.Conns {
					c.UpperNeuron.PrevLayerWeightedDelta += n.Delta * c.Weight
				}
			}
		}

		// Gradient-calculation
		for n := 0; n < len(nn.Neurons); n++ {
			for c := 0; c < len(nn.Neurons[n].Conns); c++ {
				nn.Neurons[n].Conns[c].Gradient += nn.Neurons[n].Delta * nn.Neurons[n].Conns[c].UpperNeuron.Output
			}
		}

	}

	// Weight-adjustment
	for n := 0; n < len(nn.Neurons); n++ {
		for c := 0; c < len(nn.Neurons[n].Conns); c++ {
			con := &nn.Neurons[n].Conns[c]
			con.WeightChange = td.LearningRate*con.Gradient + td.Momentum*con.WeightChange
			con.Weight += con.WeightChange
		}
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
