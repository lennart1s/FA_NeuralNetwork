package main

import (
	NN "FA_NeuralNetwork/neuralnetwork"
	"fmt"
)

func main() {
	nn := NN.NeuralNetwork{}
	settings := NN.NetworkSettings{Inputs: 2, Outputs: 1, LayerSizes: []int{2},
		ActivFunc: NN.Sigmoid, ActivDeriv: NN.SigmoidDeriv, Layered: true, UseBiases: true}

	nn.Create(settings)
	nn.RandomizeWeights(-4, 4)

	fmt.Println(nn.Run([]float64{1, 0}))
}
