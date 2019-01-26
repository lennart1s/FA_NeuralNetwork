package main

import (
	NN "FA_NeuralNetwork/neuralnetwork"
	NT "FA_NeuralNetwork/training"
	"fmt"
)

func main() {
	nn := NN.NeuralNetwork{}
	settings := NN.NetworkSettings{Inputs: 2, Outputs: 1, LayerSizes: []int{2},
		ActivFunc: NN.Sigmoid, ActivDeriv: NN.SigmoidDeriv, Layered: true, UseBiases: true}

	nn.Create(settings)
	nn.RandomizeWeights(-4, 4)

	fmt.Println(nn.Run([]float64{1, 0}))

	td := NT.TrainingData{Inputs: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}, Ideals: [][]float64{{0}, {1}, {1}, {0}}}
	td.LearningRate = 0.7
	td.Momentum = 0.3

	for err := NT.MeanSquaredError(&nn, td); err > 0.00001; err = NT.MeanSquaredError(&nn, td) {
		NT.Backpropagation(&nn, td)
	}

	fmt.Println(NT.MeanSquaredError(&nn, td))

}
