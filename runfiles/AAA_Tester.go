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

	nn.SaveTo("r")

	//nn.LoadFrom("./nn", NN.Sigmoid, NN.SigmoidDeriv)

	td := NT.TrainingData{Inputs: [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}, Ideals: [][]float64{{0}, {1}, {1}, {0}}}
	td.LearningRate = 0.7
	td.Momentum = 0.3

	count := 0
	var err float64
	for err = NT.MeanSquaredError(&nn, td); err > 0.00001; err = NT.MeanSquaredError(&nn, td) {
		NT.Backpropagation(&nn, td)
		count++
		if count%3000 == 0 {
			fmt.Println(err)
		}
	}

	fmt.Println(NT.MeanSquaredError(&nn, td))
	fmt.Println(nn.Run([]float64{0, 0}))

}
