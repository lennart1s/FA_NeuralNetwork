package main

import (
	NN "FA_NeuralNetwork/network"
	"fmt"
)

func main4() {
	nn := NN.NeuralNetwork{}
	//nn.CreateLayered([]int{2, 2, 1}, true)
	//nn.RandomizeWeights(-1, 1)
	nn.LoadFrom("./savefiles/xor.nn")

	/*td := NT.TrainingData{
		Inputs:       [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		Ideals:       [][]float64{{0}, {1}, {1}, {0}},
		Momentum:     0.3,
		LearningRate: 0.7}

	var i int
	for err := NT.MeanSquaredError(&nn, td); err > 0.001; err = NT.MeanSquaredError(&nn, td) {
		NT.TOBackpropagation(&nn, td)
		i++
		if i%2500 == 0 {
			fmt.Println(err)
		}
	}*/

	//nn.SaveTo("./savefiles/xor.nn")
	fmt.Println(nn.Run([]float64{0, 0}))
	fmt.Println(nn.Run([]float64{0, 1}))
	fmt.Println(nn.Run([]float64{1, 0}))
	fmt.Println(nn.Run([]float64{1, 1}))
}
