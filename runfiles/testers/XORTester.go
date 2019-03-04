package main

import (
	NN "FA_NeuralNetwork/network"
	NT "FA_NeuralNetwork/training"
)

func main4() {
	nn := NN.NeuralNetwork{}
	//nn.CreateLayered([]int{2, 2, 1}, true)
	//nn.RandomizeWeights(-1, 1)
	nn.LoadFrom("./savefiles/xor.nn")

	td := NT.DataSet{
		Inputs:       [][]float64{ /*{0, 0}, {0, 1}, */ {1, 0} /*, {1, 1}*/},
		Ideals:       [][]float64{ /*{0}, {1},*/ {1} /*, {0}*/},
		Momentum:     0.3,
		LearningRate: 0.7}

	/*var i int
	for err := NT.MeanSquaredError(&nn, td); err > 0.000001; err = NT.MeanSquaredError(&nn, td) {
		NT.Backpropagation(&nn, td)
		i++
		if i%2500 == 0 {
			fmt.Println(err)
		}
	}*/
	NT.Backpropagation(&nn, td)

	//nn.SaveTo("./savefiles/xor.nn")
	//fmt.Println(nn.Run([]float64{0, 0}))
	//fmt.Println(nn.Run([]float64{0, 1}))
	//fmt.Println(nn.Run([]float64{1, 0}))
	//fmt.Println(nn.Run([]float64{1, 1}))
	nn.SaveTo("./savefiles/xor_trained2.nn")
}
