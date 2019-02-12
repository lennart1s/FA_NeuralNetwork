package main

import (
	NN "FA_NeuralNetwork/network"
	NT "FA_NeuralNetwork/training"
	"fmt"
)

func main() {
	nn := NN.NeuralNetwork{}
	nn.CreateLayered([]int{2, 2, 1}, true)
	nn.RandomizeWeights(-1, 1)

	all := nn.GetLayers()
	println("Layers: ", len(all[1]))

	for i := nn.NumLayers - 1; i >= 0; i-- {
		l, _ := nn.GetLayer(i)
		println("Size of layer", i, ":", len(l))
		/*for _, n :=  range layer {
			for _, c := range c.
		}*/
	}

	td := NT.TrainingData{
		Inputs:       [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		Ideals:       [][]float64{{0}, {1}, {1}, {0}},
		Momentum:     0.3,
		LearningRate: 0.7}

	var i int
	for err := NT.MeanSquaredError(&nn, td); err > 0.000001; err = NT.MeanSquaredError(&nn, td) {
		NT.TOBackpropagation(&nn, td)
		i++
		if i%2500 == 0 {
			fmt.Println(err)
		}
	}

	nn.SaveTo("./savefiles/xor.nn")
	fmt.Println(nn.Run([]float64{1, 0}))

}
