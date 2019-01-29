package main

import (
	NN "FA_NeuralNetwork/network"
	"fmt"
)

func main() {
	nn := NN.NeuralNetwork{}
	nn.CreateLayered([]int{4, 2, 4, 1}, true)
	nn.RandomizeWeights(-2, 2)
	//nn.LoadFrom("../savefiles/nn")

	layer, parents := nn.GetLayer(0)
	all := append(parents, layer)
	fmt.Println("Layers: ", len(all))

	for i := nn.NumLayers - 1; i >= 0; i-- {
		l, _ := nn.GetLayer(i)
		fmt.Println("Size of layer", i, ":", len(l))
		/*for _, n :=  range layer {
			for _, c := range c.
		}*/
	}

	//nn.SaveTo("../savefiles/nn")
	fmt.Println(nn.Run([]float64{-3.4, 2.445, 4.5, 12}))

}
