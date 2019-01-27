package training

import (
	NN "FA_NeuralNetwork/neuralnetwork"
	"math"
)

func MeanSquaredError(nn *NN.NeuralNetwork, td TrainingData) float64 {
	var err float64

	for i := 0; i < len(td.Inputs); i++ {
		actualOut := nn.Run(td.Inputs[i])
		for j := 0; j < len(actualOut); j++ {
			err += math.Pow(td.Ideals[i][j]-actualOut[j], 2)
		}
	}
	err /= float64(len(td.Inputs))

	return err
}
