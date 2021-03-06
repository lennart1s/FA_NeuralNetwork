package training

import (
	NN "FA_NeuralNetwork/network"
	"math"
)

/*MeanSquaredError gibt den Fehlerwert des
Netzwerkes mit dem gegebenen TrainingsSet.*/
func MeanSquaredError(nn *NN.NeuralNetwork, td DataSet) float64 {
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
