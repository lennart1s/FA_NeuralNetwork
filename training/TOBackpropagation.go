package training

import (
	NN "FA_NeuralNetwork/network"
)

func TOBackpropagation(nn *NN.NeuralNetwork, td TrainingData) {
	layers := nn.GetLayers()

	for i := 0; i < len(td.Inputs); i++ {
		nn.Run(td.Inputs[i])

		onlyFlag := true
		if i == 0 {
			onlyFlag = false
		}
		for _, output := range nn.Outputs {
			resetGradients(output, onlyFlag)
			resetPrevLayerDeltas(output)
		}

		outputIndex := -1 //TODO: layerd but tree orientated
		for l := len(layers) - 1; l > 0; l-- {
			for _, neuron := range layers[l] {
				if neuron.Type == NN.OUTPUT {
					outputIndex++
				}
				calculateDelta(neuron, &nn.ActivDeriv, td.Ideals[i][outputIndex])
			}
		}

		for _, output := range nn.Outputs {
			calculateGradients(output)
		}

	}

	for _, output := range nn.Outputs {
		applyWeightChanges(output, td.LearningRate, td.Momentum)
	}

	nn.BackPropRuns += len(td.Inputs)
}

func applyWeightChanges(neuron *NN.Neuron, learningRate float64, momentum float64) {
	for _, conn := range neuron.Conns {
		conn.WeightChange = learningRate*conn.Gradient + momentum*conn.WeightChange
		conn.Weight += conn.WeightChange
		applyWeightChanges(conn.Neuron, learningRate, momentum)
	}
}

func calculateGradients(neuron *NN.Neuron) {
	neuron.CalculatedGradients = true
	for _, conn := range neuron.Conns {
		conn.Gradient += neuron.Delta * conn.Neuron.Output
		if !conn.Neuron.CalculatedGradients {
			calculateGradients(conn.Neuron)
		}
	}
}

func calculateDelta(neuron *NN.Neuron, activDeriv *NN.FloatFunction, ideal float64) {
	if neuron.Type == NN.OUTPUT {
		err := neuron.Output - ideal
		neuron.Delta = -err * (*activDeriv)(neuron.Input)
	} else if neuron.Type == NN.HIDDEN {
		neuron.Delta = (*activDeriv)(neuron.Input) * neuron.PrevLayerWeightedDelta
	}
	for _, con := range neuron.Conns {
		con.Neuron.PrevLayerWeightedDelta += neuron.Delta * con.Weight
	}
}

// Reset-Functions
func resetGradients(neuron *NN.Neuron, onlyFlag bool) {
	neuron.CalculatedGradients = false
	for _, conn := range neuron.Conns {
		if !onlyFlag {
			conn.Gradient = 0
		}
		resetGradients(conn.Neuron, onlyFlag)
	}
}

func resetPrevLayerDeltas(neuron *NN.Neuron) {
	neuron.PrevLayerWeightedDelta = 0
	for _, con := range neuron.Conns {
		resetPrevLayerDeltas(con.Neuron)
	}
}
