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
	for c := 0; c < len(neuron.Conns); c++ {
		con := neuron.Conns[c]
		con.WeightChange = learningRate*con.Gradient + momentum*con.WeightChange
		con.Weight += con.WeightChange
		applyWeightChanges(con.Neuron, learningRate, momentum)
	}
}

func calculateGradients(neuron *NN.Neuron) {
	neuron.CalculatedGradients = true
	for c := 0; c < len(neuron.Conns); c++ {
		neuron.Conns[c].Gradient += neuron.Delta * neuron.Conns[c].Neuron.Output
		if !neuron.Conns[c].Neuron.CalculatedGradients {
			calculateGradients(neuron.Conns[c].Neuron)
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
	for c := 0; c < len(neuron.Conns); c++ {
		if !onlyFlag {
			neuron.Conns[c].Gradient = 0
		}
		resetGradients(neuron.Conns[c].Neuron, onlyFlag)
	}
}

func resetPrevLayerDeltas(neuron *NN.Neuron) {
	neuron.PrevLayerWeightedDelta = 0
	for _, con := range neuron.Conns {
		resetPrevLayerDeltas(con.Neuron)
	}
}
