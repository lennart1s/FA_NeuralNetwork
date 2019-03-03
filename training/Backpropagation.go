package training

import (
	NN "FA_NeuralNetwork/network"
)

/*Backpropagation trainiert ein gegebenes Netzwerk
mit den gegeben Trainings-Daten mithilfe der
Backprop-Methode.
Bei allen Schritten außer der Delta-Wert berechnung
wird die Graphen-struktur des Netzwerkes genutzt um
die Durchläufe rekursiv geschehen zu lassen(z.B. calculateGradients).*/
func Backpropagation(nn *NN.NeuralNetwork, td DataSet) {
	layers := nn.GetLayers()

	// Werte und Indikatoren zurücksetzen
	for _, output := range nn.Outputs {
		output.UnsetCalculatedGradients(true)
		output.UnsetChangedWeights()
	}

	// für alle Trainings-Szenarien:
	for i := 0; i < len(td.Inputs); i++ {
		nn.Run(td.Inputs[i])

		// Werte und Indikatoren zurücksetzen, Gradienten werden nicht genullt!
		for _, output := range nn.Outputs {
			output.UnsetCalculatedGradients(false)
			output.ZeroPrevLayerWeightedDelta()
		}

		outputIndex := -1
		for l := len(layers) - 1; l > 0; l-- {
			for _, neuron := range layers[l] {
				// Berechne Delta-Werte
				if neuron.Type == NN.OUTPUT {
					outputIndex++
					calculateOuptutNeuronDelta(neuron, &nn.ActivDeriv, td.Ideals[i][outputIndex])
				} else if neuron.Type == NN.HIDDEN {
					calculateDelta(neuron, &nn.ActivDeriv)
				}
				// Passe PrevLayerWeightedDelta für Neuronen in höhere Schicht an
				if neuron.Type == NN.OUTPUT || neuron.Type == NN.HIDDEN {
					for _, con := range neuron.Conns {
						con.Neuron.PrevLayerWeightedDelta += neuron.Delta * con.Weight
					}
				}
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

func calculateOuptutNeuronDelta(neuron *NN.Neuron, activDeriv *NN.FloatFunction, ideal float64) {
	err := neuron.Output - ideal
	neuron.Delta = -err * (*activDeriv)(neuron.Input)
}

func calculateDelta(neuron *NN.Neuron, activDeriv *NN.FloatFunction) {
	neuron.Delta = (*activDeriv)(neuron.Input) * neuron.PrevLayerWeightedDelta
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

func applyWeightChanges(neuron *NN.Neuron, learningRate float64, momentum float64) {
	neuron.ChangedWeights = true
	for _, conn := range neuron.Conns {
		conn.WeightChange = learningRate*conn.Gradient + momentum*conn.WeightChange
		conn.Weight += conn.WeightChange
		if !conn.Neuron.ChangedWeights {
			applyWeightChanges(conn.Neuron, learningRate, momentum)
		}
	}
}
