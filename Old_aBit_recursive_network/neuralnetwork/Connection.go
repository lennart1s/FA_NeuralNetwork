package neuralnetwork

type Connection struct {
	UpperNeuron *Neuron `json:"-"`
	UpperNeuronID UUID.UUID

	Weight float64

	WeightChange float64
	Gradient     float64
}