package network

type Connection struct {
	*Neuron //`json:"-"`

	//ConnectedNeuronId UUID.UUID

	Weight float64

	WeightChange float64
	Gradient     float64
}
