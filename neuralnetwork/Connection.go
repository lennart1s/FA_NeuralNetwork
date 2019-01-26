package neuralnetwork

type Connection struct {
	UpperNeuron   *Neuron `json:"-"`
	UpperNeuronID int

	Weight float64

	LastWeightChange float64
	Gradient         float64
}
