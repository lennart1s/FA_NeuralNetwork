package neuralnetwork

type Connection struct {
	UpperNeuron   *Neuron `json:"-"`
	UpperNeuronID int

	Weight float64

	WeightChange float64
	Gradient     float64
}
