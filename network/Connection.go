package network

import UUID "github.com/google/uuid"

type Connection struct {
	*Neuron `json:"-"`

	ConnectedNeuronId UUID.UUID

	Weight float64

	WeightChange float64
	Gradient     float64
}
