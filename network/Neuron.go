package network

import (
	UUID "github.com/google/uuid"
)

type Neuron struct {
	Id UUID.UUID

	Type int

	Conns []Connection

	Input  float64
	Output float64

	Processed           bool
	CalculatedGradients bool

	Delta                  float64
	PrevLayerWeightedDelta float64
}

func NewNeuron(Type int) Neuron {
	id, _ := UUID.NewUUID()
	n := Neuron{Type: Type, Id: id}
	return n
}

func (n *Neuron) ConnectTo(o *Neuron) {
	n.Conns = append(n.Conns, Connection{Neuron: o})
}

const (
	INPUT  = 0
	HIDDEN = 1
	BIAS   = 2
	OUTPUT = 3
)

func (n *Neuron) Process(activFunc FloatFunction) {
	if n.Type == INPUT {
		n.Output = n.Input
	} else if n.Type == BIAS {
		n.Output = 1
	} else {
		n.Input = 0
		for c := 0; c < len(n.Conns); c++ {
			if !n.Conns[c].Processed {
				n.Conns[c].Process(activFunc)
			}
			n.Input += n.Conns[c].Weight * n.Conns[c].Output
		}
		n.Output = activFunc(n.Input)
	}
	n.Processed = true
}

func (n *Neuron) SetUnprocessed() {
	n.Processed = false
	for _, c := range n.Conns {
		c.SetUnprocessed()
	}
}
