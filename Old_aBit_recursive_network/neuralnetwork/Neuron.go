package neuralnetwork

import UUID "github.com/google/uuid"

type Neuron struct {
	Id UUID.UUID

	Type int

	Layer int

	Conns []Connection

	Input  float64
	Output float64

	Processed bool

	Delta                  float64
	PrevLayerWeightedDelta float64
}

func NewNeuron(Type int) Neuron {
	id, _ := UUID.NewUUID()
	neuron := Neuron{Type: Type, Id: id}
	return neuron
}

func (n *Neuron) ConnectTo(o *Neuron) {
	c := Connection{UpperNeuron: o, UpperNeuronID: o.Id}
	n.Conns = append(n.Conns, c)
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
			if !n.Conns[c].UpperNeuron.Processed {
				n.Conns[c].UpperNeuron.Process(activFunc)
			}
			n.Input += n.Conns[c].Weight * n.Conns[c].UpperNeuron.Output
		}
		n.Output = activFunc(n.Input)
	}
	n.Processed = true
}
