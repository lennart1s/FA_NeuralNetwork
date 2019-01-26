package neuralnetwork

type Neuron struct {
	Type int

	Layer int

	Conns []Connection

	Input     float64
	Output    float64
	Processed bool
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
		n.Output = 0
		for c := 0; c < len(n.Conns); c++ {
			if !n.Conns[c].UpperNeuron.Processed {
				n.Conns[c].UpperNeuron.Process(activFunc)
			}
			n.Output += n.Conns[c].Weight * n.Conns[c].UpperNeuron.Output
		}
		n.Output = activFunc(n.Output)
	}
	n.Processed = true
}
