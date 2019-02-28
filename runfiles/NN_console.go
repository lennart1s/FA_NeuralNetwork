package main

import (
	NN "FA_NeuralNetwork/network"
	NT "FA_NeuralNetwork/training"
	"bufio"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"
)

var console *bufio.Reader

var nn NN.NeuralNetwork
var td NT.TrainingData

func main() {
	commands["help"] = command{Description: "Listet alle eingetragenen Befehle. 'help [cmd]' gibt mehr Infos", Event: handleHelp}
	console = bufio.NewReader(os.Stdin)

	fmt.Println("Gestartet! Warte auf User-Input...")

	for input, _, err := console.ReadLine(); string(input) != "exit" && err == nil; input, _, err = console.ReadLine() {
		parts := strings.Split(string(input), " ")

		cmd, present := commands[parts[0]]
		if present {
			args := parts[1:]
			cmd.Event(args)
			fmt.Println("")
		} else {
			fmt.Println("Unbekannter Befehel. Versuche 'help' für eine Liste von Befehlen")
		}
	}

	fmt.Println("Programm wird gestoppt...")
}

func handleHelp(args []string) {
	if len(args) != 0 {
		for _, arg := range args {
			cmd, present := commands[arg]
			if present {
				fmt.Println("Info:", arg)
				fmt.Println("", cmd.Description)
				for _, a := range cmd.Additional {
					fmt.Printf("\t%v\n", a)
				}
			} else {
				fmt.Printf("Couldn't find command '%v'.", arg)
			}
		}
	} else {
		fmt.Println("Befehle: ")
		for k, v := range commands {
			fmt.Printf(" %s\t%s\n", k, v.Description)
		}
	}
}

func handleCreate(args []string) {
	fmt.Println("Neues Netzwerk wird erstellt:")
	var inputs, outputs int
	var hidden []int
	var wMin, wMax float64
	var useBias bool

	for _, arg := range args {
		var err error
		if strings.HasPrefix(arg, "i") {
			inputs, err = strconv.Atoi(strings.Replace(arg, "i", "", 1))
		} else if strings.HasPrefix(arg, "o") {
			outputs, err = strconv.Atoi(strings.Replace(arg, "o", "", 1))
		} else if strings.HasPrefix(arg, "h") {
			for _, s := range strings.Split(strings.Replace(arg, "h", "", 1), ",") {
				var i int
				i, err = strconv.Atoi(s)
				hidden = append(hidden, i)
			}
		} else if strings.HasPrefix(arg, "w") {
			parts := strings.Split(strings.Replace(arg, "w", "", 1), ":")
			wMin, err = strconv.ParseFloat(parts[0], 64)
			wMax, err = strconv.ParseFloat(parts[1], 64)
		} else if strings.HasPrefix(arg, "b") {
			useBias, err = strconv.ParseBool(strings.Replace(arg, "b", "", 1))
		}

		if err != nil {
			fmt.Printf("\tError parsing argument: '%v'\n", arg)
			return
		}
	}
	fmt.Println("\tInput-Neurons:", inputs)
	fmt.Println("\tHidden-Neurons:", hidden)
	fmt.Println("\tOutput-Neurons:", outputs)
	fmt.Printf("\tWeight-Range: %v:%v\n", wMin, wMax)
	fmt.Println("\tUse-Bias:", useBias)

	layers := []int{inputs}
	layers = append(layers, hidden...)
	layers = append(layers, outputs)
	fmt.Println("Creating network...")
	nn.CreateLayered(layers, useBias)
	fmt.Println("Generating weights...")
	nn.RandomizeWeights(wMin, wMax)
	fmt.Println("Netzwerk wurde erfolgreich erstellt!")
}

func handleSave(args []string) {
	if reflect.DeepEqual(nn, NN.NeuralNetwork{}) {
		fmt.Println(" No current Network. Please load or create a Network")
		return
	}
	fmt.Println(" Saving network...")
	if len(args) < 1 {
		fmt.Println("Using default path: './default'")
		nn.SaveTo("./default")
	} else {
		fmt.Printf(" Writing to '%v'...\n", args[0])
		nn.SaveTo(args[0])
	}
	fmt.Println(" Saved the current Network!")
}

func handleLoad(args []string) {
	if !reflect.DeepEqual(nn, NN.NeuralNetwork{}) {
		nn.SaveTo("./autosave")
		fmt.Println(" Autosaved previous network")
	}
	fmt.Println(" Loading Network...")
	if len(args) < 1 {
		fmt.Println(" Using default path: './default'")
		nn.LoadFrom("./default")
	} else {
		fmt.Printf(" Loading from '%v'...\n", args[0])
		nn.LoadFrom(args[0])
	}
	fmt.Println(" Loaded network!")
}

func handleRun(args []string) {
	if reflect.DeepEqual(nn, NN.NeuralNetwork{}) {
		fmt.Println(" No current Network. Please load or create a Network")
		return
	}
	if len(args) != len(nn.Inputs) {
		fmt.Printf(" Number of inputs(%v) does not match number of Input-Neurons(%v)!\n", len(args), len(nn.Inputs))
		return
	}
	var inputs []float64
	for _, arg := range args {
		val, err := strconv.ParseFloat(arg, 64)
		if err != nil {
			fmt.Printf(" Error while parsing input: '%v'\n", arg)
			return
		}
		inputs = append(inputs, val)
	}
	out := nn.Run(inputs)
	fmt.Println(" Output:", out)
}

var commands = map[string]command{
	"create": command{Description: "Erstellt ein neues NeuronalesNetz.", Event: handleCreate,
		Additional: []string{"i*\tErstellt *(int) Input-Neuronen", "o*\tErstellt *(int) Output-Neuronen",
			"h*,*...\tErstellt Hidden-Layer mit *(int) Neuronen", "w*:*\tGeneriert zufällige Gewichtungen zwischen *(float) und *(float)",
			"b*\tBenutze Bias-Neuronen: *(bool)"}},
	"exit": command{Description: "Beendet das Programm."},
	"save": command{Description: "Speichert das aktuelle Netzwerk in einer Datei(json-Format).", Event: handleSave,
		Additional: []string{"*\tSpeichert die Datei im Pfad *(string)"}},
	"load": command{Description: "Lädt ein Netzwerk aus einer Datei(json-Format). Dies überschriebt das momentane Netzwerk", Event: handleLoad,
		Additional: []string{"*\tLädt die Datei im Pfad *(string)"}},
	"run": command{Description: "Berechnet die Ausgabe des momentanen Netzwerkes mit gegebenen Input.", Event: handleRun,
		Additional: []string{"* * *...\tBerechnet die Ausgabe mit den Inputs *(float) und gibt sie aus."}},
}

type command struct {
	Description string
	Additional  []string

	Event eventFunction
}

type eventFunction func(args []string)
