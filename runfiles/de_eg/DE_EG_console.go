package main

import (
	NN "FA_NeuralNetwork/network"
	MC "FA_NeuralNetwork/tools"
	"fmt"
	"os"
	"strings"
)

var nn NN.NeuralNetwork

func main() {
	// Initialisierung
	args := os.Args[1:]
	fmt.Println("Netzwerk wird geladen...")
	nn = NN.NeuralNetwork{}
	err := nn.LoadFrom("./savefiles/de_eg_mse7")
	if err != nil {
		fmt.Println(err)
	}

	if len(args) <= 0 {
		// Nutze-Eingabe als Input verwenden
		fmt.Println("Warte auf Eingabe...    Das Programm kann mit '/exit' oder ctrl+c beendet werden.")
		MC.StartListener()
		for input := MC.GetNext(); input != "/exit"; input = MC.GetNext() {
			if input != "" {
				runWith(input)
			}
		}
	} else {
		// Alle Programm-Argumente als Input verwenden
		for _, arg := range args {
			runWith(arg)
		}
	}

	fmt.Println("Programm wurde beendet.")
}

func runWith(input string) {
	str, valid := validate(input)
	if !valid {
		fmt.Printf(" Keine valide Eingabe: '%v'\n", input)
		return
	}
	out := nn.Run(stringToNetworkInput(str))
	fmt.Printf(" Input: '%v'\n", str)
	fmt.Printf("  -> De: %v\n", out[0])
	fmt.Printf("  -> Eg: %v\n", out[1])
	if out[0] > out[1] {
		fmt.Println(" Output: Deutsch")
	} else {
		fmt.Println(" Output: Englisch")
	}
	fmt.Println("")
}

func validate(str string) (string, bool) {
	str = strings.ToUpper(str)
	str = strings.Replace(str, "Ä", "AE", -1)
	str = strings.Replace(str, "Ö", "OE", -1)
	str = strings.Replace(str, "Ü", "UE", -1)
	str = strings.Replace(str, "ß", "SS", -1)
	val := len(str) <= 14
	for _, r := range []rune(str) {
		if int(r) < 65 || int(r) > 90 {
			val = false
		}
	}
	return str, val
}

func stringToNetworkInput(str string) []float64 {
	var input []float64

	runes := []rune(str)
	for _, r := range runes {
		val := int(r)
		if val < 65 || val > 90 {
			println(string(r))
			panic("Invalid rune")
		}
		input = append(input, make([]float64, val-65)...)
		input = append(input, 1)
		input = append(input, make([]float64, 26-(val-65)-1)...)
	}
	input = append(input, make([]float64, 26*14-len(input))...)

	return input
}
