package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	console := bufio.NewReader(os.Stdin)

	fmt.Println("Gestartet! Warte auf User-Input...")

	for input, _, err := console.ReadLine(); string(input) != "exit" && isOk(err); input, _, err = console.ReadLine() {
		cmd := string(input)

		fmt.Println("Entered:", cmd)
	}

	fmt.Println("Programm wird gestoppt...")

	//nn := createNetwork()
	//nn.SaveTo("./savefiles/pre.nn")
	//nn := NN.NeuralNetwork{}
	//nn.LoadFrom("./realBelow10")

	/*correct := 0

	td := createTrainingData()
	for i := 0; i < len(td.Inputs); i++ {
		out := nn.Run(td.Inputs[i])
		if td.Ideals[i][0] == 1 && out[0] > out[1] {
			correct++
		} else if td.Ideals[i][1] == 1 && out[1] > out[0] {
			correct++
		}
	}
	fmt.Println(correct, "/", len(td.Inputs))*/
}

/*func validate(str string) (string, bool) {
	str = strings.ToUpper(str)
	str = strings.Replace(str, "Ä", "AE", -1)
	str = strings.Replace(str, "Ö", "OE", -1)
	str = strings.Replace(str, "Ü", "UE", -1)
	str = strings.Replace(str, "ß", "SS", -1)
	val := len(str) >= minLetters && len(str) <= maxLetters
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
	input = append(input, make([]float64, 26*maxLetters-len(input))...)

	return input
}*/

func isOk(err error) bool {
	if err != nil {
		panic(err)
		return false
	}
	return true
}
