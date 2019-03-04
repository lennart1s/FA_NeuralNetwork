package main

import (
	NN "FA_NeuralNetwork/network"
	NT "FA_NeuralNetwork/training"
	"fmt"
	"io/ioutil"
	"math/rand"
	"strings"
	"time"
)

var germanWords []string
var englishWords []string

var minLetters = 4
var maxLetters = 14

var trainingSetSize = 600

func main() {
	loadTextFiles()

	//nn := createNetwork()
	//nn.SaveTo("./savefiles/pre.nn")
	nn := NN.NeuralNetwork{}
	nn.LoadFrom("./savefiles/autosave")

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

	//in := stringToNetworkInput("WALKING")
	//fmt.Println(nn.Run(in))

	td := createTrainingData()

	fmt.Println("Starting training...")
	count := 0
	for err := NT.MeanSquaredError(&nn, td); err > 0.005; NT.Backpropagation(&nn, td) {
		err = NT.MeanSquaredError(&nn, td)
		count++
		if count%4 == 0 {
			fmt.Println(err, "|", nn.BackPropRuns)
			td = createTrainingData()
		}
		if count%8 == 0 {
			nn.SaveTo("./savefiles/autosave")
			fmt.Println("Autosaved network!")
		}
	}

	nn.SaveTo("./savefiles/german_english_below0_5.nn")
}

func createNetwork() NN.NeuralNetwork {
	nn := NN.NeuralNetwork{}
	nn.CreateLayered([]int{maxLetters * 26, maxLetters * 26, 2}, true)
	nn.RandomizeWeights(-3, 3)

	return nn
}

func createTrainingData() NT.DataSet {
	rand.Seed(int64(time.Now().Nanosecond() * time.Now().Minute()))
	td := NT.DataSet{}
	td.LearningRate = 0.0005
	td.Momentum = 0.00002
	for i := 0; i < trainingSetSize/2; i++ {
		german, gVal := validate(germanWords[rand.Intn(len(germanWords))])
		for !gVal {
			german, gVal = validate(germanWords[rand.Intn(len(germanWords))])
		}
		td.Inputs = append(td.Inputs, stringToNetworkInput(german))
		td.Ideals = append(td.Ideals, []float64{1, 0})

		english, eVal := validate(englishWords[rand.Intn(len(englishWords))])
		for !eVal {
			english, eVal = validate(englishWords[rand.Intn(len(englishWords))])
		}
		td.Inputs = append(td.Inputs, stringToNetworkInput(english))
		td.Ideals = append(td.Ideals, []float64{0, 1})

	}

	return td
}

func loadTextFiles() {
	data, err := ioutil.ReadFile("./resources/german_words.txt")
	check(err)
	germanWords = strings.Split(string(data), "\r\n")

	data, err = ioutil.ReadFile("./resources/english_words.txt")
	check(err)
	englishWords = strings.Split(string(data), "\r\n")

	fmt.Println("Successfully read text-files!")
}

func validate(str string) (string, bool) {
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
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
