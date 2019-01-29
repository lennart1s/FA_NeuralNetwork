package training

import (
	"encoding/json"
	"io/ioutil"
	"os"
)

type TrainingData struct {
	Inputs [][]float64
	Ideals [][]float64

	LearningRate float64
	Momentum     float64
}

func (td *TrainingData) SaveTo(path string) {
	bytes, err := json.MarshalIndent(td, "", "\t")
	check(err)

	file, err := os.Create(path)
	check(err)
	defer file.Close()

	_, err = file.Write(bytes)
	check(err)
}

func (td *TrainingData) LoadFrom(path string) {
	bytes, err := ioutil.ReadFile(path)
	check(err)

	err = json.Unmarshal(bytes, &td)
	check(err)
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
