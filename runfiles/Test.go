package main

import (
	MC "FA_NeuralNetwork/tools"
	"fmt"
	"time"
)

func maian() {
	MC.StartListener()
	/*time.Sleep(8 * time.Second)
	val, ok := MC.GetNext()
	if ok {
		fmt.Println(val)
	}
	val, ok = MC.GetNext()
	if ok {
		fmt.Println(val)
	}*/
	for {
		val, ok := MC.GetNext()
		if ok {
			fmt.Println(val)
		} else {
			fmt.Print(".")
		}
		time.Sleep(500 * time.Millisecond)
	}
}
