package tools

import (
	"bufio"
	"os"
)

var console *bufio.Reader
var inputs = make(chan string, 256)

func StartListener() {
	console = bufio.NewReader(os.Stdin)
	go func() {
		for {
			bytes, pref, err := console.ReadLine()
			check(err)
			var add []byte
			for pref {
				add, pref, err = console.ReadLine()
				bytes = append(bytes, add...)
				check(err)
			}
			input := string(bytes)
			inputs <- input
		}
	}()
}

func GetNext() (string, bool) {
	select {
	case val, ok := <-inputs:
		if ok {
			return val, true
		}
	default:
		return "", false
	}
	return "", false
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
