package tools

import (
	"bufio"
	"os"
)

var console *bufio.Reader
var inputs = make(chan string, 256)

/*StartListener startet eine
Nutzer-Eingabe-Abfrage. Alle
regestrierten Eingaben werden
in einem String-Channel gespeichert.*/
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

/*GetNext gibt den nächsten User-Input aus.
Ist der Channel leer wird auf die nächste
Eingabe gewartet. Folglich kann diese
Funktion blockieren.*/
func GetNext() string {
	return <-inputs
}

/*HasNext gibt ein bool zurück,
ob eine Eingabe vorhanden ist.
Ist dies der Fall wird auch die
Eingabe zurück gegeben.*/
func HasNext() (string, bool) {
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
