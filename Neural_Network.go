package main

import (
	"fmt"
	"github.com/anthonyjp87/dot"
	"github.com/anthonyjp87/sigmoid"
	"math/rand"
)

func main() {

	x := [][]float64{
		[]float64{0, 0, 1},
		[]float64{1, 0, 1},
		[]float64{0, 1, 1},
		[]float64{1, 1, 1},
		[]float64{0, 0, 1},
		[]float64{0, 0, 1},
		[]float64{0, 0, 1},
		[]float64{1, 1, 1},
	}
	xtrans := make([][]float64, 0)

	for h := 0; h < 3; h++ {
		xcol := make([]float64, 0)
		for k := range x {
			xcol = append(xcol, x[k][h])
		}
		xtrans = append(xtrans, xcol)
	}

	y := []float64{0, 0, 1, 1, 0, 0, 1, 1}

	fmt.Println("x", x)
	fmt.Println("xtrans", xtrans)
	fmt.Println("y", y)

	rand.Seed(int64(10))
	a := 2*rand.Float64() - 1
	b := 2*rand.Float64() - 1
	c := 2*rand.Float64() - 1
	syn0 := []float64{a, b, c}

	fmt.Println("syn0", syn0)

	for i := 0; i < 100; i++ {

		l1 := make([]float64, 0)

		for i := range x {
			l1 = append(l1, sigmoid.Sigmoid(dot.Dot(x[i], syn0), false))
		}

		l1_error := make([]float64, 0)

		for j := 0; j < len(l1); j++ {
			l1_error = append(l1_error, y[j]-l1[j])
		}
		l1_delta1 := make([]float64, 0)

		for key := 0; key < len(l1_error); key++ {
			l1_delta1 = append(l1_delta1, sigmoid.Sigmoid(l1[key], true))
		}

		l1_delta2 := make([]float64, 0)

		for n := range l1_delta1 {
			l1_delta2 = append(l1_delta2, l1_delta1[n]*l1_error[n])
		}

		syn := make([]float64, 0)

		for o := range xtrans {
			syn = append(syn, dot.Dot(xtrans[o], l1_delta2))
		}

		sy := make([]float64, 0)

		for m := 0; m < len(syn); m++ {
			sy = append(sy, syn0[m]+syn[m])
		}
		for m := 0; m < len(syn); m++ {
			syn0 = append(syn0[:m], sy[m])
		}

		// fmt.Println("l1_error", l1_error)
		// fmt.Println("l1_delta1", l1_delta1)
		// fmt.Println("l1_delta2", l1_delta2)
		// fmt.Println("syn", syn)
		fmt.Println("syn0I", syn0)
		fmt.Println("l1", l1)
	}
}
