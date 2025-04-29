package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"
	"sort"
)

type Edge struct {
	Source, Target, Weight int
}

func createAdjacencyLists(csvFilePath string) ([][][2]int, [][][2]int, []int, map[int]int, map[int]int) {
	nodeSet := make(map[int]struct{})
	
	file, err := os.Open(csvFilePath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	_, err = reader.Read() // Skip header
	if err != nil {
		log.Fatal(err)
	}

	for {
		row, err := reader.Read()
		if err != nil {
			break
		}

		source, _ := strconv.Atoi(row[0])
		target, _ := strconv.Atoi(row[1])

		nodeSet[source] = struct{}{}
		nodeSet[target] = struct{}{}
	}

	var nodeList []int
	for node := range nodeSet {
		nodeList = append(nodeList, node)
	}
	sort.Ints(nodeList)

	nodeToIndex := make(map[int]int)
	indexToNode := make(map[int]int)
	for idx, node := range nodeList {
		nodeToIndex[node] = idx
		indexToNode[idx] = node
	}

	n := len(nodeList)
	outAdj := make([][][2]int, n)
	inAdj := make([][][2]int, n)

	// Re-opening the file to parse edges
	file.Seek(0, 0)
	reader = csv.NewReader(file)
	_, err = reader.Read() // Skip header again
	if err != nil {
		log.Fatal(err)
	}

	for {
		row, err := reader.Read()
		if err != nil {
			break
		}

		source, _ := strconv.Atoi(row[0])
		target, _ := strconv.Atoi(row[1])
		weight, _ := strconv.Atoi(row[2])

		srcIdx := nodeToIndex[source]
		tgtIdx := nodeToIndex[target]
		outAdj[srcIdx] = append(outAdj[srcIdx], [2]int{tgtIdx, weight})
		inAdj[tgtIdx] = append(inAdj[tgtIdx], [2]int{srcIdx, weight})
	}

	return outAdj, inAdj, nodeList, nodeToIndex, indexToNode
}

func generateRandomPermutation(n int) ([]int, []int) {
	permutation := rand.Perm(n)
	pos := make([]int, n)
	for idx, node := range permutation {
		pos[node] = idx
	}
	return permutation, pos
}

func writeOrderedNodesToCSV(permutation, pos []int, outputCSVFile string, indexToNode map[int]int) {
	file, err := os.Create(outputCSVFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"node_id", "order in the permutation"})
	for idx := range permutation {
		node := indexToNode[idx]
		writer.Write([]string{strconv.Itoa(node), strconv.Itoa(pos[idx])})
	}
}

func initialScore(outAdj [][][2]int, state []int) int {
	score := 0
	for i := range outAdj {
		for _, edge := range outAdj[i] {
			if state[i] < state[edge[0]] {
				score += edge[1]
			}
		}
	}
	return score
}

func computeChange(nodes, prev, cur []int, outAdj [][][2]int, inAdj [][][2]int, state []int) int {
	delta := 0
	actual := make(map[int]int)
	for i, node := range nodes {
		actual[node] = cur[i]
	}

	for i, node := range nodes {
		prevPos := prev[i]
		curPos := cur[i]
		for _, edge := range outAdj[node] {
			nextPos := state[edge[0]]
			if target, found := actual[edge[0]]; found {
				nextPos = target
			}
			if prevPos < state[edge[0]] {
				delta -= edge[1]
			}
			if curPos < nextPos {
				delta += edge[1]
			}
		}
		for _, edge := range inAdj[node] {
			if _, found := actual[edge[0]]; found {
				continue
			}
			if state[edge[0]] < prevPos {
				delta -= edge[1]
			}
			if state[edge[0]] < curPos {
				delta += edge[1]
			}
		}
	}
	return delta
}

func move(n int, state []int, outAdj [][][2]int, inAdj [][][2]int) int {
	//return 0
	i := rand.Intn(n)
	j := rand.Intn(n-1)
	if j >= i {
		j++
	}
	indices := []int{i, j}
	_ = indices
	currentOrder := make([]int, 2)
	for i, idx := range indices {
		currentOrder[i] = state[idx]
	}

	bestOrder := make([]int, len(currentOrder))
	copy(bestOrder, currentOrder)
	bestDelta := 0
	

	perms := permutations(currentOrder)
	for _, perm := range perms {
		delta := computeChange(indices, currentOrder, perm, outAdj, inAdj, state)
		if delta > bestDelta {
			bestDelta = delta
			copy(bestOrder, perm)
		}
	}
	
	for i, node := range indices {
		state[node] = bestOrder[i]
	}

	return bestDelta
}

func permutations(arr []int) [][]int {
	var res [][]int
	permute(arr, 0, &res)
	return res
}

func permute(arr []int, l int, res *[][]int) {
	if l == len(arr)-1 {
		cpy := make([]int, len(arr))
		copy(cpy, arr)
		*res = append(*res, cpy)
		return
	}

	for i := l; i < len(arr); i++ {
		arr[l], arr[i] = arr[i], arr[l]
		permute(arr, l+1, res)
		arr[l], arr[i] = arr[i], arr[l]
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	csvFilePath := ""
	outputCSVFile := ""
	outAdj, inAdj, nodeList, nodeToIndex, indexToNode := createAdjacencyLists(csvFilePath)
	_ = nodeToIndex

	n := len(nodeList)
	permutation, pos := generateRandomPermutation(n)

	file, err := os.Open("")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	_, err = reader.Read() // Skip header
	if err != nil {
		log.Fatal(err)
	}

	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		node, _ := strconv.Atoi(record[0])
		order, _ := strconv.Atoi(record[1])
		pos[nodeToIndex[node]] = order
	}

	state := pos
	score := initialScore(outAdj, state)

	numIters := 1000000000
	outputFrac := 100

	for iter := 0; iter < numIters; iter++ {
		delta := move(n, state, outAdj, inAdj)
		if delta == -1 {
			break
		}
		score += delta

		if iter%outputFrac == 0 {
			fmt.Printf("Iteration %d, score: %d\n", iter, score)
		}
	}

	writeOrderedNodesToCSV(permutation, pos, outputCSVFile, indexToNode)

	fmt.Printf("Ordered nodes saved to %s\n", outputCSVFile)
	fmt.Printf("Final score: %d\n", score)
	fmt.Printf("Actual score: %d\n", initialScore(outAdj, state))
}
