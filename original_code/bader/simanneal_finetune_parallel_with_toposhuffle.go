package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Global adjacency lists
var outAdj [][][2]int32
var inAdj [][][2]int32

type set map[int64]struct{}
type result struct {
	score int
	id    int
}

var graph [][]int64
var nodeToSortedIdx map[int64]int32 = make(map[int64]int32)
var sortedIdxToNode []int64

var outLookup map[int64]map[int64]int32 = make(map[int64]map[int64]int32)
var inLookup map[int64]map[int64]int32 = make(map[int64]map[int64]int32)

var state []int32

var logs []LogEntry

// command line arguments
type CLIArgs struct {
	inFile             string
	outFile            string
	logFile            string
	graphFile          string
	maxIters           int
	infiniteIters      bool
	verbosity          int // 0: nothing, 1: best solution each iteration of outerLoop, 2: previous + final energies + "thread started", 10: everything
	updates            int
	itersPerThread     int
	threads            int
	goBackToBestWindow int
	tmin               float64
	tmax               float64
}

var cliArgs CLIArgs
var threads, itersPerThread, maxIters, updates, verbosity, goBackToBestWindow int
var tmin, tmax float64
var inPath, outPath, graphPath, logPath string

const defaultGoBackToBestWindow int = 100_000_000
const toposhuffleFrequency int = 50_000_000
const defaultUpdates int = 2500
const defaultTmin float64 = 0.001
const defaultTmax float64 = 0.1
const defaultVerbosity int = 10
const defaultIterationsToSave int = 15_000_000

func cliExit() {
	flag.Usage()
	os.Exit(1)
}

func init() {
	flag.StringVar(&cliArgs.inFile, "in", "", "Input file (CSV) - a solution to finetune")
	flag.StringVar(&cliArgs.outFile, "out", "", "Output file (CSV) - the best solution")
	flag.StringVar(&cliArgs.graphFile, "graph", "", "Graph file (CSV) - the actual graph")
	flag.StringVar(&cliArgs.logFile, "log", "", "Log file (CSV) - the data log")
	flag.IntVar(&cliArgs.threads, "threads", 0, "Number of threads")
	flag.IntVar(&cliArgs.itersPerThread, "iters-per-thread", 0, "Number of iterations per thread")
	flag.BoolVar(&cliArgs.infiniteIters, "infinite-iters", false, "Infinite iterations")
	flag.IntVar(&cliArgs.maxIters, "max-iters", 0, "Maximum number of iterations (per thread)")
	flag.IntVar(&cliArgs.verbosity, "verbosity", defaultVerbosity, "Verbosity")
	flag.IntVar(&cliArgs.updates, "updates", defaultUpdates, "Number of energy updates to perform")
	flag.IntVar(&cliArgs.goBackToBestWindow, "go-back-to-best-window", defaultGoBackToBestWindow, "Go back to best window size")
	flag.Float64Var(&cliArgs.tmin, "tmin", defaultTmin, "Minimum temperature")
	flag.Float64Var(&cliArgs.tmax, "tmax", defaultTmax, "Maximum temperature")

	flag.Parse()

	if cliArgs.inFile == "" || cliArgs.outFile == "" || cliArgs.graphFile == "" || cliArgs.threads == 0 || cliArgs.itersPerThread == 0 {
		fmt.Println("One or more required flags not provided")
		cliExit()
	}

	if !cliArgs.infiniteIters && cliArgs.maxIters == 0 {
		fmt.Println("Error: --infinite-iters or --max-iters must be specified")
		cliExit()
	}

	if cliArgs.infiniteIters && cliArgs.maxIters != 0 {
		fmt.Println("Error: --infinite-iters and --max-iters are mutually exclusive")
		cliExit()
	}

	if cliArgs.infiniteIters {
		maxIters = -1
	} else {
		maxIters = cliArgs.maxIters
	}

	if cliArgs.tmin == 0.0 {
		fmt.Println("tmin must be non-zero")
		cliExit()
	}

	if cliArgs.tmin > cliArgs.tmax {
		fmt.Println("tmin must be less than tmax")
		cliExit()
	}

	if cliArgs.verbosity < 0 || cliArgs.verbosity > 10 {
		fmt.Println("verbosity must be between 0 and 10")
		cliExit()
	}
	if cliArgs.verbosity > 2 && cliArgs.verbosity != 10 {
		fmt.Println("verbosity must be 0, 1, 2, or 10; high verbosity is 10")
		cliExit()
	}

	if cliArgs.verbosity == 10 {
		updates = cliArgs.updates
	} else {
		updates = 0
	}

	goBackToBestWindow = cliArgs.goBackToBestWindow
	threads = cliArgs.threads
	itersPerThread = cliArgs.itersPerThread

	tmin = cliArgs.tmin
	tmax = cliArgs.tmax
	verbosity = cliArgs.verbosity

	inPath = cliArgs.inFile
	outPath = cliArgs.outFile
	graphPath = cliArgs.graphFile

	if cliArgs.logFile != "" {
		logPath = cliArgs.logFile
	} else {
		logPath = cliArgs.inFile + "_log.csv"
	}
}

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}

func readCSV(file string) [][]string {
	f, err := os.Open(file)
	handleErr(err)
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	handleErr(err)

	return records
}

func writeToCSV(file string, state []int32, sortedIdxToNode []int64) {
	f, err := os.Create(file)
	handleErr(err)
	defer f.Close()

	wr := csv.NewWriter(f)
	defer wr.Flush()
	header := []string{"Node ID", "Order"}
	wr.Write(header)

	for i, node := range sortedIdxToNode {
		wr.Write([]string{strconv.FormatInt(node, 10), strconv.Itoa(int(state[i]))})
	}
}

// Add logging to a file
type LogEntry struct {
	startTime            time.Time
	iterations           int
	cumulativeIterations int
	energy               int
}

func (l LogEntry) String() string {
	return fmt.Sprintf("%d,%d,%d,%d", l.startTime.Unix(), l.iterations, l.cumulativeIterations, l.energy)
}
func (l LogEntry) Slice() []string {
	return strings.Split(l.String(), ",")
}

func writeLogs() {
	f, err := os.Create(logPath)
	handleErr(err)
	defer f.Close()

	wr := csv.NewWriter(f)
	defer wr.Flush()
	header := []string{"Start Time", "Iterations", "Cumulative Iterations", "Energy"}
	wr.Write(header)

	for _, log := range logs {
		wr.Write(log.Slice())
	}
}

func energy(state []int32) int {
	var total int64 = 0
	for _, row := range graph {
		source, target, weight := row[0], row[1], row[2]
		if state[nodeToSortedIdx[source]] < state[nodeToSortedIdx[target]] {
			total += weight
		}
	}
	return int(-total)
}

func computeChange(nodes, prev, cur []int32, outAdj [][][2]int32, inAdj [][][2]int32, state []int32) int {
	var delta int32 = 0

	a := nodes[0]
	b := nodes[1]

	for i, node := range nodes {
		prevPos := prev[i]
		curPos := cur[i]
		for _, edge := range outAdj[node] {
			nextPos := state[edge[0]]
			if edge[0] == a {
				nextPos = cur[0]
			}

			if edge[0] == b {
				nextPos = cur[1]
			}

			if prevPos < state[edge[0]] {
				delta -= edge[1]
			}
			if curPos < nextPos {
				delta += edge[1]
			}
		}

		for _, edge := range inAdj[node] {
			if edge[0] == a || edge[0] == b {
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
	return int(delta)
}

func move(state []int32) (energyDelta int, a int32, b int32) {
	n := len(state)
	// Select two distinct random indices
	a = int32(rand.Intn(n))
	b = int32(rand.Intn(n - 1))
	if b >= a {
		b++
	}

	// Capture current and proposed orders for the two nodes
	currentOrder := []int32{state[a], state[b]}
	proposedOrder := make([]int32, 2)
	copy(proposedOrder, currentOrder)
	proposedOrder[0], proposedOrder[1] = proposedOrder[1], proposedOrder[0]

	// Compute the energy change using computeChange
	indices := []int32{a, b}
	energyDelta = computeChange(indices, currentOrder, proposedOrder, outAdj, inAdj, state)

	// Apply the swap
	state[a], state[b] = state[b], state[a]

	return -energyDelta, a, b
}

func moveIfImpactful(state []int32) (energyDelta int, a int32, b int32) {
	n := len(state)
	// Select two distinct random indices
	a = int32(rand.Intn(n))
	//Pick a random node that shares an edge with a
	// var b1 int32 = int32(rand.Intn(len(outAdj[a])))
	// var b2 int32 = int32(rand.Intn(len(inAdj[a])))
	for len(outAdj[a]) == 0 && len(inAdj[a]) == 0 {
		a = int32(rand.Intn(n))
	}

	var b_i int = rand.Intn(len(outAdj[a]) + len(inAdj[a]))

	// If it's between 0 and len(outAdj[a])-1, pick an out edge
	// If it's between len(outAdj[a]) and len(outAdj[a])+len(inAdj[a])-1, pick an in edge

	if b_i < len(outAdj[a]) {
		b = outAdj[a][b_i][0]
	} else {
		b_i -= len(outAdj[a])
		b = inAdj[a][b_i][0]
	}

	// Capture current and proposed orders for the two nodes
	currentOrder := []int32{state[a], state[b]}
	proposedOrder := make([]int32, 2)
	copy(proposedOrder, currentOrder)
	proposedOrder[0], proposedOrder[1] = proposedOrder[1], proposedOrder[0]

	// Compute the energy change using computeChange
	indices := []int32{a, b}
	energyDelta = computeChange(indices, currentOrder, proposedOrder, outAdj, inAdj, state)

	// Apply the swap
	state[a], state[b] = state[b], state[a]

	return -energyDelta, a, b
}

func randomToposort(state []int32) {
	n := len(state)
	ordering := make([]int32, 0, n)
	curOutAdj := make([][][2]int32, n)
	queue := make([]int32, 0, n)
	indeg := make([]int32, n)
	for i := 0; i < n; i++ {
		indeg[i] = 0
	}

	for i := 0; i < n; i++ {
		for _, edge := range outAdj[i] {
			neighbor := edge[0]
			if state[i] < state[neighbor] {
				//totWeight += edge[1]
				curOutAdj[i] = append(curOutAdj[i], edge)
				indeg[neighbor]++
			}
		}
	}
	//fmt.Println("Weight ", totWeight)
	for i := 0; i < n; i++ {
		if indeg[i] == 0 {
			queue = append(queue, int32(i))
		}
	}
	//fmt.Println("Length of queu", len(queue))
	var totWeight int32 = 0

	for len(ordering) < n {
		qIdx := rand.Intn(len(queue))
		//qIdx := len(queue)-1
		node := queue[qIdx]
		//swap with end
		queue[qIdx], queue[len(queue)-1] = queue[len(queue)-1], queue[qIdx]
		//pop last element
		queue = queue[:len(queue)-1]
		ordering = append(ordering, node)
		for _, edge := range curOutAdj[node] {
			neighbor := edge[0]
			totWeight += edge[1]
			indeg[neighbor]--
			if indeg[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}
	fmt.Println("Weight of topsort ", totWeight)

	for i := 0; i < n; i++ {
		state[ordering[i]] = int32(i)
	}

}

func parallelAnneal(ctx context.Context, resChan chan<- result, id int, tmin float64, tmax float64, steps int, state []int32) {
	step := 0
	t := tmax
	e := energy(state)

	if tmin <= 0.0 {
		fmt.Println("Tmin needs to be nonzero")
		resChan <- result{score: e, id: id}
		return
	}

	tfactor := -math.Log(tmax / tmin)

	prevState := make([]int32, len(state))
	copy(prevState, state)
	prevEnergy := e

	bestState := make([]int32, len(state))
	copy(bestState, state)
	bestEnergy := e

	trials, accepts, improves := 0, 0, 0

	var updateWavelength int
	if updates > 1 {
		updateWavelength = steps / updates
	} else {
		updateWavelength = -1
	}

	if verbosity == 10 {
		fmt.Println("Steps: ", step, " Temperature: ", t, " Energy: ", e)
	}

	for step < steps {
		select {
		case <-ctx.Done():
			// fmt.Println("ctx cancelled")
			copy(state, bestState)
			if verbosity >= 2 {
				fmt.Println("\n", "Best energy: ", bestEnergy)
			}
			resChan <- result{score: bestEnergy, id: id}
			return
		default:
			step++

			// what to do in case tmax == tmin
			if tmax-tmin == 0.0 {
				t = tmax
			} else {
				t = tmax * math.Exp(tfactor*float64(step)/float64(steps))
			}
			// dE, a, b := move(state)
			dE, a, b := moveIfImpactful(state)

			e += dE
			trials++
			if step%goBackToBestWindow == 0 {
				copy(state, bestState)
				e = bestEnergy
			}

			if dE > 0 && math.Exp(-float64(dE)/t) < rand.Float64() && step%goBackToBestWindow != 0 {
				// Restore previous state
				state[a], state[b] = state[b], state[a]
				e = prevEnergy
			} else {
				accepts++
				if dE < 0 {
					improves++
				}

				if goBackToBestWindow != 0 {
					prevState[a], prevState[b] = prevState[b], prevState[a]
				} else {
					copy(prevState, state)
				}

				prevEnergy = e

				if e < bestEnergy {
					bestEnergy = e
					copy(bestState, state)
				}
			}
			if toposhuffleFrequency > 0 && step%toposhuffleFrequency == 0 {
				if verbosity >= 2 {
					fmt.Println("Toposorting during annealing")
				}
				randomToposort(state)
			}
			if updates > 1 {
				if updateWavelength > 0 && step%updateWavelength == 0 {
					acceptPercent := fmt.Sprintf("%2.5f%%", (float32(accepts)/float32(trials))*100)
					improvePercent := fmt.Sprintf("%2.5f%%", (float32(improves)/float32(trials))*100)
					fmt.Println("Steps: ", step, "\tTemperature: ", t, "\tEnergy: ", int(e), "\tAccept: ", acceptPercent, "\tImprove", improvePercent)
					trials, accepts, improves = 0, 0, 0
				}
			}
		}
	}
	copy(state, bestState)

	if verbosity >= 1 {
		fmt.Println("\n", "Best energy: ", bestEnergy)
	}

	resChan <- result{score: bestEnergy, id: id}
}
func main() {
	graphRecords := readCSV(graphPath)
	nodesSet := make(set)

	for i := 1; i < len(graphRecords); i++ {
		row := graphRecords[i]
		sourceIDStr, targetIDStr, edgeWeightStr := row[0], row[1], row[2]

		sourceID, err := strconv.ParseInt(sourceIDStr, 10, 64)
		handleErr(err)
		targetID, err := strconv.ParseInt(targetIDStr, 10, 64)
		handleErr(err)
		edgeWeight, err := strconv.Atoi(edgeWeightStr)
		handleErr(err)

		graph = append(graph, []int64{sourceID, targetID, int64(edgeWeight)})
		nodesSet[sourceID] = struct{}{}
		nodesSet[targetID] = struct{}{}

		setForOutNode, ok := outLookup[sourceID]
		if ok {
			setForOutNode[targetID] = int32(edgeWeight)
		} else {
			outLookup[sourceID] = make(map[int64]int32)
			outLookup[sourceID][targetID] = int32(edgeWeight)
		}

		setForInNode, ok := inLookup[targetID]
		if ok {
			setForInNode[sourceID] = int32(edgeWeight)
		} else {
			inLookup[targetID] = make(map[int64]int32)
			inLookup[targetID][sourceID] = int32(edgeWeight)
		}
	}

	sortedIdxToNode = make([]int64, len(nodesSet))
	i := 0
	for k := range nodesSet {
		sortedIdxToNode[i] = k
		i++
	}

	sort.Slice(sortedIdxToNode, func(i int, j int) bool {
		return sortedIdxToNode[i] < sortedIdxToNode[j]
	})

	for i, v := range sortedIdxToNode {
		nodeToSortedIdx[v] = int32(i)
	}

	// Initialize outAdj and inAdj based on outLookup and inLookup
	outAdj = make([][][2]int32, len(sortedIdxToNode))
	inAdj = make([][][2]int32, len(sortedIdxToNode))

	for i := 0; i < len(sortedIdxToNode); i++ {
		node := sortedIdxToNode[i]
		for target, weight := range outLookup[node] {
			outAdj[i] = append(outAdj[i], [2]int32{nodeToSortedIdx[target], weight})
		}
		for source, weight := range inLookup[node] {
			inAdj[i] = append(inAdj[i], [2]int32{nodeToSortedIdx[source], weight})
		}
	}

	// Initialize state with a random permutation
	state = make([]int32, len(nodesSet))
	for i, res := range rand.Perm(len(nodesSet)) {
		state[i] = int32(res)
	}
	state_records := readCSV(inPath)

	for i, row := range state_records {
		if i == 0 {
			continue
		}
		nodeID, err := strconv.ParseInt(row[0], 10, 64)
		handleErr(err)
		order, err := strconv.Atoi(row[1])
		handleErr(err)
		state[nodeToSortedIdx[nodeID]] = int32(order)
	}

	// Channel to listen for interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Create a context that we can cancel
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // Ensure context is canceled when main returns

	// Start a goroutine to handle the interrupt signal
	go func() {
		<-sigChan // Wait for a signal
		if verbosity >= 1 {
			fmt.Println("\nReceived interrupt signal. Stopping...")
		}
		cancel() // Cancel the context
	}()

	outerLoopCount := 0

	//Initial value log
	logs = append(logs, LogEntry{startTime: time.Now(), iterations: 0, cumulativeIterations: 0, energy: energy(state)})

	for {
		outerLoopCount++
		if maxIters > 0 && outerLoopCount > maxIters/itersPerThread {
			break
		}

		select {
		case <-ctx.Done():
			if verbosity >= 1 {
				fmt.Println("Interrupt received, saving results and exiting...")
				fmt.Println("Iterations: ", outerLoopCount*itersPerThread)
			}
			writeToCSV(outPath, state, sortedIdxToNode)
			return
		default:
			resChan := make(chan result, threads) // Buffered channel
			var wg sync.WaitGroup

			statesArr := make([][]int32, threads)
			for i := 0; i < threads; i++ {
				statesArr[i] = make([]int32, len(state))
				copy(statesArr[i], state)
			}

			for i := 0; i < threads; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					if verbosity >= 2 {
						fmt.Printf("%d-th started at "+time.Now().Format(time.RFC850)+"\n", i)
					}
					randomToposort(statesArr[i])
					parallelAnneal(ctx, resChan, i, tmin, tmax, itersPerThread, statesArr[i])
					if verbosity >= 2 {
						fmt.Printf("%d-th done at "+time.Now().Format(time.RFC850)+"\n", i)
					}
				}(i)
			}

			// Wait for all goroutines to finish or context to be cancelled
			go func() {
				wg.Wait()
				close(resChan)
			}()

			var bestEnergy int
			bestState := make([]int32, len(state))
			var bestIdx int

			// Collect results
			for res := range resChan {
				if res.score < bestEnergy {
					bestEnergy = res.score
					bestIdx = res.id
				}
			}

			// Update logs
			logs = append(logs, LogEntry{startTime: time.Now(), iterations: itersPerThread, cumulativeIterations: itersPerThread * outerLoopCount, energy: bestEnergy})

			// Check if we've been interrupted
			if ctx.Err() != nil {
				if verbosity >= 1 {
					fmt.Println("Interrupted during computation. Saving current best results...")
					fmt.Println("Best energy: ", bestEnergy)
					fmt.Println("Iterations: ", outerLoopCount*itersPerThread)
				}
				writeToCSV(outPath, statesArr[bestIdx], sortedIdxToNode)
				writeLogs()
				return
			}

			copy(bestState, statesArr[bestIdx])
			copy(state, bestState)

			if verbosity >= 1 {
				fmt.Printf("Final energy: %d\n", bestEnergy)
				fmt.Println("Iterations: ", outerLoopCount*itersPerThread)
				fmt.Println("------------------------------------------")
			}

			if itersPerThread >= defaultIterationsToSave || (itersPerThread < defaultIterationsToSave && outerLoopCount%(defaultIterationsToSave/itersPerThread) == 0) {
				if verbosity >= 2 {
					fmt.Println("Saving results...")
				}
				writeToCSV(outPath, bestState, sortedIdxToNode)
				writeLogs()
			}

			// writeToCSV(outPath, bestState, sortedIdxToNode)
		}
	}
	writeLogs()
}
