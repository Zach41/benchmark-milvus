package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"

	milvusClient "github.com/xiaocai2333/milvus-sdk-go/v2/client"
	"github.com/xiaocai2333/milvus-sdk-go/v2/entity"
)

func benchmark(cfg Config, getQueryFn func() []entity.Vector) Results {
	opts := []grpc.DialOption{grpc.WithInsecure(),
		grpc.WithBlock(),                   //block connect until healthy or timeout
		grpc.WithTimeout(20 * time.Second)} // set connect timeout to 2 Second
	client, err := milvusClient.NewGrpcClient(context.Background(), cfg.Origin, opts...)
	if err != nil {
		fatal(err)
	}
	searchParams := newSearchParams(cfg.Params.Ef, cfg.IndexType)
	var times []time.Duration
	m := &sync.Mutex{}

	queues := make([][][]entity.Vector, cfg.Parallel)
	for i := 0; i < cfg.Total; i++ {
		query := getQueryFn()
		worker := i % cfg.Parallel
		queues[worker] = append(queues[worker], query)
	}
	wg := &sync.WaitGroup{}
	start := time.Now()
	for _, queue := range queues {
		wg.Add(1)
		go func(queue [][]entity.Vector) {
			defer wg.Done()
			for _, query := range queue {
				before := time.Now()
				_, err = client.Search(context.Background(), cfg.CollectionName, cfg.PartitionNames, cfg.Expr, cfg.OutputFields,
					query, cfg.FieldName, entity.MetricType(cfg.MetricType), cfg.Limit, searchParams, 1)
				if err != nil {
					fatal(err)
				}
				m.Lock()
				times = append(times, time.Since(before))
				m.Unlock()
			}
		}(queue)
	}

	wg.Wait()
	return analyze(cfg, times, time.Since(start))
}

func newSearchParams(p int, indexType string) entity.SearchParam {
	if indexType == "HNSW" {
		searchParams, err := entity.NewIndexHNSWSearchParam(p)
		if err != nil {
			panic(err)
		}
		return searchParams
	} else if indexType == "IVF_FLAT" {
		searchParams, err := entity.NewIndexIvfFlatSearchParam(p)
		if err != nil {
			panic(err)
		}
		return searchParams
	} else if indexType == "IVF_SQ8" {
		searchParams, err := entity.NewIndexIvfSQ8SearchParam(p)
		if err != nil {
			panic(err)
		}
		return searchParams
	}
	panic("illegal search params")
}

var targetPercentiles = []int{50, 90, 95, 98, 99}

type Results struct {
	Min               time.Duration
	Max               time.Duration
	Mean              time.Duration
	Took              time.Duration
	QueriesPerSecond  float64
	Percentiles       []time.Duration
	PercentilesLabels []int
	Total             int
	Successful        int
	Failed            int
	Parallelization   int
}

func analyze(cfg Config, times []time.Duration, total time.Duration) Results {
	out := Results{
		Min:               math.MaxInt64,
		PercentilesLabels: targetPercentiles,
	}

	var sum time.Duration

	for _, t := range times {
		if t < out.Min {
			out.Min = t
		}

		if t > out.Max {
			out.Max = t
		}

		out.Successful++
		sum += t
	}
	out.Total = cfg.Total
	out.Failed = cfg.Total - out.Successful
	out.Parallelization = cfg.Parallel
	out.Mean = sum / time.Duration(len(times))
	out.Took = total
	out.QueriesPerSecond = float64(len(times)) / float64(float64(total)/float64(time.Second))

	sort.Slice(times, func(a, b int) bool {
		return times[a] < times[b]
	})
	percentilePos := func(percentile int) int {
		return int(float64(len(times)*percentile)/100) + 1
	}

	out.Percentiles = make([]time.Duration, len(targetPercentiles))
	for i, percentile := range targetPercentiles {
		pos := percentilePos(percentile)
		if pos >= len(times) {
			pos = len(times) - 1
		}
		out.Percentiles[i] = times[pos]
	}

	return out
}

func (r Results) WriteTextTo(w io.Writer) (int64, error) {
	b := strings.Builder{}

	for i, percentile := range targetPercentiles {
		b.WriteString(
			fmt.Sprintf("p%q, %q\n", percentile, r.Percentiles[i]),
		)
	}
	n, err := w.Write([]byte(fmt.Sprintf(
		"Results\nSuccessful: %d\nMin: %s\nMean: %s\n%s\nTook: %s\nQPS: %f\n",
		r.Successful, r.Min, r.Mean, r.Took, b.String(), r.QueriesPerSecond)))
	return int64(n), err
}

type resultsJSON struct {
	Metadata           resultsJSONMetadata   `json:"metadata"`
	Latencies          map[string]int64      `json:"latencies"`
	LatenciesFormatted map[string]string     `json:"latencies_formatted"`
	Throughput         resultsJSONThroughput `json:"throughput"`
}

type resultsJSONMetadata struct {
	Successful      int    `json:"successful"`
	Failed          int    `json:"failed"`
	Total           int    `json:"total"`
	Parallelization int    `json:"parallelization"`
	Took            int64  `json:"took"`
	TookFormatted   string `json:"took_formatted"`
}

type resultsJSONThroughput struct {
	QPS float64 `json:"qps"`
}

func (r Results) WriteJsonTo(w io.Writer) (int, error) {
	obj := resultsJSON{
		Metadata: resultsJSONMetadata{
			Successful:      r.Successful,
			Total:           r.Total,
			Failed:          r.Failed,
			Parallelization: r.Parallelization,
			Took:            int64(r.Took),
			TookFormatted:   fmt.Sprint(r.Took),
		},
		Latencies: map[string]int64{
			"mean": int64(r.Mean),
			"min":  int64(r.Min),
		},
		LatenciesFormatted: map[string]string{
			"mean": fmt.Sprint(r.Mean),
			"min":  fmt.Sprint(r.Min),
		},
		Throughput: resultsJSONThroughput{
			QPS: r.QueriesPerSecond,
		},
	}

	for i, percentile := range targetPercentiles {
		obj.Latencies[fmt.Sprintf("p%d", percentile)] = int64(r.Percentiles[i])
		obj.LatenciesFormatted[fmt.Sprintf("p%d", percentile)] = fmt.Sprint(r.Percentiles[i])
	}

	bytes, err := json.MarshalIndent(obj, "", "  ")
	if err != nil {
		return 0, err
	}

	return w.Write(bytes)
}
