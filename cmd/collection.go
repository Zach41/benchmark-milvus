package cmd

import (
	"encoding/json"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/xiaocai2333/milvus-sdk-go/v2/entity"
)

var datasetCmd = &cobra.Command{
	Use:   "locust",
	Short: "Benchmark vectors from an existing collection",
	Long:  "Specify an existing collection as a list of query vectors in a .json file or json str to parse the query vectors and then query them with the specified parallelism",
	Run: func(cmd *cobra.Command, args []string) {
		cfg := globalConfig
		cfg.Mode = "locust"
		if err := json.NewDecoder(strings.NewReader(cfg.FormatParams)).Decode(&cfg.SearchParams); err != nil {
			fatal(err)
		}

		if err := cfg.Validate(); err != nil {
			fatal(err)
		}

		q, err := parseVectorsFromFile(cfg)
		if err != nil {
			fatal(err)
		}
		cfg.Nq = len(q)

		var w io.Writer
		if cfg.OutputFile == "" {
			w = os.Stdout
		} else {
			f, err := os.Create(cfg.OutputFile)
			if err != nil {
				fatal(err)
			}

			defer f.Close()
			w = f
		}
		result := benchmarkDataset(cfg, q)
		if cfg.OutputFormat == "json" {
			result.WriteJsonTo(w)
		} else if cfg.OutputFormat == "text" {
			result.WriteTextTo(w)
		}

		if cfg.OutputFile != "" {
			infof("results successfully written to %q", cfg.OutputFile)
		}
	},
}

func initDataset() {
	rootCmd.AddCommand(datasetCmd)

	// TODO: use parameters to distinguish file or json str
	datasetCmd.PersistentFlags().StringVarP(&globalConfig.Origin,
		"Origin", "u", "", "host for Milvus")
	datasetCmd.PersistentFlags().StringVarP(&globalConfig.QueryFile,
		"queryFile", "q", "", "Point to the queries file, (.json)")
	datasetCmd.PersistentFlags().StringVarP(&globalConfig.FormatParams,
		"searchParams", "s", "", "params for operation")
	datasetCmd.PersistentFlags().IntVarP(&globalConfig.Parallel,
		"parallel", "p", 1, "Set the number of parallel threads which send queries")
	datasetCmd.PersistentFlags().StringVarP(&globalConfig.OutputFormat,
		"format", "f", "text", "Output format, one of [text, json]")
	datasetCmd.PersistentFlags().IntVarP(&globalConfig.Total,
		"total", "t", 1, "run times for test")

	//datasetCmd.PersistentFlags().StringVarP(&globalConfig.OutputFile,
	//	"output", "o", "", "Filename for an output file. If none provided, output to stdout only")

}

type Queries [][]float32

func parseVectorsFromFile(cfg Config) (Queries, error) {
	var q Queries
	if strings.Contains(cfg.QueryFile, ".json") {
		f, err := os.Open(cfg.QueryFile)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		if err := json.NewDecoder(f).Decode(&q); err != nil {
			return nil, err
		}
	} else {
		if err := json.NewDecoder(strings.NewReader(cfg.QueryFile)).Decode(&q); err != nil {
			return nil, err
		}
	}
	return q, nil
}

func benchmarkDataset(cfg Config, queries Queries) Results {
	getQueryFunc := func() []entity.Vector {
		vectors := make([]entity.Vector, 0)
		for _, query := range queries {
			vectors = append(vectors, entity.FloatVector(query))
		}
		return vectors
	}
	return benchmark(cfg, getQueryFunc)
}
