package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var globalConfig Config

func init() {
	initDataset()
}

var rootCmd = &cobra.Command{
	Use:   "benchmarker",
	Short: "Milvus Benchmarker",
	Long:  "A Milvus Benchamrker",
	Run: func(cmd *cobra.Command, args []string) {
		// fmt.Printf("Hello")
		datasetCmd.Execute()
	},
}

func Execute() {
	if err := datasetCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
