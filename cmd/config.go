package cmd

import (
	"time"

	"github.com/pkg/errors"
)

type Config struct {
	SearchParams
	Mode         string
	Origin       string
	Nq           int
	Parallel     int
	QueryFile    string
	FormatParams string
	Total        int
	OutputFormat string
	OutputFile   string
}

type SearchParams struct {
	CollectionName string   `json:"collection_name"`
	PartitionNames []string `json:"partition_names"`
	FieldName      string   `json:"fieldName"`
	IndexType      string   `json:"index_type"`
	MetricType     string   `json:"metric_type"`
	Params         struct {
		Dim int `json:"dim"`
		Ef  int `json:"ef"`
	} `json:"params"`
	Limit        int           `json:"limit"`
	Expr         string        `json:"expr"`
	OutputFields []string      `json:"output_fields"`
	Timeout      time.Duration `json:"timeout"`
}

func (c Config) Validate() error {
	if err := c.validateCommon(); err != nil {
		return err
	}
	switch c.Mode {
	case "random-vectors":
		return c.validateRandomVectors()
	case "random-text":
		return c.validateRandomText()
	case "locust":
		return c.validateDataset()
	default:
		return errors.Errorf("unrecongnized mod %q", c.Mode)
	}
}
func (c Config) validateCommon() error {
	if c.Origin == "" {
		return errors.Errorf("origin must be set")
	}
	if c.CollectionName == "" {
		return errors.Errorf("collectionName must be set")
	}

	switch c.OutputFormat {
	case "text":
		c.OutputFormat = "text"
	case "json", "":
		c.OutputFormat = "json"
	default:
		return errors.Errorf("unsupported output format %q, must be one of [text, json]",
			c.OutputFormat)
	}
	return nil
}

func (c Config) validateRandomText() error {
	return nil
}

func (c Config) validateRandomVectors() error {
	if c.Params.Dim == 0 {
		return errors.Errorf("dimension must be set and larger than 0")
	}
	return nil
}

func (c Config) validateDataset() error {
	if c.QueryFile == "" {
		return errors.Errorf("query vectors must be provided by file or json str")
	}
	return nil
}
