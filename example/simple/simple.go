package main

import (
	"context"
	"fmt"
	"log"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/zilliztech/milvus_benchmark/milvus_benchmark/benchmarker/internal/hdf5"
)

const (
	collName = "hdf5_test"
)

func main() {
	ctx := context.Background()
	fmt.Println("start connecting to Milvus")
	client, err := client.NewGrpcClient(ctx, "localhost:19530")
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	has, err := client.HasCollection(ctx, collName)
	if err != nil {
		log.Fatal(err)
	} else if has {
		client.DropCollection(ctx, collName)
	}

	fmt.Printf("create collection `%s`\n", collName)
	schema := &entity.Schema{
		CollectionName: collName,
		Description:    "hdf5_test",
		AutoID:         false,
		Fields: []*entity.Field{
			{
				Name:       "int64",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
				AutoID:     false,
			},
			{
				Name:     "vec",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					entity.TYPE_PARAM_DIM: "25",
				},
			},
		},
	}
	if err := client.CreateCollection(ctx, schema, 2); err != nil {
		log.Fatal(err)
	}
	fmt.Println("create index on `vec`")
	idx, err := entity.NewIndexIvfFlat(entity.L2, 128)
	if err != nil {
		log.Fatal(err)
	}
	if err := client.CreateIndex(ctx, collName, "vec", idx, false); err != nil {
		log.Fatal(err)
	}
	if err = client.LoadCollection(ctx, collName, false); err != nil {
		log.Fatal(err)
	}
	fmt.Println("insert data...")

	batchInsert(ctx, client)

	if err := client.Flush(ctx, collName, false); err != nil {
		log.Fatalf("failed to flush data, err: %v", err)
	}

	if err := client.LoadCollection(ctx, collName, false); err != nil {
		log.Fatal(err)
	}

	fmt.Println("do search query")
	dosearch(ctx, client)
}

func batchInsert(ctx context.Context, client client.Client) {
	h5obj, err := hdf5.Open("../../benchmark-data/glove-25-angular.hdf5")
	if err != nil {
		log.Fatal(err)
	}
	defer h5obj.Close()
	trainDataset, err := h5obj.Read("/train")
	if err != nil {
		log.Fatal(err)
	}
	trainList, err := trainDataset.ReadFloatMatrix()
	if err != nil {
		log.Fatal(err)
	}
	batch, total := 10000, 0
	for offset := 0; offset < len(trainList); offset += batch {
		curBatch := batch
		if curBatch > len(trainList)-offset {
			curBatch = len(trainList) - offset
		}
		idList := make([]int64, 0, curBatch)
		embeddingList := make([][]float32, 0, curBatch)

		for i := 0; i < curBatch; i++ {
			idList = append(idList, int64(i))
			embeddingList = append(embeddingList, trainList[offset+i])
		}
		idColData := entity.NewColumnInt64("int64", idList)
		embeddingColData := entity.NewColumnFloatVector("vec", 25, embeddingList)
		if _, err := client.Insert(ctx, collName, "", idColData, embeddingColData); err != nil {
			log.Fatal(err)
		}
		total += curBatch
		fmt.Printf("inserted %d rows, total: %d\n", curBatch, total)
	}

}

func dosearch(ctx context.Context, client client.Client) {
	h5obj, err := hdf5.Open("../../benchmark-data/glove-25-angular.hdf5")
	if err != nil {
		log.Fatal(err)
	}
	defer h5obj.Close()
	testDataset, err := h5obj.Read("/test")
	if err != nil {
		log.Fatal(err)
	}
	testList, err := testDataset.ReadFloatMatrix()
	if err != nil {
		log.Fatal(err)
	}
	sp, err := entity.NewIndexFlatSearchParam(10)
	if err != nil {
		log.Fatal(err)
	}
	batch, casz := 10, 0
	for offset := 0; offset < len(testList); offset += batch {
		curBatch := batch
		if curBatch > len(testList)-offset {
			curBatch = len(testList) - offset
		}
		vecs2search := make([]entity.Vector, 0, curBatch)
		for i := 0; i < curBatch; i++ {
			vecs2search = append(vecs2search, entity.FloatVector(testList[offset+i]))
		}
		rets, err := client.Search(ctx, collName, nil, "", []string{"int64"}, vecs2search, "vec", entity.L2, 100, sp)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("batch search %d results:\n", casz)
		for _, ret := range rets {
			printResult(&ret)
		}
		casz += 1
	}
}

func printResult(sRet *client.SearchResult) {
	ids := make([]int64, 0, sRet.ResultCount)
	scores := make([]float32, 0, sRet.ResultCount)

	var idCol *entity.ColumnInt64
	for _, field := range sRet.Fields {
		if field.Name() == "int64" {
			c, ok := field.(*entity.ColumnInt64)
			if ok {
				idCol = c
			}
		}
	}
	for i := 0; i < sRet.ResultCount; i++ {
		val, err := idCol.ValueByIdx(i)
		if err != nil {
			log.Fatal(err)
		}
		ids = append(ids, val)
		scores = append(scores, sRet.Scores[i])
	}
	fmt.Printf("IDs: %v, scores: %v\n", ids, scores)
}
