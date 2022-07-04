import time
from gowrapper import BenchMarker
import h5py
import subprocess
import json
from pymilvus import Collection, CollectionSchema, connections, FieldSchema, DataType, utility
from gowrapper import BenchMarker

coll_name = "test"
vec_name = "vec"
default_index = {"index_type": "HNSW", "params": {"M": 16, "efConstruction": 128}, "metric_type": "L2"}
fname = "./benchmark-data/glove-25-angular.hdf5"

topKs = [1, 10, 100]
ef_array = [64, 128, 256, 512]

def create_collections():
    int64 = FieldSchema("int64", DataType.INT64, auto_id=False, is_primary=True)
    vec_field = FieldSchema(vec_name, DataType.FLOAT_VECTOR, dim=25)
    coll = BenchMarker(coll_name, CollectionSchema([int64, vec_field]))
    return coll


def parse_vectors():
    vector_write_array = []
    with h5py.File(fname, 'r') as f:
        # train_vectors_len = len(f['train'])
        # test_vectors_len = len(f['test'])
        i = 0
        for vector in f['train']:
            vector_write_array.append(vector.tolist())
            i += 1
    return vector_write_array


def insert(coll, vectors):
    i = 0
    per_batch = 10000
    vector_len = len(vectors)
    while vector_len > 0:
        ids = []
        batch_vectors = []
        if vector_len > per_batch:
            ids = [x+i for x in range(per_batch)]
            batch_vectors = vectors[i:i+per_batch]
            i += per_batch
            vector_len -= per_batch
        else:
            ids = [x+i for x in range(vector_len)]
            batch_vectors = vectors[i:i+vector_len]
            i += vector_len
            vector_len -= vector_len
        coll.insert([ids, batch_vectors])
        print("insert num_rows: ", i)

def search(coll):
    with h5py.File(fname, 'r') as f:
        vector_write_array = []
        for vector in f['test']:
            vector_write_array.append(vector.tolist())
        coll.set_parallel(144)
        coll.set_total(10000)
        i = 0
        for ef in ef_array:
            for k in topKs:
                print("Case%d: <TopK %d, NProbe %d>" % (i, k, ef))
                coll.benchmark_search([vector_write_array[0]], vec_name, {"metric_type": "L2", "params":{"nprobe": ef}},k)
                i = i + 1
                # run_speed_test(k, 144, "localhost:19530", [vector_write_array[0]], ef)


if __name__ == "__main__":
    connections.connect(host="localhost", port=19530)

    coll = create_collections()

    print("drop collection...")
    coll.drop()
    print("drop collection done!")
    coll = create_collections()

    print("create index...")
    createidxParams = {"sync": True, "async": False}
    coll.create_index(vec_name, index_params=default_index, **createidxParams)
    print("create index done!")

    print("insert...")
    insert(coll, parse_vectors())
    print("insert done...")

    print("create index...")
    coll.create_index(vec_name, index_params=default_index, **createidxParams)
    utility.wait_for_index_building_complete(coll_name)
    # time.sleep(60)
    print("create index done!")

    print("load collection...")
    coll.load()
    print("load collection done!")

    print("do search...")
    search(coll)
    print("search done!")

    print("drop collection...")
    coll.drop()
    print("drop collection done!")




