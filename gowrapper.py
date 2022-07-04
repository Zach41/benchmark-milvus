from pymilvus import (
    Collection,Index
)
import subprocess, json, os

class BenchMarker(Collection):

    def __init__(self, name, schema=None, using="default", shards_num=2, **kwargs):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        super().__init__(name, schema, using, shards_num, **kwargs)
        self.indexes_set = dict()
        self._gopkg = dir_path
    def set_parallel(self, parallel):
        self._parallel = parallel
    
    def set_gopkg(self, gopkg):
        self._gopkg = gopkg

    def create_index(self, field_name, index_params={}, timeout=None, **kwargs) -> Index:
        self.indexes_set[field_name] = index_params
        return super().create_index(field_name, index_params, timeout, **kwargs)

    def benchmark_search(self, data, anns_field, param, limit, expr=None, partition_names=None, output_fields=None, timeout=None, round_decimal=-1, **kwargs):
        index_info = self.indexes_set[anns_field]
        index_type = ""
        if index_info is not None:
            index_type = index_info["index_type"]
        params = param["params"]
        if params is not None:
            params["ef"] = params["nprobe"]
        query_json = {
            "collection_name": self.name, 
            "partition_names": partition_names,
            "fieldName": anns_field,
            "metric_type": param["metric_type"],
            "index_type": index_type,
            "params": params,
            "limit": limit,
            "expr": expr,
            "output_fields": output_fields,
            "timeout": timeout,
        }
        conn = super()._get_connection()
        process = subprocess.Popen(
            cwd=self._gopkg,
            args=['go', 'run', '.', 'locust', '-u', conn.server_address, '-q', json.dumps(data, indent=2),
            '-s', json.dumps(query_json, indent=2), '-p', str(self._parallel), '-f', 'json', '-t', str(10000)],
            stdout=subprocess.PIPE)
        result_raw = process.communicate()[0].decode('utf-8')
        print(result_raw)