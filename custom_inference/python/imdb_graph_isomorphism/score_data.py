import avro.io
import avro
from avro.datafile import DataFileReader, DataFileWriter
from io import BytesIO, BufferedWriter
import requests
import pandas as pd

def load_schema(schema_path): 
    schema = avro.schema.parse(open(schema_path, "rb").read())
    return schema


def score(graphs, schema, url, port):
    """
    graphs is expected to be a list of dictionaries, where each entry in the 
    list represents a graph with 
    * key idx -> index value
    * key nodes -> list of ints representing vertices of the graph
    * key edges -> list of list of ints representing edges of graph
    """
    
    stream = BufferedWriter(BytesIO())
    writer = DataFileWriter(stream, avro.io.DatumWriter(), schema)
    # writer = DataFileWriter(open("imdb-graph.avro", "wb"), DatumWriter(), schema)
    for graph in graphs:
        writer.append({"edges": graph["edges"], "vertices": graph["vertices"], "index": graph["idx"], "label": graph.get("label")})
        writer.flush()
    raw_bytes = stream.raw.getvalue()
    writer.close()
    
    url = "{}:{}/predictUnstructured/?ret_mode=binary".format(url.strip("/"), port)

    payload = raw_bytes
    headers = {
      'Content-Type': 'application/octet-stream'
    }

    response = requests.request("POST", url, headers=headers, data = payload)

    return response
