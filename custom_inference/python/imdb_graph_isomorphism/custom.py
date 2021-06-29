#import pandas as pd
import torch
import os
import io
from io import BytesIO
import avro.io
import avro
from avro.datafile import DataFileReader, DataFileWriter
from gin import *
import dgl
import pandas as pd

def load_model(input_dir):
   """
   This hook can be implemented to adjust logic in the scoring mode.

   load_model hook provides a way to implement model loading your self.
   This function should return an object that represents your model. This object will
   be passed to the predict hook for performing predictions.
   This hook can be used to load supported models if your model has multiple artifacts, or
   for loading models that drum does not natively support

   :param input_dir: the directory to load serialized models from
   :returns: Object containing the model - the predict hook will get this object as a parameter
   """
   model = GIN(2, 2, 1, 20, 2, 0, 0.01, "sum", "sum")
   model.load_state_dict(torch.load(os.path.join(input_dir, "gin_model.h5")))
   return model

def score_unstructured(model, data, query, **kwargs):
   print("Incoming content type params: ", kwargs)
   print("Incoming data type: ", type(data))
   print("Incoming query params: ", query)
    
   bytes_reader = io.BytesIO(data)
   parsed_data = DataFileReader(bytes_reader, avro.io.DatumReader())

   gs = []
   for graph in parsed_data:
      e = graph["edges"] 
      u,v = list(zip(*e))
      g = dgl.graph((u,v))
      g.ndata["attr"] = torch.ones(g.num_nodes(), 1)
      gs.append(g)
   batched_graph = dgl.batch(gs)
   feats = batched_graph.ndata['attr'].float()
   preds = F.softmax(model(batched_graph, feats), dim=1).detach().numpy()
   return pd.DataFrame(preds, columns = ["neg_class", "pos_class"]).to_json(orient="records")
