import pandas as pd
import os
from pathlib import Path
import pickle
import dgl 
import torch
from graph_isomorphism_network import *
from torch.utils.data import DataLoader

# def init(code_dir):

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels
    


def load_model(code_dir):
    model = GIN(2, 2, 1, 20, 2, 0, 0.01, "sum", "sum")
    model.load_state_dict(torch.load(os.path.join(code_dir, "gin_model.h5")))
    return model

def fit(X, y, output_dir, **kwargs):

    model = GIN(2, 2, 1, 20, 2, 0, 0.01, "sum", "sum")
    dgl_graphs = X["dgl_graph"].values
    dgl_graphs = list( map ( lambda x: pickle.loads(eval(x)), dgl_graphs))

    dataset = []
    for g, l in zip(dgl_graphs, y.values):
        num_nodes = g.num_nodes()
        g.ndata["attr"] = torch.ones(g.num_nodes(), 1)
        g.ndata["label"] = torch.ones(num_nodes, ) if l == 1 else torch.zeros(num_nodes, )
        dataset.append((g, torch.tensor(l)))


    dataloader = DataLoader(dataset,batch_size=1024,collate_fn=collate,drop_last=False,shuffle=True)

    opt = torch.optim.Adam(model.parameters(),lr=0.01)

    for epoch in range(500):
        for batched_graph, label in dataloader:
            feats = batched_graph.ndata['attr'].float()
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, label)
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 100 == 0:
            print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

    output_dir_path = Path(output_dir)
    if output_dir_path.exists() and output_dir_path.is_dir():
        torch.save(model.state_dict(), "{}/gin_model.h5".format(output_dir))

def score(data, model, **kwargs): 
    dgl_graphs = data["dgl_graph"].values
    pos_class = kwargs["positive_class_label"]
    neg_class = kwargs["negative_class_label"]
    dgl_graphs = list( map ( lambda x: pickle.loads(eval(x)), dgl_graphs))
    for g in dgl_graphs:
        g.ndata["attr"] = torch.ones(g.num_nodes(), 1)
    batched_graph = dgl.batch(dgl_graphs)
    feats = batched_graph.ndata['attr'].float()
    preds = F.softmax(model(batched_graph, feats), dim=1).detach().numpy()
    return pd.DataFrame(preds, columns = [neg_class, pos_class])
