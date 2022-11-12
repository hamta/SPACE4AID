import networkx as nx
import random
from collections import defaultdict
import numpy as np
import generate_json_branch_partition_version
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Your codes ....
dir, Dict=generate_json_branch_partition_version.generate_system("",1)
with open(dir, 'w') as fp:
        json.dump(Dict, fp, cls=NpEncoder, indent=4)

is_DAG=False
Seed=20
while not is_DAG:
    G = nx.gnp_random_graph(10,0.5,seed=Seed,directed=True)
    DAG = nx.DiGraph([(u,v,{'weight':random.random()}) for (u,v) in G.edges() if u<v])
    is_DAG = nx.is_directed_acyclic_graph(DAG)
    Seed += 1
edge_list=list(DAG.in_edges())
Edge_weight_list=[]
for edge in edge_list:
    weight=DAG.get_edge_data(edge[0], edge[1])['weight']
    Edge_weight_list.append((edge[0], edge[1], weight))


d = defaultdict(list)

for k, *v in Edge_weight_list:
    d[k].append(v)

LL=list(d.items())
n={}
for edges in LL:
    m={}
    next_list=[]
    trans_list=[]
    for des in edges[1]:
        next_list.append("c"+str(des[0]+1))
        trans_list.append(des[1])
    m["next"]=next_list
    m["transition_probability"]=trans_list
    #m["data_size"]=edge[3]
    n["c"+str(edges[0]+1)]=m


GC_number=3
paths_list=[]
nodes=list(DAG.nodes)
while len(paths_list) < GC_number:
    source=np.random.choice(nodes)
    destination=np.random.choice(nodes)
    if not source == destination:
        paths=[path for path in nx.all_simple_paths(DAG, source, destination)]
        if len(paths)>0:
            paths_list.append(np.random.choice(paths))

nx.draw(DAG)

x=1
