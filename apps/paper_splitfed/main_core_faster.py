import copy

import torch
from matplotlib import pyplot as plt

from apps.paper_splitfed.SplitFed import SplitFed
from apps.paper_splitfed.core import clusters
from src.apis import lambdas
from src.apis.federated_tools import aggregate
from src.data.data_loader import preload
from apps.main_split import models, funcs, dist
from apps.main_split.server import Server

# the max number of client in each cluster, use to avoid accuracy problem and cat. for.
cluster_limit = 4
# initialize clients, it should be replaced with weight divergence analysis.
# In this case, we have 5 clusters in each we have x clients
client_cluster_size = 20

# sr=slow_rate: probability of a created clients to be considered as slow with low capabilities
# increase virtual training time
sr = 0.4
rounds = 100

# init data
client_model = models.MnistClient(784, 32, 10)
server_model = models.MnistServer(784, 32, 10)
clients_data = preload('mnist', dist.clustered(client_cluster_size, 300), tag=f'cluster{client_cluster_size}p{300}')
train_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[0]).map(lambdas.as_tensor)
test_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[1]).reduce(lambdas.dict2dc).as_tensor()

# split learning
fast_clients, slow_clients = clusters.generate_speed(train_data, client_model, client_cluster_size, cluster_limit)
split_fast = SplitFed(copy.deepcopy(server_model), copy.deepcopy(client_model), fast_clients, test_data, 1)
split_slow = SplitFed(copy.deepcopy(server_model), copy.deepcopy(client_model), slow_clients, test_data, 1)
split_slow.one_round()
while rounds > 0:
    rounds -= 1
    rest = split_slow.one_round()['round_exec_time']
    print('slow:', rest)
    while rest > 0:
        ex = split_fast.one_round()['round_exec_time']
        rest -= ex
        print('speed:', ex, 'rest:', rest)
    # cross aggregate
    print('async aggregation running, merging slow clients with fast server')
    print('acc before async:', split_fast.infer())
    # split_fast.crossgregate2(split_slow)
    # print('acc after async:', split_fast.infer())

print('total exec time:', sum(split_fast.round_exec_times))
