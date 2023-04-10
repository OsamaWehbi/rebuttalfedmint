import copy

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
rounds = 100

# init data
client_model = models.MnistClient(784, 32, 10)
server_model = models.MnistServer(784, 32, 10)
clients_data = preload('mnist', dist.clustered(client_cluster_size, 300), tag=f'cluster{client_cluster_size}p{300}')
train_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[0]).map(lambdas.as_tensor)
test_data = clients_data.map(lambda k, dc: dc.shuffle(45).split(0.9)[1]).reduce(lambdas.dict2dc).as_tensor()

# split learning
clustered_clients = clusters.generate(train_data, client_model, client_cluster_size, cluster_limit)
split_learning = SplitFed(server_model, client_model, clustered_clients, test_data, 1)
while rounds > 0:
    rounds -= 1
    split_learning.one_round()
    print('acc:', split_learning.acc[-1])
