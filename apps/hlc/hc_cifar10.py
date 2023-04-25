import logging
import statistics
import sys
import random
import torch
import xlsxwriter
from collections import defaultdict

from apps.hlc.apis import SaveClientsModels
from src import apis

sys.path.append('../../')

from src.apis.rw import IODict
from src.apis.extensions import Dict, TorchModel
from src.federated.subscribers.fed_plots import EMDWeightDivergence
from src.federated.subscribers.resumable import Resumable
from src.federated.subscribers.sqlite_logger import SQLiteLogger
from libs.model.cv.resnet import resnet56
from torch import nn
import libs.model.cv.cnn
from src.data.data_container import DataContainer
from src.federated.subscribers.logger import FederatedLogger, TqdmLogger
from src.federated.subscribers.timer import Timer
from src.apis import lambdas,utils
from apps.hlc import apis
from apps.hlc.apis import SaveClientsModels
from src.data import data_loader
from src.federated.components import metrics, client_selectors, aggregators, trainers
from src.federated.federated import Events
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams
from src.federated.components.trainer_manager import SeqTrainerManager
from src.data.data_distributor import LabelDistributor, ShardDistributor
from src.data.data_loader import preload
from src.federated.components.client_scanners import DefaultScanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
logger.info('Generating Data --Started')
# client_data = data_loader.cifar10_10shards_100c_600min_600max()

def global_infer(batched_data, all_clusters):
    test_results = {}
    for cmid in all_clusters:
        test = all_clusters[cmid].infer(batched_data, device='cuda')
        test_results[cmid] = test[0]
    return test_results

def poison(dc: DataContainer, rate):
    total_size = len(dc)
    poison_size = int(total_size * rate)
    for i in range(0, poison_size):
        dc.y[i] = 0 if dc.y[i] != 0 else random.randint(1, 9)


# client_data = preload('cifar10', LabelDistributor(100, 5, 550, 600))
client_data = preload('cifar10', ShardDistributor(200,3))
# data = preload('cifar10')
# data = data.concat(ddata)
# client_data = ShardDistributor(200,3).distribute(data)
server1 = {key: client_data[key] for key in client_data.keys() if 0 <= key <= 49}
server2 = {key: client_data[key] for key in client_data.keys() if 50 <= key <= 99}


def splter(client_data, pois=True):
    poisc = 0
    train_data = Dict()
    test_data = Dict()
    for trainer_id, data in client_data.items():
        data = data.shuffle().as_tensor()
        train, test = data.split(0.7)
        test_data[trainer_id] = test
        # train_data[trainer_id] = train
        if pois and poisc < 15:
            poison(train, 0.5)
        train_data[trainer_id] = train
        poisc += 1
    return train_data, test_data

trdatalist =[]
tsdatalist =[]
td, ts = splter(server1)
trdatalist.append(td)
tsdatalist.append(ts)
td, ts = splter(server2)
trdatalist.append(td)
tsdatalist.append(ts)

logger.info('Generating Data --Ended')


def create_model(name):
    if name == 'resnet':
        return resnet56(10, 3, 32)
    else:
        global trainers_train
        global test_data
        global client_data
        # cifar10 data reduced to 1 dimension from 32,32,3. cnn32 model requires the image shape to be 3,32,32
        for i in range(2):
            trdatalist[i] = trdatalist[i].map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
            tsdatalist[i] = tsdatalist[i].map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
            client_data = client_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
            # trainers_train = trainers_train.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
            # test_data = test_data.map(lambdas.reshape((-1, 32, 32, 3))).map(lambdas.transpose((0, 3, 1, 2)))
        return libs.model.cv.cnn.Cifar10Model()


initialize_model = create_model('cnn')

# transfer data to normal dic and batch it
for i in range(2):
    tsdatalist[i] = tsdatalist[i].reduce(lambdas.reducers.dict2dc).as_tensor().batch(50)

#### warm up begin
# client_data = preload('cifar10', ShardDistributor(200,3))
# client_data = preload('mnist', LabelDistributor(100, 2, 600, 600))
# test_data = preload('mnist10k').as_tensor()
warmup_rounds = 2
server_num = 2
training_rounds = 500
total_selection = 10
servers_list = []
cached_models_list = []
for i in range(server_num):
    cached_models_list.append(SaveClientsModels())
    # trainers configuration
    trainer_params = TrainerParams(
        trainer_class=trainers.TorchTrainer,
        batch_size=50, epochs=1, optimizer='sgd',
        criterion='cel', lr=0.1)

    # fl parameters
    federated = FederatedLearning(
        trainer_manager=SeqTrainerManager(),
        trainer_config=trainer_params,
        aggregator=aggregators.AVGAggregator(),
        metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
        client_scanner=DefaultScanner(client_data),
        client_selector=client_selectors.All(),
        trainers_data_dict=trdatalist[i],
        initial_model=lambda: initialize_model,
        num_rounds=warmup_rounds,
        desired_accuracy=0.99
    )

    # (subscribers)
    federated.add_subscriber(TqdmLogger())
    federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
    federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
    cached_models_list[i].attach(federated)

    logger.info("------------------------")
    logger.info("start federated learning")
    logger.info("------------------------")
    servers_list.append(federated)
for i in range(2):
    servers_list[i].start()
### warm uppppp ending

server_meta_mod = ["server1","server2"]


# I have the weights of each clients do clusternig

servers_cluster_list = []

for i in range(2):
    clients_weights = cached_models_list[i].clients_weights
    clusters_clients = utils.hc_clustering(clients_weights, 5)
    clusters = defaultdict(list)
    for cid, cluster_id in clusters_clients.items():
        clusters[cluster_id].append(cid)
    clusters_federations = {}
    selection_per_round = apis.get_nearly_equal_numbers(total_selection, len(clusters))
    print(selection_per_round)
    #end clustering



    #create federated servers for each cluster
    for c in clusters:
        cluster_clients_data = Dict(trdatalist[i]).select(clusters[c])
        trainer_params = TrainerParams(
            trainer_class=trainers.TorchTrainer,
            batch_size=50, epochs=5, optimizer='sgd',
            criterion='cel', lr=0.1)
        federated = FederatedLearning(
            trainer_manager=SeqTrainerManager(),
            trainer_config=trainer_params,
            aggregator=aggregators.AVGAggregator(),
            metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
            client_scanner=DefaultScanner(cluster_clients_data),
            client_selector=client_selectors.Random(selection_per_round[c]),
            trainers_data_dict=trdatalist[i],
            initial_model=lambda: initialize_model,
            num_rounds=training_rounds,
            desired_accuracy=0.99
        )
        federated.add_subscriber(TqdmLogger())
        federated.add_subscriber(FederatedLogger([Events.ET_TRAINER_SELECTED, Events.ET_ROUND_FINISHED]))
        federated.add_subscriber(Timer([Timer.FEDERATED, Timer.ROUND]))
        clusters_federations[c] = federated
        clusters_federations[c].init()
    servers_cluster_list.append(clusters_federations)

# to write the results
workbook = xlsxwriter.Workbook("results_acc.xlsx")

for i in range(2):
    # create worksheet in the work book for each server
    worksheet = workbook.add_worksheet(server_meta_mod[i])
    # repeat untill reach the disired number of rounds
    for r in range(training_rounds):
        # in each server we have multiple clusters each is an independent server

        # this is for saving the model of each cluster at each round
        cluster_models = {}
        for fe_id, fed in servers_cluster_list[i].items():
            # we want to run it just for one round we did the init() in the previous step
            # fed.start()
            fed.one_round()
            # now we want to check accuracy at each round
            cluster_models[fe_id] = TorchModel(fed.context.model)
        # print("###########################################")
        res =global_infer(tsdatalist[i], cluster_models)
        # print(res)
        worksheet.write('A' + str(r+1), r+1)
        worksheet.write('B' + str(r+1), float(sum(res.values())/len(res.values())))
        # print("###########################################")
# for cf in clusters_federations:
#     cluster_models[cf] = TorchModel(clusters_federations[cf].context.model)

# close the book
workbook.close()

def global_infer_max(data, all_clusters):
    all_res = []
    for i in range(len(data)):
        test_results = {}
        for cmid in all_clusters:
            test = all_clusters[cmid].infer(data.select([i]).batch(0), device='cpu')
            test_results[cmid] = test[0]
        all_res.append(max(test_results.values()))
    return statistics.mean(all_res)


# print(global_infer_max(test_data.select(range(20)), cluster_models))
#end the process


