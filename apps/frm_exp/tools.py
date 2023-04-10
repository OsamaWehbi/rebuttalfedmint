import json

import matplotlib.pyplot as plt

from src.apis import utils
from src.apis.db_graph_tools import Graphs
from src.apis.fed_sqlite import FedDB

graphs = Graphs(FedDB('res.db'))
plt.rcParams.update({'font.size': 28})


# b2448567fb6168f3a8b16800a59b78b4 	 | 	 logistic_e50_b100_r1000_s3_mnist_cr01_lr0.1
# ced17c0a1a6e785ee6e00e5b9cc09987 	 | 	 logistic_e1_b100_r1000_s3_mnist_cr01_lr0.1
# t3e160f06ed4dba4a5f7680507bf51b96 	 | 	 logistic_e50_b100_r1000_s1_mnist_cr01_lr0.1
# t292a4bcc8f0e27f7c9038232e16f5b51 	 | 	 logistic_e50_b100_r1000_s10_mnist_cr01_lr0.1
# e417e51941f6b7309f17b79263ac3c78 	 | 	 logistic_e1_b100_r1000_s10_mnist_cr01_lr0.1
# a497baab26f56f612868932ceff1ec38 	 | 	 logistic_e1_b100_r1000_s1_mnist_cr01_lr0.1
# t250ad5f5c32f2c2196c7a4017884fdc0 	 | 	 cnn_e50_b100_r1000_s10_mnist_cr01_lr0.01
# t279026eedd7ea46de8ca0d3708d0109e 	 | 	 cnn_e50_b100_r1000_s3_mnist_cr01_lr0.01
# t446210a8694062aa4df49e859de1c682 	 | 	 cnn_e1_b100_r1000_s1_mnist_cr01_lr0.01
# t6cc98699d45986f7752c06cf8f7c6a42 	 | 	 cnn_e50_b100_r1000_s1_mnist_cr01_lr0.01
# bcb21a4827975c3b7e8eaabfa9c05630 	 | 	 cnn_e1_b100_r1000_s3_mnist_cr01_lr0.01
# t3fd7364d5df3cbb36c6fe9e7d8af38c4 	 | 	 cnn_e1_b100_r1000_s10_mnist_cr01_lr0.01
# t7e484a2e21e6c2b950e9313324f0a6e5 	 | 	 logistic_mnist_shard_e50_b100_r1000_s5_mnist_cr01_lr0.1
# ec2c99cd740e08ad623ffcb8626c4413 	 | 	 logistic_mnist_shard_e1_b100_r1000_s2_mnist_cr01_lr0.1
# t91a9f0e66c594f7cd59a0c74de1250fd 	 | 	 logistic_mnist_shard_e1_b100_r1000_s5_mnist_cr01_lr0.1
# t4ee08fbad2a10dc3e5ea08e8d015880f 	 | 	 logistic_mnist_shard_e50_b100_r1000_s2_mnist_cr01_lr0.1
# t21ddca43c6d70a4ddf639778d3ffca12 	 | 	 cnn_mnist_shard_e1_b100_r1000_s2_mnist_cr01_lr0.01
# t2bee5c947572fad2abb66bfc04244020 	 | 	 cnn_mnist_shard_e50_b100_r1000_s2_mnist_cr01_lr0.01
# f2851ba810e2565b5b1b68eb7821f145 	 | 	 cnn_mnist_shard_e1_b100_r1000_s5_mnist_cr01_lr0.01
# t3c88f3db8e531d895e41d1d2c7bdfbe6 	 | 	 cnn_mnist_shard_e50_b100_r1000_s5_mnist_cr01_lr0.01
# e84adbe186a30aeb316f939077ce9fd9 	 | 	 logistic_mnist_dirichlet_e50_b100_r1000_s10_mnist_cr01_lr0.1
# t1f353037ce191bd79453c2df93034283 	 | 	 logistic_mnist_dirichlet_e50_b100_r1000_s0.5_mnist_cr01_lr0.1
# d1f5ed4e5813eb58ac2fe879350364ae 	 | 	 logistic_mnist_dirichlet_e1_b100_r1000_s10.0_mnist_cr01_lr0.1
# t9b8e9f99db6fcd32289142c35da24eef 	 | 	 logistic_mnist_dirichlet_e50_b100_r1000_s10.0_mnist_cr01_lr0.1
# t82568070f6b04360dee21fb3170117c8 	 | 	 logistic_mnist_dirichlet_e1_b100_r1000_s0.5_mnist_cr01_lr0.1
# t6552d9824646601b3a13d8a6ac5620bb 	 | 	 cnn_mnist_dirichlet_e1_b100_r1000_s0.5_mnist_cr01_lr0.01
# t85dcca81f0cd266c603b948334903760 	 | 	 cnn_mnist_dirichlet_e50_b100_r1000_s10.0_mnist_cr01_lr0.01
# t2f8f8ba9557db627e3ac50492c95ccb8 	 | 	 cnn_mnist_dirichlet_e50_b100_r1000_s0.5_mnist_cr01_lr0.01
# t7992d9c5172f9a824581b0d3ed7a0022 	 | 	 cnn_mnist_dirichlet_e1_b100_r1000_s10.0_mnist_cr01_lr0.01
# f351d35ec53dbcef9cc5317466927a5d 	 | 	 logistic_mnist_unique_e1_b100_r1000_s0.5_mnist_cr01_lr0.1
# t140fef62447563213f4ce9f9d713ab6d 	 | 	 logistic_mnist_unique_e50_b100_r1000_s0.5_mnist_cr01_lr0.1
# ca3bb56c51abeff65dbc582a64ac0e61 	 | 	 logistic_mnist_unique_e50_b100_r1000_s0.5_mnist_cr05_lr0.1
# t8b87c810b90f5fa3b822d2bd45be7afa 	 | 	 logistic_mnist_unique_e1_b100_r1000_s0.5_mnist_cr05_lr0.1
# c59e1196684a535785261b54304d9d03 	 | 	 cnn_mnist_unique_e1_b100_r1000_s0.5_mnist_cr05_lr0.01
# ddd016756db445fae3f0cf86025e28be 	 | 	 cnn_mnist_unique_e50_b100_r1000_s0.5_mnist_cr05_lr0.01
# ee123b8f3c6bd831af1fc7c79cfae6ca 	 | 	 cnn_cifar10_dirichlet_e1_b25_r1000_s10.0_cifar10_cr01_lr0.01
# t34ee08a9bad0172a9122c9a1f7acc634 	 | 	 cnn_cifar10_dirichlet_e1_b25_r1000_s0.5_cifar10_cr01_lr0.01
# t546c682610caeb8be14058612061cd8f 	 | 	 cnn_cifar10_dirichlet_e50_b25_r1000_s0.5_cifar10_cr01_lr0.01
# t704ebecaeff0cb2ec092fae0fbdabfa8 	 | 	 cnn_cifar10_dirichlet_e50_b25_r1000_s10.0_cifar10_cr01_lr0.01
# t9cb0decb9ec5633c8a390e0c7ddb9c78 	 | 	 cnn_femnist_dirichlet_e50_b25_r1000_s10.0_femnist_cr01_lr0.001
# d74397d57999ca4b4116064a723560e5 	 | 	 cnn_femnist_dirichlet_e50_b25_r500_s10.0_femnist_cr01_lr0.001
# t77c619f357217f8a615730d48d7a48a3 	 | 	 cnn_femnist_dirichlet_e50_b25_r500_s0.5_femnist_cr01_lr0.001
# t4af296ce080c8262ce52d64909546c00 	 | 	 cnn_femnist_dirichlet_e1_b25_r500_s0.5_femnist_cr01_lr0.001
# t5784e8c713f7a124257f769b7d1a525d 	 | 	 cnn_femnist_dirichlet_e1_b25_r500_s10.0_femnist_cr01_lr0.001


def all_true(arr, item):
    for a in arr:
        if str(a).startswith('!') and a in item:
            return False
        if a not in item:
            return False
    return True


def sett(res, key, val):
    if key in res:
        res[key] = val


def dell(res, key):
    if key in res:
        del res[key]


def preprocess(res):
    dell(res, 'e84adbe186a30aeb316f939077ce9fd9')
    # mnist dir
    sett(res, 'd1f5ed4e5813eb58ac2fe879350364ae', 'α=10, E=1')
    sett(res, 't9b8e9f99db6fcd32289142c35da24eef', 'α=10, E=50')
    sett(res, 't1f353037ce191bd79453c2df93034283', 'α=0.5, E=50')
    sett(res, 't82568070f6b04360dee21fb3170117c8', 'α=0.5, E=1')
    # mnist shards
    sett(res, 't7e484a2e21e6c2b950e9313324f0a6e5', 's=5, E=50')
    sett(res, 'ec2c99cd740e08ad623ffcb8626c4413', 's=2, E=1')
    sett(res, 't91a9f0e66c594f7cd59a0c74de1250fd', 's=5, E=1')
    sett(res, 't4ee08fbad2a10dc3e5ea08e8d015880f', 's=2, E=50')
    # mnist unique
    dell(res, 'f351d35ec53dbcef9cc5317466927a5d')
    dell(res, 't140fef62447563213f4ce9f9d713ab6d')
    sett(res, 'ca3bb56c51abeff65dbc582a64ac0e61', 'E=50')
    sett(res, 't8b87c810b90f5fa3b822d2bd45be7afa', 'E=1')
    # mnist labels
    dell(res, 'b2448567fb6168f3a8b16800a59b78b4')
    dell(res, 'ced17c0a1a6e785ee6e00e5b9cc09987')
    sett(res, 't3e160f06ed4dba4a5f7680507bf51b96', 'L=1, E=50')
    sett(res, 't292a4bcc8f0e27f7c9038232e16f5b51', 'L=10, E=50')
    sett(res, 'e417e51941f6b7309f17b79263ac3c78', 'L=10, E=1')
    sett(res, 'a497baab26f56f612868932ceff1ec38', 'L=1, E=1')
    # cifar10 dirichlet
    # "a7e608edff16eb5c5d5d6cdcb501f4b3": "cifar_dir_new_e50_b25_r500_dis#cifar_dir_10_cifar10_cr1000_lr0.01",
    # "t36bc9e08cdaa9a72061081f1d24a23ef": "cifar_dir_new_e50_b25_r500_dis#cifar_dir_05_cifar10_cr1000_lr0.01",
    # "t25a24103eb2b92c2c0cbce779eef74c7": "cifar_dir_new_e1_b25_r500_dis#cifar_dir_05_cifar10_cr1000_lr0.01",
    # "t93243395f861e633d1005d26ecfb65eb": "cifar_dir_new_e1_b25_r500_dis#cifar_dir_10_cifar10_cr1000_lr0.01"
    sett(res, 't93243395f861e633d1005d26ecfb65eb', 'α=10, E=1')
    sett(res, 't25a24103eb2b92c2c0cbce779eef74c7', 'α=0.5, E=1')
    sett(res, 't36bc9e08cdaa9a72061081f1d24a23ef', 'α=0.5, E=50')
    sett(res, 'a7e608edff16eb5c5d5d6cdcb501f4b3', 'α=10, E=50')
    # femnist dirichlet
    dell(res, 't9cb0decb9ec5633c8a390e0c7ddb9c78')
    # "t6aaee12789f8113a35386824013bd525": "femnist_dir_new_e50_b25_r500_dis#femnist_dir_10_femnist_cr1000_lr0.001",
    # "t935300f5b3d209ae9a721be670f75b03": "femnist_dir_new_e1_b25_r500_dis#femnist_dir_05_femnist_cr1000_lr0.001",
    # "t2eadd78de89fe9d1f9ef5231336ac929": "femnist_dir_new_e1_b25_r500_dis#femnist_dir_10_femnist_cr1000_lr0.001",
    # "t3a0712e5d43d07316347738ca6cf275b": "femnist_dir_new_e50_b25_r500_dis#femnist_dir_05_femnist_cr1000_lr0.001"
    sett(res, 't6aaee12789f8113a35386824013bd525', 'α=0.5, E=50')
    sett(res, 't3a0712e5d43d07316347738ca6cf275b', 'α=10, E=50')
    sett(res, 't935300f5b3d209ae9a721be670f75b03', 'α=10, E=1')
    sett(res, 't2eadd78de89fe9d1f9ef5231336ac929', 'α=0.5, E=1')

    # mnist cnn
    # "t6552d9824646601b3a13d8a6ac5620bb": "cnn_mnist_dirichlet_e1_b100_r1000_s0.5_mnist_cr01_lr0.01",
    # "t85dcca81f0cd266c603b948334903760": "cnn_mnist_dirichlet_e50_b100_r1000_s10.0_mnist_cr01_lr0.01",
    # "t2f8f8ba9557db627e3ac50492c95ccb8": "cnn_mnist_dirichlet_e50_b100_r1000_s0.5_mnist_cr01_lr0.01",
    # "t7992d9c5172f9a824581b0d3ed7a0022": "cnn_mnist_dirichlet_e1_b100_r1000_s10.0_mnist_cr01_lr0.01",
    sett(res, 't6552d9824646601b3a13d8a6ac5620bb', 'α=0.5, E=1')
    sett(res, 't85dcca81f0cd266c603b948334903760', 'α=10, E=50')
    sett(res, 't2f8f8ba9557db627e3ac50492c95ccb8', 'α=0.5, E=50')
    sett(res, 't7992d9c5172f9a824581b0d3ed7a0022', 'α=10, E=1')

    # cifar shards
    # "e02487267acb0edc062b301b66faf03c": "cifar_shard_new_e1_b25_r500_dis#cifar_shards_2_cifar10_cr1000_lr0.01",
    # "t1dc912269e1629be797106bcb9cfae3f": "cifar_shard_new_e50_b25_r500_dis#cifar_shards_2_cifar10_cr1000_lr0.01",
    # "t7bdaf03eaa44f3778dc78d85366214de": "cifar_shard_new_e50_b25_r500_dis#cifar_shards_5_cifar10_cr1000_lr0.01",
    # "ad626fe1b6dab217f33679110a847009": "cifar_shard_new_e1_b25_r500_dis#cifar_shards_5_cifar10_cr1000_lr0.01"
    sett(res, 't7bdaf03eaa44f3778dc78d85366214de', 'S=5, E=50')
    sett(res, 'ad626fe1b6dab217f33679110a847009', 'S=5, E=1')
    sett(res, 't1dc912269e1629be797106bcb9cfae3f', 'S=2, E=50')
    sett(res, 'e02487267acb0edc062b301b66faf03c', 'S=2, E=1')

    # femnist shards
    # "t046ec7159180743b734df826cc304a68": "femnist_shard_new_e1_b25_r500_dis#femnist_shards_5_femnist_cr1000_lr0.001",
    # "t3ae37af2a3d6884ca3538088a0751b22": "femnist_shard_new_e50_b25_r500_dis#femnist_shards_5_femnist_cr1000_lr0.001",
    # "t4ef9f85f1a03450f279397c88e473e53": "femnist_shard_new_e1_b25_r500_dis#femnist_shards_2_femnist_cr1000_lr0.001",
    # "t4418120b06ec88298df8236a0543bffb": "femnist_shard_new_e50_b25_r500_dis#femnist_shards_2_femnist_cr1000_lr0.001"
    sett(res, "t3ae37af2a3d6884ca3538088a0751b22", 'S=5, E=50')
    sett(res, "t046ec7159180743b734df826cc304a68", 'S=5, E=1')
    sett(res, "t4418120b06ec88298df8236a0543bffb", 'S=2, E=50')
    sett(res, "t4ef9f85f1a03450f279397c88e473e53", 'S=2, E=1')
    return res


def filter(*query):
    res = {}
    for key, val in graphs.db().tables().items():
        if all_true(query, val):
            res[key] = val
    return res


def pretty(val):
    print(json.dumps(val, indent=4))


def round_size(size):
    return lambda vals: vals[:size]


def plot_builder(field, tables):
    colors = ['b', 'r', '#117733', '#DDCC77']
    linestyles = ['-.', '-', 'dashdot', 'solid']
    index = 0
    plts = []
    res = preprocess(tables)
    pretty(res)
    for k, v in res.items():
        plts.append({
            'session_id': k,
            'field': field,
            'config': {'color': colors[int(index % len(colors))], 'label': v,
                       'linestyle': linestyles[int(index % len(linestyles))]},
            'transform': round_size(30)
        })
        index += 1
    return plts


def plt_config(plt):
    plt.grid()
    plt.legend(loc='best')
    # plt.xlabel('Round')
    # plt.ylabel('Accuracy')


root = './plts/'


def plot(configs, save_file, show=False):
    save_file = f'{root}{save_file}' if save_file else False
    graphs.plot(configs, save_path=save_file, plt_func=plt_config, show=show)


def get_info(table_name, field):
    return graphs.db().get(table_name, field)
