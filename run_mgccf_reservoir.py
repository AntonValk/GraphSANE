import os, time, datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from scipy.special import softmax

import config2
from log2 import Logger
from sampler import pad_adj, WarpSampler
from data_utils.data_generator import Data
from data_utils.utils import *
from data_utils.reservoir_util import *
from data_utils.preprocessing import generate_index_dict, convert_dict_to_list
from metrics import *
from scipy.stats import wasserstein_distance
from models.mgccf_reservoir import GNN as MGCCF
from sklearn.neighbors import kneighbors_graph
import random


def write_prediction_to_logger(logger, precision, recall, MAP, ndcg, epoch, name):
    if logger is not None:
        logger.write('Epoch: {} ({}) \n'.format(epoch, name))
        logger.write(str(precision) + '\n')
        logger.write(str(recall) + '\n')
        logger.write(str(MAP) + '\n')
        logger.write(str(ndcg) + '\n')
    else:
        print('Epoch: {} ({}) \n'.format(epoch, name))
        print(precision)
        print(recall)
        print(MAP)
        print(ndcg)


def convert_list_to_adj(adj, n_item):
    '''
    convert lists of list of users' item neighbors to items' categorical bag of words
    '''
    adj_mat = np.zeros((len(adj), n_item))
    for user, idx in enumerate(adj):
        adj_mat[user][idx] = 1
    return adj_mat


def convert_adj_to_cat(old_list, n_item, n_cluster, y_kmeans):
    '''
    convert lists of list of users' item neighbors to items' categorical bag of words
    :param old_list: list of lists of users' item neighbors as indices
    :param n_item: number of items
    :param n_cluster: number of clusters
    :param y_kmeans: labels assigned by Kmeans clustering
    :return: bag-of-word matrix in the shape of # of users * number of clusters
    '''
    old_adj = convert_list_to_adj(old_list, n_item)
    new_adj = np.zeros((old_adj.shape[0], n_cluster))
    for i in range(old_adj.shape[0]):
        for j in range(len(y_kmeans)):
            new_adj[i][y_kmeans[j]] += old_adj[i][j]

    return new_adj


def convert_adj_to_cat_mean(old_list, emb, n_cluster, y_kmeans):
    '''
    :param old_list: list of lists of users' item neighbors as indices
    :param emb: item_embedding
    :param n_cluster: number of clusters
    :param y_kmeans: labels assigned by Kmeans clustering
    :return: matrix in shape of # of user * # of cluster * embedding size
    '''
    new_adj = np.zeros((len(old_list), n_cluster, emb.shape[1]))
    for i in range(new_adj.shape[0]):
        neighs = old_list[i]
        for j in range(n_cluster):
            if len(np.where(y_kmeans[neighs] == j)[0]) > 0:
                new_adj[i][j] = np.mean(emb[np.where(y_kmeans[neighs] == j)[0]], axis=0)
    return new_adj




def generate_local_neighbor(u_adj_list, i_emb, distill_mode, num_neigh):
    '''
    :param u_adj_list: dictionary of neighbor indices (N_1 * emb_size)
    :param i_emb: neighbor pool embedding matrix (N_2 * emb_size)
    :param distill_mode: metric to calculate distances
    :param num_neigh: number of neighbors to generate
    :return u_ls_matrix: Matrix of distances to users (N1 * num_neigh)
    :return u_ls_index: Matrix of indices to neighbors to users (N1 * num_neigh)
    '''

    u_ls_matrix = np.zeros([u_emb.shape[0], num_neigh])
    u_ls_index = np.zeros([u_emb.shape[0], num_neigh])
    for u, i_list in u_adj_list.items():
        if u >= u_emb.shape[0]:
            break
        if len(i_list) > 0:
            i_list = [x for x in i_list if x < i_emb.shape[0]]

            # sample version
            if len(i_list) > 10:
                i_list = random.sample(i_list, 10)
            else:
                i_list = random.choices(i_list, k=10)

            u_1hop_emb = np.take(i_emb, i_list, axis=0)
            if distill_mode == 'euc':
                u_i_distance = np.square(np.linalg.norm(u_1hop_emb - u_emb[u], axis=1))
            elif distill_mode == 'inner_product':
                u_i_distance = np.sum(u_1hop_emb * u_emb[u], axis=1)
            elif distill_mode == 'poly':
                u_i_distance = np.square(np.sum(u_1hop_emb * u_emb[u], axis=1))
            elif distill_mode == 'rbf':
                u_i_distance = np.square(np.linalg.norm(u_1hop_emb - u_emb[u], axis=1))
                u_i_distance = np.exp(-0.5 * u_i_distance)
            else:
                raise NotImplementedError
            u_ls = softmax(u_i_distance)  # 1 * num_neigh
            # u_ls_matrix[u][i_list] = u_ls
            u_ls_matrix[u] = u_ls
            u_ls_index[u] = i_list
        else:
            i_list = random.sample(range(i_emb.shape[0]), num_neigh)
            u_ls_index[u] = i_list
            u_i_distance = np.ones(num_neigh) / num_neigh
            u_ls_matrix[u] = u_i_distance
    return u_ls_matrix, u_ls_index

def get_local_structure(u_adj_list, v_adj_list, u_emb, i_emb, distill_mode):
    '''
    :param u_adj_list: dictionary of item indices as neighbors to users
    :param v_adj_list: dictionary of user indices as neighbors to items
    :param u_emb: user_embedding matrix (N_1 * emb_size)
    :param i_emb: item_embedding matrix (N_2 * emb_size)
    :param distill_mode: metric to calculte distances
    :param num_neigh: number of neighbor to choose
    :return u_ls_matrix: Matrix of distances to users (N1 * num_neigh)
    :return u_ls_index: Matrix of indices to neighbors to users (N1 * num_neigh)
    :return i_ls_matrix: Matrix of distances to users (N2 * num_neigh)
    :return i_ls_index: Matrix of indices to neighbors to users (N2 * num_neigh)
    '''
    assert distill_mode != ''
    # static graph case for now
    # u_ls_matrix = np.zeros([u_emb.shape[0], i_emb.shape[0]])
    # i_ls_matrix = np.zeros([i_emb.shape[0], u_emb.shape[0]])

    u_ls_matrix = np.zeros([u_emb.shape[0], 10])
    i_ls_matrix = np.zeros([i_emb.shape[0], 10])
    u_ls_index = np.zeros([u_emb.shape[0], 10])
    i_ls_index = np.zeros([i_emb.shape[0], 10])

    for u, i_list in u_adj_list.items():
        if u >= u_emb.shape[0]:
            break
        if len(i_list) > 0:
            i_list = [x for x in i_list if x < i_emb.shape[0]]

            # sample version
            if len(i_list) > 10:
                i_list = random.sample(i_list, 10)
            else:
                i_list = random.choices(i_list, k=10)

            u_1hop_emb = np.take(i_emb, i_list, axis=0)
            if distill_mode == 'euc':
                u_i_distance = np.square(np.linalg.norm(u_1hop_emb - u_emb[u], axis=1))
            elif distill_mode == 'inner_product':
                u_i_distance = np.sum(u_1hop_emb * u_emb[u], axis=1)
            elif distill_mode == 'poly':
                u_i_distance = np.square(np.sum(u_1hop_emb * u_emb[u], axis=1))
            elif distill_mode == 'rbf':
                u_i_distance = np.square(np.linalg.norm(u_1hop_emb - u_emb[u], axis=1))
                u_i_distance = np.exp(-0.5 * u_i_distance)
            else:
                raise NotImplementedError
            u_ls = softmax(u_i_distance)
            # u_ls_matrix[u][i_list] = u_ls
            u_ls_matrix[u] = u_ls
            u_ls_index[u] = i_list
        else:
            i_list = random.sample(range(i_emb.shape[0]), 10)
            u_ls_index[u] = i_list
            u_i_distance = np.ones(10) / 10
            u_ls_matrix[u] = u_i_distance

    for i, u_list in v_adj_list.items():
        if i >= i_emb.shape[0]:
            break
        if len(u_list) > 0:
            u_list = [x for x in u_list if x < u_emb.shape[0]]

            if(len(u_list)) == 0:
                u_list = random.sample(range(u_emb.shape[0]), 10)
                i_ls_index[i] = u_list
                i_u_distance = np.ones(10) / 10
                i_ls_matrix[i] = i_u_distance

                return u_ls_matrix, i_ls_matrix, u_ls_index, i_ls_index

            # sample version
            if len(u_list) > 10:
                u_list = random.sample(u_list, 10)
            else:
                u_list = random.choices(u_list, k=10)

            i_1hop_emb = np.take(u_emb, u_list, axis=0)
            if distill_mode == 'euc':
                i_u_distance = np.square(np.linalg.norm(i_1hop_emb - i_emb[i], axis=1))
            elif distill_mode == 'inner_product':
                i_u_distance = np.sum(i_1hop_emb * i_emb[i], axis=1)
            elif distill_mode == 'poly':
                i_u_distance = np.square(np.sum(i_1hop_emb * i_emb[i], axis=1))
            elif distill_mode == 'rbf':
                i_u_distance = np.square(np.linalg.norm(i_1hop_emb - i_emb[i], axis=1))
                i_u_distance = np.exp(-0.5 * i_u_distance)
            else:
                raise NotImplementedError
            i_ls = softmax(i_u_distance)
            # i_ls_matrix[i][u_list] = i_ls
            i_ls_matrix[i] = i_ls
            i_ls_index[i] = u_list
        else:
            u_list = random.sample(range(u_emb.shape[0]), 10)
            i_ls_index[i] = u_list
            i_u_distance = np.ones(10) / 10
            i_ls_matrix[i] = i_u_distance

            # i_u_distance = np.ones(u_emb.shape[0]) / u_emb.shape[0]
            # i_ls_matrix[i] = i_u_distance

    return u_ls_matrix, i_ls_matrix, u_ls_index, i_ls_index


def load_self_neighbours(file_path, data_group, n_rows, n_neighbours, graph_adj_matrix, n_negative=None):
    '''
    :param file_path: file path to save u-u or i-i graph generated by sklearn kneighbors
    :param data_group: indicate specific usage or user/item group of the folder for saving purpose
    :param n_rows: number of users or items
    :param n_neighbours: number of neighbors
    :param graph_adj_matrix: user_item or item_user adjacency matrix
    :param n_negative: number of negative samples, default is none
    :return self_neighs: self positive neighbors indices of users/items N * num_postive_neigh
    :return negative_samples: self negative neighbors indices of users/items N * num_negative_neigh
    '''
    graph_file_path = file_path[:-4] + f'_{n_neighbours}_' + data_group + file_path[-4:]
    if os.path.isfile(graph_file_path):
        self_neigh_graph = load_pickle(graph_file_path, '')
    else:
        self_neigh_graph = kneighbors_graph(graph_adj_matrix, n_neighbours, mode='distance', metric='cosine',
                                            include_self=False)
        save_pickle(self_neigh_graph, graph_file_path[:-4], '')
    self_neighs = self_neigh_graph.tocoo().col
    self_neighs = np.array(np.array_split(self_neighs, n_rows))
    if n_negative is not None:
        neigh_pairs = []
        for i in range(len(self_neighs)):
            neigh_pairs += [[i, j] for j in self_neighs[i]]
        neigh_pairs = np.array(neigh_pairs)
        user_to_positive_set = {u: set(row) for u, row in enumerate(self_neighs)}

        # sample negative samples
        negative_samples = np.random.randint(
            0,
            len(self_neighs),
            size=(len(self_neighs), n_negative))

        for user_positive, negatives, i in zip(neigh_pairs,
                                               negative_samples,
                                               range(len(negative_samples))):
            user = user_positive[0]
            for j, neg in enumerate(negatives):
                while neg in user_to_positive_set[user]:
                    negative_samples[i, j] = neg = np.random.randint(0, len(self_neighs))
    else:
        negative_samples = None

    return self_neighs, negative_samples

def load_bi_neighbours(adj_mat, num_neigh, self_n_negative):
    '''
    load positive and negative neighbors from u-i graph.
    :param adj_mat: user-item adjacency matrix
    :param num_neigh: number of positive neighbors
    :param self_n_negative: number of negative neighbors
    :return:Four lists of lists
    '''
    u_adj_dict, v_adj_dict = sparse_adj_matrix_to_dicts(adj_mat)
    u_pos_neighs, v_pos_neighs = pad_adj(u_adj_dict, num_neigh, adj_mat.shape[0]), \
                                 pad_adj(v_adj_dict, num_neigh, adj_mat.shape[1])
    # adj_mat = np.array(adj_mat.todense()).astype(np.float64)
    u_neg_neighs = []
    for i in range(adj_mat.shape[0]):
        neg_idx = list(set(np.arange(len(v_adj_dict))) - set(u_adj_dict[i]))
        u_neg_neighs.append(random.sample(neg_idx, self_n_negative))
    i_neg_neighs = []
    for i in range(adj_mat.shape[1]):
        neg_idx = list(set(np.arange(len(u_adj_dict))) - set(v_adj_dict[i]))
        i_neg_neighs.append(random.sample(neg_idx, self_n_negative))

    return u_pos_neighs, v_pos_neighs, u_neg_neighs, i_neg_neighs



def get_model_prediction(parser, checkpoint, train_matrix, n_user, n_item, graph_path=None, return_type='',
                         test_info=None):
    if test_info is not None:
        n_test_user, n_test_item = test_info
    else:
        n_test_user, n_test_item = n_user, n_item

    # for MGCCF
    user_self_neighs, _ = load_self_neighbours(graph_path[0], 'train', n_test_user, parser.num_neigh, train_matrix)
    item_self_neighs, _ = load_self_neighbours(graph_path[1], 'train', n_test_item, parser.num_neigh,
                                            train_matrix.transpose())

    # u_adj_dict, v_adj_dict = adj_matrix_to_dicts(train_matrix)
    u_adj_dict, v_adj_dict = sparse_adj_matrix_to_dicts(train_matrix)
    u_adj_list, v_adj_list = pad_adj(u_adj_dict, parser.max_degree, n_test_item), pad_adj(v_adj_dict, parser.max_degree,
                                                                                          n_test_user)

    # initialize model
    model = MGCCF([eval(parser.embedded_dimension)[0], n_user, n_item],
                  eval(parser.embedded_dimension)[1:],
                  parser.max_degree,
                  eval(parser.gcn_sample),
                  ['adam', parser.learning_rate, parser.epsilon],
                  'my_mean',
                  parser.activation,
                  parser.neighbor_dropout,
                  parser.l2,
                  parser.num_self_neigh,
                  parser.num_neg,
                  wg_dimension=parser.wg_dimension,
                  wg_act=parser.wg_act)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with model.graph.as_default():
        saver = tf.train.Saver(max_to_keep=100)
        # do not restore embeddings and optimizer vars for fine-tuning
        # print("#####################################################################################")
        # print(saver._var_list)
        new_var_list = [x for x in saver._var_list if ("embedding" not in x.name and "Adam" not in x.name and
                                                       "center" not in x.name and "input" not in x.name and
                                                       "user_weight" not in x.name and "transformation" not in x.name)]
        saver_2 = tf.train.Saver(var_list=new_var_list)

    with tf.Session(graph=model.graph, config=sess_config) as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # load checkpoints
        saver_2.restore(sess, checkpoint)
        old_u_emb_val = tf.train.load_variable(checkpoint, 'model/user_embedding')
        old_i_emb_val = tf.train.load_variable(checkpoint, 'model/item_embedding')
        u_emb_val = sess.run(model.user_embeddings)
        i_emb_val = sess.run(model.item_embeddings)

        assert old_u_emb_val.shape[0] == u_emb_val.shape[0]
        assert old_i_emb_val.shape[0] == i_emb_val.shape[0]

        model.user_embeddings.load(u_emb_val, sess)
        model.item_embeddings.load(i_emb_val, sess)

        feed_dict = {model.u_adj_info_ph: u_adj_list,
                     model.v_adj_info_ph: v_adj_list,
                     model.u_u_graph_ph: user_self_neighs,
                     model.v_v_graph_ph: item_self_neighs}

        if n_test_user > n_user:
            new_users_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('user', old_u_emb_val,
                                                                                       train_matrix)
        else:
            new_users_init = None
        if n_test_item > n_item:
            new_items_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('item', old_i_emb_val,
                                                                                       train_matrix)
        else:
            new_items_init = None

        items = np.arange(0, n_test_item, 1, dtype=int)
        users = np.arange(0, n_test_user, 1, dtype=int)

        rating_preds, user_rep, item_rep = model.predict(users, items, n_test_user, new_users_init, new_items_init)
        rating_preds, user_rep, item_rep = sess.run([rating_preds, user_rep, item_rep], feed_dict)

    if return_type == 'reps':
        return user_rep, item_rep
    else:
        return rating_preds


def train_model(parser, train_info, val_info, test_info, old_train_set, old_train_matrix,
                n_epoch, n_old_user=0, n_old_item=0, node_deg_delta=None, logger=None, load_checkpoint='',
                save_checkpoint='', graph_path=None, process_saver=False, weights_saver='', medium_saver=''):
    train_set, n_user, n_item, train_matrix = train_info
    val_set, n_user_val, n_item_val, val_matrix = val_info
    test_set, n_user_test, n_item_test, test_matrix = test_info

    # for MGCCF
    user_self_neighs, _ = load_self_neighbours(graph_path[0], 'train', n_user, parser.num_neigh, train_matrix)
    item_self_neighs, _ = load_self_neighbours(graph_path[1], 'train', n_item, parser.num_neigh, train_matrix.transpose())
    user_self_neighs_val, _ = load_self_neighbours(graph_path[0], 'val', n_user_val, parser.num_neigh, val_matrix)
    item_self_neighs_val, _ = load_self_neighbours(graph_path[1], 'val', n_item_val, parser.num_neigh,
                                                val_matrix.transpose())
    user_self_neighs_test, _ = load_self_neighbours(graph_path[0], 'test', n_user_test, parser.num_neigh, test_matrix)
    item_self_neighs_test, _ = load_self_neighbours(graph_path[1], 'test', n_item_test, parser.num_neigh,
                                                 test_matrix.transpose())

    # prepare train data
    u_adj_dict, v_adj_dict = sparse_adj_matrix_to_dicts(train_matrix)

    u_adj_list, v_adj_list = pad_adj(u_adj_dict, parser.max_degree, n_item), pad_adj(v_adj_dict, parser.max_degree,
                                                                                     n_user)

    u_adj_dict_val, v_adj_dict_val = sparse_adj_matrix_to_dicts(val_matrix)
    u_adj_list_val, v_adj_list_val = pad_adj(u_adj_dict_val, parser.max_degree, n_item_val), pad_adj(v_adj_dict_val,
                                                                                                     parser.max_degree,
                                                                                                     n_user_val)
    u_adj_dict_test, v_adj_dict_test = sparse_adj_matrix_to_dicts(test_matrix)
    u_adj_list_test, v_adj_list_test = pad_adj(u_adj_dict_test, parser.max_degree, n_item_test), pad_adj(
        v_adj_dict_test, parser.max_degree, n_user_test)

    # initialize model
    model = MGCCF([eval(parser.embedded_dimension)[0], n_user, n_item],
                  eval(parser.embedded_dimension)[1:],
                  parser.max_degree,
                  eval(parser.gcn_sample),
                  ['adam', parser.learning_rate, parser.epsilon],
                  'my_mean',
                  parser.activation,
                  parser.neighbor_dropout,
                  parser.l2,
                  parser.num_self_neigh,
                  parser.num_neg,
                  parser.con_positive,
                  parser.con_negative,
                  parser.trans_positive,
                  inc_reg=[parser.lambda_mse, parser.lambda_distillation, parser.lambda_global_distill],
                  old_num_user=n_old_user,
                  old_num_item=n_old_item,
                  distill_mode = parser.distill_mode,
                  k_centroids=eval(parser.k_centroids),
                  tau=parser.tau,
                  num_neigh=parser.num_neigh,
                  local_distill_mode=parser.local_mode,
                  contrastive_mode=parser.contrastive_mode,
                  layer_wise=parser.layer_wise,
                  adaptive_mode = parser.adaptive_mode,
                  lambda_contrastive = parser.lambda_contrastive,
                  lambda_soft = parser.lambda_soft,
                  wg_dimension=parser.wg_dimension,
                  wg_act = parser.wg_act,
                  nu = parser.nu)

    num_pairs = 0
    for i in range(len(train_set)):
        num_pairs += len(train_set[i])
    num_iter = int(num_pairs / parser.batch_pairs) + 1
    iter_time = []

    sampler = WarpSampler(train_set,
                          n_item,
                          batch_size=parser.batch_pairs,
                          n_negative=parser.num_neg,
                          n_workers=2,
                          check_negative=True)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with model.graph.as_default():
        saver = tf.train.Saver(max_to_keep=100)
        # do not restore embeddings and optimizer vars for fine-tuning
        new_var_list = [x for x in saver._var_list if ("embedding" not in x.name and "Adam" not in x.name and
                                                       "center" not in x.name and "input" not in x.name and
                                                       "user_weight" not in x.name and "transformation" not in x.name)]
        saver_2 = tf.train.Saver(var_list=new_var_list)

    with tf.Session(graph=model.graph, config=sess_config) as sess:

        # initialize variables
        sess.run(tf.global_variables_initializer())
        # load checkpoints
        u_emb_val = sess.run(model.user_embeddings)
        i_emb_val = sess.run(model.item_embeddings)

        # load existing model
        if load_checkpoint != "":
            saver_2.restore(sess, load_checkpoint)
            # only load existing nodes' embedding
            old_u_emb_val = tf.train.load_variable(load_checkpoint, 'model/user_embedding')
            old_i_emb_val = tf.train.load_variable(load_checkpoint, 'model/item_embedding')

            # If it is incremental blocks, u_emb_val would be set as old_embedding + average of similar
            # old user embedding for new users. So for items.
            u_emb_val[:old_u_emb_val.shape[0], ] = old_u_emb_val
            i_emb_val[:old_i_emb_val.shape[0], ] = old_i_emb_val

            # # initialize new node as mean of 2-hop neighbours
            if parser.new_node_init == '2hop_mean':
                if n_user_train > n_old_user:
                    new_users_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('user', old_u_emb_val,
                                                                                               train_matrix)
                    u_emb_val[n_old_user:, ] = new_users_init
                if n_item_train > n_old_item:
                    new_items_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('item', old_i_emb_val,
                                                                                               train_matrix)
                    i_emb_val[n_old_item:, ] = new_items_init

            # teacher's 1-hop local structure
            if parser.lambda_distillation > 0 and parser.local_mode == 'LSP_s':
                u_ls_matrix, i_ls_matrix, u_ls_index, i_ls_index = get_local_structure(u_adj_dict, v_adj_dict,
                                                                                       old_u_emb_val, old_i_emb_val,
                                                                                       parser.distill_mode)
                '''	
                # ============= unsampleed version ===========	
                #     train_sp_idx, _, __ = convert_sparse_matrix_to_sparse_tensor_input(train_matrix[:n_old_user, :n_old_item]) # shape: [U*I(ones) x 2]	
                #     train_sp_expand_idx = np.repeat(train_sp_idx, repeats=eval(parser.embedded_dimension)[2], axis=0) # shape: [128*U*I(ones) x 2]	
                #     emb_expansion = np.arange(eval(parser.embedded_dimension)[2])	
                #     broadcasted_emb_expansion = np.broadcast_to(emb_expansion, [train_sp_idx.shape[0], emb_expansion.shape[0]]).reshape(-1, 1) # shape: [128*U*I x 1]	
                #     train_sp_expand_idx = np.concatenate((train_sp_expand_idx, broadcasted_emb_expansion), axis=1)  # shape: [128*U*I(ones) x 3]	
                '''
            if parser.lambda_distillation > 0 and parser.local_mode == 'local_anchor':

                time_info.append(('start calc local_anchor coef', time.time()))
                # simple one-hop
                sub_matrix = train_matrix[:old_u_emb_val.shape[0], :old_i_emb_val.shape[0]]
                sub_matrix_dense = np.array(sub_matrix.todense()).astype(np.float64)

                u_1hop_means = np.zeros(old_u_emb_val.shape)
                for u in range(len(u_1hop_means)):
                    u_idx = sub_matrix_dense[u].nonzero()
                    u_1hop_means[u] = np.mean(np.take(old_i_emb_val, u_idx[0], axis=0), axis=0)
                u_i_prod = np.sum(old_u_emb_val * u_1hop_means, axis=1)

                i_1hop_means = np.zeros(old_i_emb_val.shape)
                for i in range(len(i_1hop_means)):
                    i_idx = (sub_matrix_dense.T)[i].nonzero()
                    i_1hop_means[i] = np.mean(np.take(old_u_emb_val, i_idx[0], axis=0), axis=0)
                i_u_prod = np.sum(old_i_emb_val * i_1hop_means, axis=1)

                u_i_adj_idx = np.stack(np.nonzero(sub_matrix_dense), axis=-1)
                i_u_adj_idx = np.stack(np.nonzero(sub_matrix_dense.T), axis=-1)
                time_info.append(('finish calc local_anchor coef', time.time()))





            if parser.lambda_global_distill > 0:

                time_info.append(('start calc global_anchor coef', time.time()))
                n_u_anchor, n_i_anchor = eval(parser.k_centroids)

                u_kmeans = KMeans(n_clusters=n_u_anchor, random_state=0, n_jobs=10)
                u_idx = u_kmeans.fit_predict(old_u_emb_val)
                u_cluster_matrix = np.zeros([n_u_anchor, old_u_emb_val.shape[0]])  # n_anchor * n_user
                u_anchor_points = []
                u_cluster = np.zeros(old_u_emb_val.shape[0])

                for k in range(n_u_anchor):
                    k_idx = np.where(u_idx == k)[0]
                    u_cluster[k_idx] = k
                    u_cluster_matrix[k, k_idx] = 1


                    u_anchor_points.append(np.mean(np.take(old_u_emb_val, k_idx, axis=0), axis=0))
                u_anchor_points = np.array(u_anchor_points)

                i_kmeans = KMeans(n_clusters=n_i_anchor, random_state=0, n_jobs=10)
                i_idx = i_kmeans.fit_predict(old_i_emb_val)
                i_cluster_matrix = np.zeros([n_i_anchor, old_i_emb_val.shape[0]])
                i_anchor_points = []
                i_cluster = np.zeros(old_i_emb_val.shape[0])
                for k in range(n_i_anchor):
                    k_idx = np.where(i_idx == k)[0]
                    i_cluster[k_idx] = k
                    i_cluster_matrix[k, k_idx] = 1


                    # positive_samples = np.take(old_i_emb_val, k_idx, axis=0)
                    # negative_samples = np.take(old_i_emb_val, nk_idx, axis=0)

                    i_anchor_points.append(np.mean(np.take(old_i_emb_val, k_idx, axis=0), axis=0))
                i_anchor_points = np.array(i_anchor_points)


                u_cluster_adj_idx = np.stack(np.nonzero(u_cluster_matrix), axis=-1)
                i_cluster_adj_idx = np.stack(np.nonzero(i_cluster_matrix), axis=-1)
                anchor_points = np.concatenate([u_anchor_points, i_anchor_points], axis=0)

                # ===== clusters probability distillation ========

                u_gs_matrix = np.zeros([old_u_emb_val.shape[0], anchor_points.shape[0]])
                i_gs_matrix = np.zeros([old_i_emb_val.shape[0], anchor_points.shape[0]])

                for u, u_emb in enumerate(old_u_emb_val):
                    u_gs_matrix[u, :n_u_anchor] = np.sum(u_emb * anchor_points[:n_u_anchor], axis=1)
                    u_gs_matrix[u, n_u_anchor:] = np.sum(u_emb * anchor_points[n_u_anchor:], axis=1)
                    u_gs_matrix[u, :n_u_anchor] = softmax(u_gs_matrix[u, :n_u_anchor] / parser.tau)
                    u_gs_matrix[u, n_u_anchor:] = softmax(u_gs_matrix[u, n_u_anchor:] / parser.tau)

                for i, i_emb in enumerate(old_i_emb_val):
                    i_gs_matrix[i, :n_u_anchor] = np.sum(i_emb * anchor_points[:n_u_anchor], axis=1)
                    i_gs_matrix[i, n_u_anchor:] = np.sum(i_emb * anchor_points[n_u_anchor:], axis=1)
                    i_gs_matrix[i, :n_u_anchor] = softmax(i_gs_matrix[i, :n_u_anchor] / parser.tau)
                    i_gs_matrix[i, n_u_anchor:] = softmax(i_gs_matrix[i, n_u_anchor:] / parser.tau)

                time_info.append(('finish calc global_anchor coef', time.time()))


            model.user_embeddings.load(u_emb_val, sess)
            model.item_embeddings.load(i_emb_val, sess)

        _epoch = 0
        best_valid_recall20, best_valid_epoch, best_test_recall20 = 0., 0., 0.
        early_stop_flag = 0
        mse_user_reg, mse_item_reg = None, None

        time_info_training = 0
        time_info_eval = 0
        time_info_sampling = 0

        while _epoch <= n_epoch:

            time_info.append(('start epoch ' + str(_epoch) + ' training', time.time()))

            if _epoch % 1 == 0:

                time_info_eval_start = time.time()

                precision, v_recall, MAP, ndcg, _, __ = evaluate_model(sess, model, val_info, train_matrix,
                                                                       u_adj_list_val, v_adj_list_val,
                                                                       user_self_neighs_val, item_self_neighs_val,
                                                                       n_batch_users=parser.batch_evaluate)
                write_prediction_to_logger(logger, precision, v_recall, MAP, ndcg, _epoch, 'validation set')

                if v_recall[-1] > best_valid_recall20:

                    # accerlerate: only check testset when sees best model
                    precision, t_recall, MAP, ndcg, user_rep, item_rep = evaluate_model(sess, model, test_info,
                                                                                        train_matrix,
                                                                                        u_adj_list_test,
                                                                                        v_adj_list_test,
                                                                                        user_self_neighs_test,
                                                                                        item_self_neighs_test,
                                                                                        n_batch_users=parser.batch_evaluate)
                    write_prediction_to_logger(logger, precision, t_recall, MAP, ndcg, _epoch, 'test set')

                    early_stop_flag = 0
                    best_valid_recall20 = v_recall[-1]
                    best_valid_epoch = _epoch
                    best_test_recall20 = t_recall[-1]

                    if save_checkpoint != "":
                        # save embedding for next block incremental learning
                        saver.save(sess, save_checkpoint)
                        # save information for later use
                        if _epoch != 0:

                            if parser.reservoir_selection == 'mse_distill_score':
                                save_pickle(mse_user_reg, save_ckpt[:-10], 'user_distill_score')
                                save_pickle(mse_item_reg, save_ckpt[:-10], 'item_distill_score')

                early_stop_flag += 1
                if early_stop_flag > parser.patience and _epoch > parser.min_epoch:
                    if logger is not None:
                        logger.write('early stopp triggered at epoch: ' + str(_epoch) + '\n')
                    else:
                        print('early stopp triggered at epoch: ', str(_epoch))
                    break

                time_info_eval_end = time.time()
                time_info_eval += time_info_eval_end - time_info_eval_start

            _epoch += 1
            if _epoch > n_epoch:
                break

            for iter in range(0, num_iter):

                time_info_sampling_start = time.time()
                user_pos, neg_samples = sampler.next_batch()

                iter_start = time.time()
                time_info_sampling += iter_start - time_info_sampling_start
                feed_dict = {model.u_id: user_pos[:, 0],
                             model.pos_item_id: user_pos[:, 1],
                             model.neg_item_id: neg_samples,
                             model.u_adj_info_ph: u_adj_list,
                             model.v_adj_info_ph: v_adj_list,
                             model.u_u_graph_ph: user_self_neighs,
                             model.v_v_graph_ph: item_self_neighs,
                             model.old_user_embedding: u_emb_val,
                             model.old_item_embedding: i_emb_val}


                if parser.lambda_mse > 0 and node_deg_delta is not None:
                    feed_dict[model.u_mse_coef] = np.take(node_deg_delta[0], user_pos[:, 0])
                    feed_dict[model.i_mse_coef] = np.take(node_deg_delta[1],
                                                          np.concatenate((user_pos[:, 1], neg_samples.flatten())))
                    feed_dict[model.u_mse_coef_dist_score] = node_deg_delta[0][:n_old_user]
                    feed_dict[model.i_mse_coef_dist_score] = node_deg_delta[1][:n_old_item]
                if parser.lambda_distillation > 0:
                    if parser.local_mode == 'LSP_s':
                        feed_dict[model.old_user_bl_ls] = u_ls_matrix
                        feed_dict[model.old_item_bl_ls] = i_ls_matrix
                        feed_dict[model.old_user_bl_idx] = u_ls_index
                        feed_dict[model.old_item_bl_idx] = i_ls_index
                    elif parser.local_mode == 'local_anchor':
                        feed_dict[model.ui_dist] = u_i_prod
                        feed_dict[model.iu_dist] = i_u_prod
                        feed_dict[model.old_u_i_adj_mat] = (u_i_adj_idx, u_i_adj_idx[:, 1])
                        feed_dict[model.old_i_u_adj_mat] = (i_u_adj_idx, i_u_adj_idx[:, 1])



                if parser.lambda_global_distill > 0:
                    # ========= cluster anchors===========
                    feed_dict[model.old_user_embedding] = u_emb_val
                    feed_dict[model.old_item_embedding] = i_emb_val

                    feed_dict[model.old_user_gs] = u_gs_matrix
                    feed_dict[model.old_item_gs] = i_gs_matrix
                    feed_dict[model.old_u_cluster_mat] = (
                    u_cluster_adj_idx, u_cluster_adj_idx[:, 1])  # sparse matrix
                    feed_dict[model.old_i_cluster_mat] = (i_cluster_adj_idx, i_cluster_adj_idx[:, 1])
                    feed_dict[model.old_u_cluster] = u_cluster
                    feed_dict[model.old_i_cluster] = i_cluster


                _, bpr_loss, l2_reg, mse_user_reg, mse_item_reg = sess.run([model.ptmzr,
                                                                model.bpr_loss,
                                                                model.reg_loss,
                                                                model.mse_user_reg,
                                                                model.mse_item_reg,],feed_dict=feed_dict)



                print('Epoch ', '%04d' % _epoch, 'iter ', '%02d' % iter,
                      'bpr_loss=', '{:.5f}, cost {:.4f} seconds'.format(bpr_loss, time.time()-iter_start))
                iter_time.append(time.time()-iter_start)

            if process_saver:
                if not os.path.exists(f"{weights_saver}/{parser.setting}"):
                    os.makedirs(f"{weights_saver}/{parser.setting}")
                # np.save(f"{weights_saver}/{parser.setting}/user_weight_{_epoch}.npy", user_weight)
            time_info.append(('finish epoch ' + str(_epoch) + ' training', time.time()))
            time_info_training = sum(iter_time)
        time_info.append(('finish final epoch training', time.time()))
        time_info.append(('total training time', time_info_training))
        time_info.append(('total eval time  ', time_info_eval))
        time_info.append(('total sampling time', time_info_sampling))
        if weights_saver:
            if not os.path.exists(f"{weights_saver}/{parser.setting}"):
                os.makedirs(f"{weights_saver}/{parser.setting}")
            # np.save(f"{weights_saver}/{parser.setting}/user_weight.npy", user_weight)
            np.save(f"{weights_saver}/{parser.setting}/old_user_embedding.npy", old_u_emb_val)
            np.save(f"{weights_saver}/{parser.setting}/old_item_embedding.npy", old_i_emb_val)
            np.save(f"{weights_saver}/{parser.setting}/user_embedding.npy", u_emb_val)
            np.save(f"{weights_saver}/{parser.setting}/item_embedding.npy", i_emb_val)


        # if parser.layer_wise_calc:
        #     if not os.path.exists(f"{medium_saver}/{parser.setting}"):
        #         os.makedirs(f"{medium_saver}/{parser.setting}")
            # np.save(f"{medium_saver}/{parser.setting}/user_all_inputs.npy", user_all_inputs, allow_pickle=True)
            # np.save(f"{medium_saver}/{parser.setting}/item_all_inputs.npy", item_all_inputs, allow_pickle=True)
            # np.save(f"{medium_saver}/{parser.setting}/user_all_ids.npy", user_all_ids, allow_pickle=True)
            # np.save(f"{medium_saver}/{parser.setting}/item_all_ids.npy", item_all_ids, allow_pickle=True)


    sampler.close()
    if parser.log_name:
        logger.write("training time: " + str(sum(iter_time)) + '\n')
        logger.write('best_valid_epoch, best_valid_recall20, best_test_recall20' + '\n')
        logger.write(str([best_valid_epoch, best_valid_recall20, best_test_recall20]) + '\n')
    else:
        print("training time: " + str(sum(iter_time)) + '\n')
        print('best_valid_epoch, best_valid_recall20, best_test_recall20')
        print(str([best_valid_epoch, best_valid_recall20, best_test_recall20]))


def evaluate_model(sess, model, test_info, train_matrix, u_adj_list, v_adj_list, user_self_neighs, item_self_neighs,
                   n_batch_users=1024):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    test_set, n_user, n_item, test_matrix = test_info
    num_batches = int(n_user / n_batch_users) + 1
    user_indexes = np.arange(n_user)
    topk = 100
    precision, recall, MAP, ndcg, ngcf_recall = [], [], [], [], []
    pred_list = None
    items = np.arange(0, n_item, 1, dtype=int)

    for batchID in range(num_batches):
        start = batchID * n_batch_users
        end = start + n_batch_users

        if batchID == num_batches - 1:
            if start < n_user:
                end = n_user
            else:
                break

        batch_user_index = user_indexes[start:end]

        feed_dict = {}
        feed_dict[model.u_adj_info_ph] = u_adj_list
        feed_dict[model.v_adj_info_ph] = v_adj_list
        feed_dict[model.u_u_graph_ph] = user_self_neighs
        feed_dict[model.v_v_graph_ph] = item_self_neighs

        n_user_train, n_item_train = model.num_user, model.num_item
        if n_user > n_user_train:
            old_user_embedding = sess.run(model.user_embeddings)
            new_users_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('user', old_user_embedding,
                                                                                       test_matrix)
        else:
            new_users_init = None
        if n_item > n_item_train:
            old_item_embedding = sess.run(model.item_embeddings)
            new_items_init = get_node_init_embedding_by_aggregating_two_hop_neighbours('item', old_item_embedding,
                                                                                       test_matrix)
        else:
            new_items_init = None

        rating_preds, user_rep, item_rep = model.predict(batch_user_index, items, n_user, new_users_init,
                                                         new_items_init)

        rating_preds, user_rep, item_rep = sess.run([rating_preds, user_rep, item_rep], feed_dict)

        train_matrix = train_matrix[:n_user_train, :n_item_train]
        rating_preds[
            train_matrix[batch_user_index[0]:min(train_matrix.shape[0], batch_user_index[-1] + 1)].nonzero()] = 0
        ind = np.argpartition(rating_preds, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_preds[np.arange(len(rating_preds))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_preds)), ::-1]
        pred_items = ind[np.arange(len(rating_preds))[:, None], arr_ind_argsort]

        if batchID == 0:
            pred_list = pred_items.copy()
        else:
            pred_list = np.append(pred_list, pred_items, axis=0)

    for k in [5, 10, 15, 20]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(batch_ndcg_at_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg, user_rep, item_rep


if __name__ == '__main__':

    time_info = []
    time_info.append(('Program begins', time.time()))



    # parse arguments
    parser = config2.parse_arguments()
    print('Using GPU ' + str(parser.device))
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.device
    os.environ['TF_DETERMINISTIC_OPS'] = str(parser.seed)
    os.environ['PYTHONHASHSEED'] = str(parser.seed)
    LOG_SAVE_PATH_PREFIX = parser.load_save_path_prefix
    # set seed
    np.random.seed(parser.seed)
    tf.random.set_random_seed(parser.seed)
    random.seed(parser.seed)

    # checkpoint and embedding save path
    save_ckpt = LOG_SAVE_PATH_PREFIX + parser.log_folder + '/' + parser.save_cp + '.ckpt' if parser.save_cp else ''
    load_ckpt = LOG_SAVE_PATH_PREFIX + parser.log_folder + '/' + parser.load_cp + '.ckpt' if parser.load_cp else ''

    # loading data
    data_generator = Data(dataset=parser.dataset, split=eval(parser.data_split),
                          shuffle=parser.shuffle, split_mode='abs', test_ratio=0,
                          seed=parser.seed, replay_ratio=parser.replay_ratio, sliding_ratio=parser.sliding_ratio)
    data_blocks = data_generator.blocks

    time_info.append(('Data loader done', time.time()))

    # if training the incremental block
    if parser.n_inc > 0:
        assert parser.n_inc + parser.block < len(data_blocks)
        saved_ckpt = []
        for inc_block in range(parser.n_inc):

            time_info.append(('n_inc ' + str(inc_block) + ' begins', time.time()))

            # create logger before training
            if parser.log_name:
                now = (datetime.datetime.utcnow() - datetime.timedelta(hours=4)).strftime(
                    "%b_%d_%H_%M_%S") + '-base' + str(parser.block - 1) + 'inc' + str(inc_block + 1)
                name = parser.log_folder + '/' + parser.log_name + '-' + parser.dataset
                log_save_path = LOG_SAVE_PATH_PREFIX + name + '/' + now
                result_log_name = parser.log_files + '/' + parser.dataset + '/' + parser.setting
                logger = Logger(result_log_name, name, now, parser.load_save_path_prefix)
                logger.open(result_log_name + f'/log.train_{inc_block+1}.txt', mode='a')
                for arg in vars(parser):
                    logger.write(arg + '=' + str(getattr(parser, arg)) + '\n')
            else:
                logger = None

            if parser.n_inc > 1:
                assert parser.save_cp == ''  # can save the updated model if there is only one inc block
                save_ckpt = log_save_path + '/model.ckpt'
                saved_ckpt.append(save_ckpt)

            cur_block = parser.block + inc_block
            next_blcok = cur_block + 1
            prev_block = cur_block - 1
            weights_saver = f"Saved_weights/{parser.dataset}_{cur_block}"
            medium_saver = f"Medium_input/{parser.dataset}_{cur_block}"

            # determine train_set and test_set
            if parser.train_mode == 'acc':
                train_set = data_blocks[cur_block]['acc_train']
            else:
                train_set = data_blocks[cur_block]['train']
            n_user_train, n_item_train = data_blocks[cur_block]['n_user_train'], data_blocks[cur_block]['n_item_train']
            n_old_user_train, n_old_item_train = data_blocks[prev_block]['n_user_train'], data_blocks[prev_block][
                'n_item_train']
            cur_block_matrix = data_blocks[cur_block]['train_matrix']
            prev_block_matrix = data_blocks[prev_block]['train_matrix']

            val_set = data_blocks[cur_block]['val']
            n_item_val, n_user_val = data_blocks[cur_block]['n_item_val'], len(val_set)
            cur_val_matrix = data_blocks[cur_block]['val_matrix']
            test_set = data_blocks[cur_block]['test']
            n_item_test, n_user_test = data_blocks[cur_block]['n_item_test'], len(test_set)

            inc_full_batch_append = ''
            if parser.inc_full_batch:
                assert parser.lambda_mse == 0
                assert parser.reservoir_mode == ''
                assert parser.inc_agg == 0
                assert parser.train_mode == 'acc'
                inc_full_batch_append = '_fb'

                n_user_train, n_item_train = n_user_val, n_item_val
                train_set = union_lists_of_list(train_set, val_set)
                train_set, val_set = split_data_randomly(train_set, test_ratio=0.05, seed=parser.seed)
                cur_block_matrix = generate_sparse_adj_matrix(train_set, n_user_train, n_item_train)
                cur_val_matrix = generate_sparse_adj_matrix(val_set, n_user_val, n_item_val)


            node_deg_delta = None
            graph_path = [
                parser.graph_path + parser.dataset + '/' + 'uu_graph_' + str(cur_block) + inc_full_batch_append + '.pkl', \
                parser.graph_path + parser.dataset + '/' + 'ii_graph_' + str(cur_block) + inc_full_batch_append + '.pkl']
            prev_graph_path = [
                parser.graph_path + parser.dataset + '/' + 'uu_graph_' + str(prev_block) + inc_full_batch_append + '.pkl', \
                parser.graph_path + parser.dataset + '/' + 'ii_graph_' + str(prev_block) + inc_full_batch_append + '.pkl']

            # calculating regularizer coefficient
            if parser.lambda_mse > 0:
                time_info.append(('start calc mse coef', time.time()))

                u_deg, i_deg = np.array(prev_block_matrix.sum(axis=1)).flatten(), np.array(
                    prev_block_matrix.sum(axis=0)).flatten()
                new_u_deg, new_i_deg = np.array(cur_block_matrix.sum(axis=1)).flatten(), np.array(
                    cur_block_matrix.sum(axis=0)).flatten()

                new_u_deg, new_i_deg = new_u_deg[:n_old_user_train], new_i_deg[:n_old_item_train]
                u_deg_delta, i_deg_delta = u_deg / (new_u_deg + 1e-8), i_deg / (new_i_deg + 1e-8)
                u_deg_norm, i_deg_norm = u_deg_delta / np.linalg.norm(u_deg_delta), i_deg_delta / np.linalg.norm(
                    i_deg_delta)

                delta_n_user = n_user_train - n_old_user_train
                delta_n_item = n_item_train - n_old_item_train
                u_mse_delta = np.concatenate([u_deg_norm, np.zeros(delta_n_user)])
                i_mse_delta = np.concatenate([i_deg_norm, np.zeros(delta_n_item)])
                node_deg_delta = [u_mse_delta, i_mse_delta]

                time_info.append(('finish calc mse coef', time.time()))


            if parser.reservoir_mode != '':

                time_info.append(('start calc reservoir', time.time()))

                assert parser.reservoir_selection != ''
                if parser.reservoir_mode == 'reservoir_sampling':
                    assert parser.sliding_ratio > 0

                print('==============', cur_block, '===============')

                if cur_block == 1:
                    replay_size = int(data_generator.data_size * parser.replay_ratio)
                    reservoir_size = int(
                        data_generator.data_size * parser.sliding_ratio) if parser.reservoir_mode == 'reservoir_sampling' else None
                    reservoir = Reservoir(data_blocks[prev_block], parser.reservoir_mode, replay_size,
                                          sample_mode=parser.reservoir_selection, merge_mode=parser.union_mode,
                                          sample_per_user=0, reservoir_size=reservoir_size)

                reservoir.set_logger(logger)
                if parser.union_mode == 'snu':
                    assert parser.replay_ratio > 0
                if parser.union_mode == 'uns':
                    union_lists = union_lists_of_list(reservoir.reservoir, train_set)


                if parser.reservoir_selection == 'boosting_inner_product' \
                        or parser.reservoir_selection == 'boosting_wasserstein':
                    assert load_ckpt != ''
                    if parser.union_mode == 'snu':
                        pred_score = get_model_prediction(parser, load_ckpt, prev_block_matrix,
                                                          reservoir.n_reservoir_user, reservoir.n_reservoir_item,
                                                          graph_path=prev_graph_path)
                        inc_train_data = reservoir.get_inc_train_data(train_set, predict_score=pred_score)
                    elif parser.union_mode == 'uns':
                        pred_info = [n_user_train, n_item_train]
                        pred_score = get_model_prediction(parser, load_ckpt, cur_block_matrix,
                                                          reservoir.n_reservoir_user, reservoir.n_reservoir_item,
                                                          graph_path=graph_path, test_info=pred_info)
                        inc_train_data = reservoir.get_inc_train_data(union_lists, predict_score=pred_score,
                                                                      n_new_user=n_user_train, n_new_item=n_item_train,
                                                                      cur_block_train_size=get_list_of_lists_size(
                                                                          train_set))
                    else:
                        raise NotImplementedError
                elif parser.reservoir_selection == 'mse_distill_score':
                    assert parser.union_mode == 'snu'
                    if cur_block == 1:
                        # for the first incremental block, we do not have distillation score,
                        # thus we use random sampling here.
                        pred_score = np.array([1])
                    else:
                        user_pred_score = load_pickle(load_ckpt[:-10], 'user_distill_score.pkl')
                        pred_score_u = np.zeros(reservoir.n_reservoir_user)
                        pred_score_u[:user_pred_score.shape[0]] = user_pred_score
                        # pred_score_u = (pred_score_u - pred_score_u.min()) / pred_score_u.std()

                        item_pred_score = load_pickle(load_ckpt[:-10], 'item_distill_score.pkl')
                        pred_score_i = np.zeros(reservoir.n_reservoir_item)
                        pred_score_i[:item_pred_score.shape[0]] = item_pred_score
                        # pred_score_i = (pred_score_i - pred_score_i.min()) / pred_score_i.std()

                        pred_score = pred_score_u.reshape(-1, 1) + pred_score_i.reshape(1, -1)
                    inc_train_data = reservoir.get_inc_train_data(train_set, predict_score=pred_score)
                elif parser.reservoir_selection in ['uniform', 'inverse_deg', 'prop_deg', 'adp_inverse_deg']:
                    if parser.union_mode == 'snu':
                        if parser.reservoir_selection == 'adp_inverse_deg':
                            new_data_mat = generate_sparse_adj_matrix(train_set, n_user_train, n_item_train)
                        else:
                            new_data_mat = None
                        inc_train_data = reservoir.get_inc_train_data(train_set, new_data_mat=new_data_mat)
                    elif parser.union_mode == 'uns':
                        inc_train_data = reservoir.get_inc_train_data(union_lists, n_new_user=n_user_train,
                                                                      n_new_item=n_item_train,
                                                                      cur_block_train_size=get_list_of_lists_size(
                                                                          train_set))
                    else:
                        raise NotImplementedError
                elif parser.reservoir_selection == 'item_embedding':
                    if parser.union_mode == 'snu':
                        old_i_embedding = tf.train.load_variable(load_ckpt, 'model/item_embedding')[
                                          :data_blocks[prev_block]["n_item_train"]]
                        inc_train_data = reservoir.get_inc_train_data_item_embedding(train_set, old_i_embedding,
                                                                                     cur_block_matrix,
                                                                                     'TODO: MANUALLY PUT A THRESHOLD HERE')
                    else:
                        raise NotImplementedError
                elif parser.reservoir_selection == 'latest':
                    assert parser.union_mode == 'snu'
                    inc_train_data = union_lists_of_list(data_blocks[cur_block]['latest_reservoir'], train_set)
                else:
                    raise NotImplementedError

                if cur_block < parser.n_inc:
                    print("update reservoir")
                    if parser.reservoir_mode == 'sliding':
                        reservoir.update(train_set, n_user_train, n_item_train,
                                         data_blocks[cur_block + 1]['sliding_lists'],
                                         data_blocks[cur_block + 1]['sliding_matrix'])
                    else:
                        reservoir.update(train_set, n_user_train, n_item_train)
                train_set = inc_train_data

                time_info.append(('finish calc reservoir', time.time()))

                if parser.inc_agg == 1:
                    assert parser.reservoir_mode == 'sliding'
                    acc_train_list = union_lists_of_list(data_blocks[cur_block]['sliding_lists'], train_set)
                    cur_block_matrix = generate_sparse_adj_matrix(acc_train_list, n_user_train, n_item_train)

            # training model
            time_info.append(('before enter train_model', time.time()))

            train_model(parser,
                        [train_set, n_user_train, n_item_train, cur_block_matrix],
                        [val_set, n_user_val, n_item_val, cur_val_matrix],
                        [test_set, n_user_test, n_item_test, data_blocks[cur_block]['test_matrix']],
                        data_blocks[cur_block - 1]['train'],
                        data_blocks[cur_block - 1]['train_matrix'],
                        parser.num_epoch,
                        n_old_user=n_old_user_train,
                        n_old_item=n_old_item_train,
                        node_deg_delta=node_deg_delta,
                        logger=logger,
                        load_checkpoint=load_ckpt,
                        save_checkpoint=save_ckpt,
                        graph_path=graph_path,
                        process_saver=True,
                        weights_saver=weights_saver,
                        medium_saver=medium_saver)

            load_ckpt = save_ckpt

        # clean up all saved models
        for ckpt in saved_ckpt:
            os.system('rm ' + ckpt + '.*')


    # training for the very first block (not incremental)
    else:
        time_info.append(('start base model training', time.time()))
        # create logger
        if parser.log_name:
            now = (datetime.datetime.utcnow() - datetime.timedelta(hours=4)).strftime("%b_%d_%H_%M_%S")
            name = parser.log_folder + '/' + parser.log_name + '-' + parser.dataset
            log_save_path = LOG_SAVE_PATH_PREFIX + name + '/' + now
            result_log_name = parser.log_files + '/' + parser.dataset + '/' + parser.setting
            logger = Logger(result_log_name, name, now, parser.load_save_path_prefix)
            logger.open(result_log_name + f'/log.train_{parser.block}.txt', mode='a')
            for arg in vars(parser):
                logger.write(arg + '=' + str(getattr(parser, arg)) + '\n')
        else:
            logger = None

        # for the first block, we use the same block for train and test
        n_user, n_item = data_blocks[parser.block]['n_user_train'], data_blocks[parser.block]['n_item_train']
        test_user, test_item = n_user, n_item

        if parser.train_mode == 'acc':
            train_set = data_blocks[parser.block]['acc_train']
        else:
            train_set = data_blocks[parser.block]['train']

        train_set, test_set = split_data_randomly(train_set, test_ratio=0.2, seed=parser.seed)
        test_set, val_set = split_data_randomly(test_set, test_ratio=0.5, seed=parser.seed)
        train_matrix = generate_sparse_adj_matrix(train_set, n_user, n_item)
        val_matrix = generate_sparse_adj_matrix(val_set, n_user, n_item)
        test_matrix = generate_sparse_adj_matrix(test_set, n_user, n_item)

        graph_path = [parser.graph_path + parser.dataset + '/' + 'uu_graph_0.npy', \
                      parser.graph_path + parser.dataset + '/' + 'ii_graph_0.npy']

        medium_saver = f"Medium_input/{parser.dataset}_0"
        train_model(parser,
                    [train_set, n_user, n_item, train_matrix],
                    [val_set, test_user, test_item, val_matrix],
                    [test_set, test_user, test_item, test_matrix],
                    None,
                    None,
                    parser.num_epoch,
                    logger=logger,
                    load_checkpoint=load_ckpt,
                    save_checkpoint=save_ckpt,
                    graph_path=graph_path,
                    medium_saver = medium_saver)

        time_info.append(('finish calc mse coef', time.time()))
