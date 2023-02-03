import tensorflow as tf
import sys
from .layers import Dense,FCL
from .aggregators import MyMeanAggregator, MaxPoolAggregator
import numpy as np

"""
TensorFlow Implementation for "Multi-Graph Convolution Collaborative Filtering, Jianing Sun et.al, IEEE ICDM 2019"
__author__: Jianing Sun
__date__: Dec. 2019
__copyright__: Montreal Research Center, Huawei Technologies
"""

def calculate_contrastive_loss(old_embedding, old_num, cur_pos_neigh, cur_neg_neigh, tau, user_weight=None):
    '''

    :param old_embedding: The embedding from previous time point
    :param old_num: The number of users/items of previous time point
    :param cur_pos_neigh: The embedding of time point t of positive neighbors from previous time point
    :param cur_neg_neigh: The embedding of time point t of negative neighbors from previous time point
    :param tau: hyperparameter
    :param user_weight: adaptive distillation weights for each user
    :return: contrastive loss term
    '''
    numerator_user = tf.math.exp(
        tf.reduce_sum(tf.multiply(tf.expand_dims(old_embedding[:old_num], 1),
                                  cur_pos_neigh) / tau, axis=2))
    denom_user = tf.reduce_sum(tf.math.exp(
        tf.reduce_sum(tf.multiply(tf.expand_dims(old_embedding[:old_num], 1),
                                  cur_neg_neigh) / tau, axis=2)))

    if user_weight != None:
        ct_loss = tf.reduce_mean(
            user_weight * tf.reduce_mean(-tf.math.log(numerator_user / denom_user), 1))
    else:
        ct_loss = tf.reduce_mean(tf.reduce_mean(-tf.math.log(numerator_user / denom_user), 1))
    return ct_loss

class WeightGenerator(object):
    '''
    Weight generator that takes state vector which indicates users' interest shift and generates personalized
    weights for each users' level of distillation.
    '''
    def __init__(self, name, input_dim, hidden_dim, output_dim=1, dropout=0.1, act='tf.nn.relu'):
        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)

        if act:
            self.act = eval(act)
        else:
            self.act = act
        self.mlp_layers = []
        self.mlp_layers.append(FCL(name=name + '_1',
                                     input_dim=input_dim,
                                     output_dim=hidden_dim,
                                     act=self.act,
                                     dropout=dropout))
        self.weight = tf.Variable(initializer([hidden_dim, output_dim]), name=name + 'output_weight')
    def __call__(self, state_vector):
        for dense_layer in self.mlp_layers:
            state_vector = dense_layer(state_vector)
        user_weights = tf.matmul(state_vector, self.weight)
        user_weights = tf.math.softplus(user_weights)
        return user_weights

def uniform_sample(ids, adj, num_samples=15):
    ''' uniform sample for central nodes
    :param ids: (N, )
    :param num_samples: number of sampled neighbors
    :return: adj_list with feature and ids. shape: (none, num_samples)
    '''
    adj_lists = tf.nn.embedding_lookup(adj, ids)  # (N, 128)
    adj_lists = tf.transpose(tf.random.shuffle(tf.transpose(adj_lists)))
    adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])  # (N, num_samples)
    return adj_lists



class GNN(object):
    def __init__(self, dataset_argv, architect_argv, adj_degree, num_samples,
                 ptmzr_argv, aggregator, act, neigh_drop_rate, l2_embed, num_self_neigh=10,
                 neg_item_num=10, con_positive=15, con_negative=15, trans_positive=10, pretrain_data=None,
                 inc_reg=[0, 0, 0], old_num_user=0, old_num_item=0, distill_mode='',k_centroids=[0, 0], tau=0, num_neigh=15,
                 local_distill_mode='', contrastive_mode = '', layer_wise=None, lambda_contrastive=0, lambda_soft=0,
                 adaptive_mode = '', wg_dimension = 128, layer_wise_calc=None, wg_act='tf.nn.relu', nu=1):
        '''
        :param dataset_argv:[embedded_dimension[0], n_user, n_item]
        :param architect_argv: embedded_dimension[1:]
        :param adj_degree: max_degree
        :param num_samples: The number of sampled [1-hop, 2-hop] neighbors for GCN
        :param ptmzr_argv: [optimizer, learning_rate, epsilon]
        :param aggregator: GCN aggregation function
        :param act: activation function
        :param neigh_drop_rate: neighbor drop rate
        :param l2_embed: weight decay of embedding
        :param num_self_neigh: the number of positive neighbors user_user of item_item_graph
        :param neg_item_num: number of negative paris for each positve pair in bpr loss
        :param pretrain_data: pretrained data for initializing embeddings, default is none
        :param inc_reg: [lambda_mse, lambda_distillation, lambda_global_distill]
        :param old_num_user: number of old user
        :param old_num_item: number of old item
        :param k_centroids: number of centroids
        :param tau: Global distillation parameter
        :param num_neigh: the number of positive neighbors for contrastive loss
        :param local_distill_mode:
        :param contrastive_mode: multi-graph or single graph
        :param layer_wise: whether use layer-wise distillation
        :param con_negative: number of negative samples of contrastive loss
        :param con_positive: number of positive samples of contrastive loss
        :param adaptive_mode: How to calculate user adaptive weights
        '''

        self.graph = tf.Graph()

        with self.graph.as_default():
            (self.input_dim, self.num_user, self.num_item) = dataset_argv

            self.layer_dims = architect_argv
            self.neg_item_num = neg_item_num
            print('input dim: %d\n'
                  'neigh_drop_rate: %g\nl2(lambda): %g\n' %
                  (self.input_dim, neigh_drop_rate, l2_embed))

            self.old_num_user, self.old_num_item = old_num_user, old_num_item

            self.k_centroids = k_centroids
            self.inc_reg = inc_reg
            self.mse_reg_flag = True if inc_reg[0] > 0 else False
            self.distillation_flag = True if inc_reg[1] > 0 else False
            self.global_distillation_flag = True if inc_reg[2] > 0 else False
            self.tau = tau
            self.num_neigh = tf.cast(num_neigh, dtype=tf.int32)
            self.distill_mode = distill_mode
            self.local_distill_mode = local_distill_mode
            self.contrastive_mode = contrastive_mode
            self.layer_wise = layer_wise
            self.layer_wise_calc = layer_wise_calc
            self.lambda_contrastive = lambda_contrastive
            self.lambda_soft = lambda_soft
            self.adaptive_mode = adaptive_mode
            self.wg_dimension = wg_dimension
            self.wg_act = wg_act
            self.nu = nu


            self.user_agg_funcs, self.item_agg_funcs = [], []
            self.layer_dims = [self.input_dim] + self.layer_dims
            self.num_layers = len(self.layer_dims)  # Total number of NN layers

            # For dimension uniformality
            if old_num_user < self.num_user:
                self.old_user_ids = tf.range(old_num_user, dtype=tf.int32)
            else:
                self.old_user_ids = tf.range(num_user, dtype=tf.int32)

            if old_num_item < self.num_item:
                self.old_item_ids = tf.range(old_num_item, dtype=tf.int32)
            else:
                self.old_item_ids = tf.range(num_item, dtype=tf.int32)

            # # for debug purpose, please ignore
            self.user_reg = tf.constant(0.0, dtype=tf.float64)
            self.item_reg = tf.constant(0.0, dtype=tf.float64)

            #
            self.num_self_neigh = num_self_neigh

            # =================================================================================================
            self.u_id = tf.placeholder(tf.int32, shape=[None, ], name='u_id')
            self.random_ids = tf.placeholder(tf.int32, shape=[None, ], name='random_id')
            self.pos_item_id = tf.placeholder(tf.int32, shape=[None, ], name='pos_item_id')
            self.neg_item_id = tf.placeholder(tf.int32, shape=[None, neg_item_num], name='neg_item_id')

            self.u_adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, adj_degree], name='u_adj_info_ph')
            self.v_adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, adj_degree], name='v_adj_info_ph')

            # Current
            self.u_u_graph_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, num_neigh],
                                                         name='u_u_graph_ph')
            self.v_v_graph_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, num_neigh],
                                                         name='v_v_graph_ph')

            self.u_u_agg_func = MyMeanAggregator('u_u_agg_func', self.num_self_neigh, self.input_dim,
                                                 self.input_dim, self.layer_dims[-1], activation=eval(act),
                                                 dropout=neigh_drop_rate)
            self.v_v_agg_func = MyMeanAggregator('v_v_agg_func', self.num_self_neigh, self.input_dim,
                                                 self.input_dim, self.layer_dims[-1], activation=eval(act),
                                                 dropout=neigh_drop_rate)

            self.u_mse_coef = tf.placeholder(tf.float64, shape=[None, ], name='u_mse_coef_ph')
            self.i_mse_coef = tf.placeholder(tf.float64, shape=[None, ], name='i_mse_coef_ph')
            self.u_mse_coef_dist_score = tf.placeholder(tf.float64, shape=[None, ], name='u_mse_coef_ds_ph')
            self.i_mse_coef_dist_score = tf.placeholder(tf.float64, shape=[None, ], name='i_mse_coef_ds_ph')
            self.old_user_embedding = tf.placeholder(tf.float64, shape=[None, self.layer_dims[0]], name='u_old_emb_ph')
            self.old_item_embedding = tf.placeholder(tf.float64, shape=[None, self.layer_dims[0]], name='i_old_emb_ph')
            # For layer_wise
            self.old_user_all_inputs = tf.placeholder(tf.float64, shape=[self.num_layers, None, self.layer_dims[0]],
                                                      name='u_old_all_inputs')
            self.old_item_all_inputs = tf.placeholder(tf.float64, shape=[self.num_layers, None, self.layer_dims[0]],
                                                      name='i_old_all_inputs')
            # For contrastive_loss
            self.old_uu_pos_neighs = tf.placeholder(tf.int32, shape=[None, con_positive], name='old_uu_pos_neighs')
            self.old_ii_pos_neighs = tf.placeholder(tf.int32, shape=[None, con_positive], name='old_ii_pos_neighs')
            self.old_uu_neg_neighs = tf.placeholder(tf.int32, shape=[None, con_negative], name='old_uu_neg_neighs')
            self.old_ii_neg_neighs = tf.placeholder(tf.int32, shape=[None, con_negative], name='old_ii_neg_neighs')

            self.old_ui_pos_neighs = tf.placeholder(tf.int32, shape=[None, con_positive], name='old_ui_pos_neighs')
            self.old_iu_pos_neighs = tf.placeholder(tf.int32, shape=[None, con_positive], name='old_iu_pos_neighs')
            self.old_ui_neg_neighs = tf.placeholder(tf.int32, shape=[None, con_negative], name='old_ui_neg_neighs')
            self.old_iu_neg_neighs = tf.placeholder(tf.int32, shape=[None, con_negative], name='old_iu_neg_neighs')
            # For adaptive weights
            # self.old_ui_pos_trans = tf.placeholder(tf.int32, shape=[None, trans_positive], name='old_ui_pos_trans')
            #
            # self.user_item_cat_diff = tf.placeholder(tf.float64, shape=[None, k_centroids[1] * self.input_dim],
            #                                          name="user_item_cat_count")
            # self.user_weight_icount = WeightGenerator('user_weight_item_count', k_centroids[1] * self.input_dim,
            #                                           self.wg_dimension, act = self.wg_act)
            # self.user_weight_icluster = WeightGenerator('user_weight_item_cluster', k_centroids[1],
            #                                           self.wg_dimension, act=self.wg_act)
            # self.user_weight_itrans = WeightGenerator('user_weight_item_trans', trans_positive,
            #                                             self.wg_dimension, act=self.wg_act)
            #************************************************************************************************

            # local anchor distillation
            self.ui_dist = tf.placeholder(tf.float64, shape=[self.old_num_user], name='ui_dist_ph')
            self.iu_dist = tf.placeholder(tf.float64, shape=[self.old_num_item], name='iu_dist_ph')
            self.old_u_i_adj_mat = tf.sparse.placeholder(tf.int32, shape=[self.old_num_user, self.old_num_item],
                                                         name='ui_1hop_mat_ph')
            self.old_i_u_adj_mat = tf.sparse.placeholder(tf.int32, shape=[self.old_num_item, self.old_num_user],
                                                         name='iu_1hop_mat_ph')

            # LSP_s
            self.old_user_bl_ls = tf.placeholder(tf.float64, shape=[self.old_num_user, 10], name='u_old_b_ls_ph')
            self.old_item_bl_ls = tf.placeholder(tf.float64, shape=[self.old_num_item, 10], name='i_old_b_ls_ph')
            self.old_user_bl_idx = tf.placeholder(tf.int32, shape=[self.old_num_user, 10], name='u_old_b_id_ph')
            self.old_item_bl_idx = tf.placeholder(tf.int32, shape=[self.old_num_item, 10], name='i_old_b_id_ph')

            # global anchor distillation
            self.old_user_gs = tf.placeholder(tf.float64,
                                              shape=[self.old_num_user, self.k_centroids[0] + self.k_centroids[1]],
                                              name='u_old_gs_ph')
            self.old_item_gs = tf.placeholder(tf.float64,
                                              shape=[self.old_num_item, self.k_centroids[1] + self.k_centroids[0]],
                                              name='u_old_gs_ph')
            self.old_u_cluster_mat = tf.sparse.placeholder(tf.int32, shape=[self.k_centroids[0], self.old_num_user],
                                                           name='old_u_cluster_mat_ph')
            self.old_i_cluster_mat = tf.sparse.placeholder(tf.int32, shape=[self.k_centroids[1], self.old_num_item],
                                                           name='old_i_cluster_mat_ph')
            self.old_u_cluster = tf.placeholder(tf.int32, shape=[self.old_num_user, ], name="old_u_labels")
            self.old_i_cluster = tf.placeholder(tf.int32, shape=[self.old_num_item, ], name="old_i_labels")

            # =================================================================================================

            self.num_samples = num_samples
            self.l2_embed = l2_embed
            self.pretrain_data = pretrain_data


            gcn_act = eval(act)
            self.neigh_dropout = neigh_drop_rate
            self.wg_dimension = wg_dimension

            for i in range(1, self.num_layers):
                if aggregator == 'my_mean':
                    self.user_agg_funcs.append(MyMeanAggregator('user_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                self.input_dim, self.layer_dims[i - 1],
                                                                self.layer_dims[i],
                                                                activation=gcn_act, dropout=neigh_drop_rate))
                    self.item_agg_funcs.append(MyMeanAggregator('item_graph_agg_%d' % i, self.num_samples[::-1][i - 1],
                                                                self.input_dim, self.layer_dims[i - 1],
                                                                self.layer_dims[i],
                                                                activation=gcn_act, dropout=neigh_drop_rate))

            self.user_agg_funcs = self.user_agg_funcs[::-1]
            self.item_agg_funcs = self.item_agg_funcs[::-1]
            # ===
            self.mse_user_reg, self.mse_item_reg = tf.zeros(1), tf.zeros(1)

            self.bpr_loss, self.reg_loss, self.dist_loss =  self.model_fn('model')

            assert ptmzr_argv[0].lower() == 'adam'
            _learning_rate, _epsilon = ptmzr_argv[1:3]
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                assert ptmzr_argv[0].lower() == 'adam'
                self.ptmzr = tf.compat.v1.train.AdamOptimizer(learning_rate=_learning_rate,
                                                              epsilon=_epsilon).minimize(
                    self.bpr_loss + self.reg_loss + self.dist_loss)

    def model_fn(self, scope):
        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64, seed=0)
        with tf.compat.v1.variable_scope(scope):
            if self.pretrain_data is None:
                self.user_embeddings = tf.Variable(initializer([self.num_user, self.input_dim]), name='user_embedding')
                self.item_embeddings = tf.Variable(initializer([self.num_item, self.input_dim]), name='item_embedding')
                print('=== using xavier initialization for embeddings, no pretrain')
            else:
                self.user_embeddings = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                   name='user_embedding', dtype=tf.float32)
                self.item_embeddings = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                   name='item_embedding', dtype=tf.float32)
                print('=== using pretrained data for initializing embeddings, but still trainable')

            # ============================================================

            batch_size = tf.shape(self.u_id)[0]

            pos_users = tf.reshape(uniform_sample(self.u_id, self.u_u_graph_ph, 1), [-1, ])
            all_user_rep, all_user_embed, _, _ = self.graphconv('user_gcn', tf.concat([self.u_id, pos_users], 0),
                                                          self.user_embeddings, self.item_embeddings, 'user')


            all_user_rep = tf.concat([all_user_rep, all_user_embed], 1)

            user_rep, pos_user_rep = tf.split(all_user_rep, [batch_size, batch_size])
            user_embed, pos_user_embed = tf.split(all_user_embed, [batch_size, batch_size])

            user_user_distance = tf.reduce_sum(tf.math.pow(user_rep - pos_user_rep, 2)) \
                                 + tf.reduce_sum(tf.math.pow(user_embed - pos_user_embed, 2))
            user_rep = tf.expand_dims(user_rep, 1)


            # ================================================================
            pos_items_neighs = tf.reshape(uniform_sample(self.pos_item_id, self.v_v_graph_ph, 1), [-1, ])
            all_pos_item_rep, all_pos_item_embed, _, _ = self.graphconv('pos_item_gcn',
                                                                  tf.concat([self.pos_item_id, pos_items_neighs], 0),
                                                                  self.user_embeddings, self.item_embeddings,
                                                                  'pos_item')

            all_pos_item_rep = tf.concat([all_pos_item_rep, all_pos_item_embed], 1)
            pos_item_rep, pos_item_neigh_rep = tf.split(all_pos_item_rep, [batch_size, batch_size])
            pos_item_embed, pos_item_neigh_embed = tf.split(all_pos_item_embed, [batch_size, batch_size])

            pos_item_item_dist = tf.reduce_sum(tf.math.pow(pos_item_rep - pos_item_neigh_rep, 2)) \
                                 + tf.reduce_sum(tf.math.pow(pos_item_embed - pos_item_neigh_embed, 2))

            # ============================
            neg_item_rep, neg_item_embed, _, _ = self.graphconv('neg_item_gcn', self.neg_item_id, self.user_embeddings,
                                                          self.item_embeddings, 'neg_item')

            neg_item_rep = tf.concat([neg_item_rep, neg_item_embed], 2)
            item_rep = tf.concat([tf.expand_dims(pos_item_rep, 1), neg_item_rep], 1)
            item_embed = tf.concat([tf.expand_dims(pos_item_embed, 1), neg_item_embed], 1)

            # self.item_medium_input = tf.concat([pos_item_layer_input, neg_item_layer_input], 1)

            # === BPR loss
            pos_rating = tf.reduce_sum(tf.multiply(tf.squeeze(user_rep, 1), pos_item_rep), 1)
            pos_rating = tf.expand_dims(pos_rating, 1)
            pos_rating = tf.tile(pos_rating, [1, self.neg_item_num])
            pos_rating = tf.reshape(pos_rating, [tf.shape(pos_rating)[0] * self.neg_item_num, 1])

            batch_neg_item_embedding = tf.transpose(neg_item_rep, [0, 2, 1])
            neg_rating = tf.matmul(user_rep, batch_neg_item_embedding)
            neg_rating = tf.squeeze(neg_rating, 1)
            neg_rating = tf.reshape(neg_rating, [tf.shape(neg_rating)[0] * self.neg_item_num, 1])

            bpr_loss = pos_rating - neg_rating
            bpr_loss = tf.nn.sigmoid(bpr_loss)
            bpr_loss = -tf.math.log(bpr_loss)
            bpr_loss = tf.reduce_sum(bpr_loss)

            if self.mse_reg_flag:
                old_u_emb_matrix = tf.gather(self.old_user_embedding, self.u_id)
                old_pos_i_emb_matrix = tf.gather(self.old_item_embedding, self.pos_item_id)
                old_neg_i_emb_matrix = tf.gather(self.old_item_embedding, self.neg_item_id)
                old_i_emb_matrix = tf.concat([tf.expand_dims(old_pos_i_emb_matrix, 1), old_neg_i_emb_matrix], 1)

                user_reg = tf.reduce_sum(tf.squared_difference(user_embed, old_u_emb_matrix), 1) * self.u_mse_coef
                item_reg = tf.reshape(tf.reduce_sum(tf.squared_difference(item_embed, old_i_emb_matrix), 2),
                                      [-1]) * self.i_mse_coef

                reduced_user_reg = tf.reduce_sum(user_reg)
                reduced_item_reg = tf.reduce_sum(item_reg)

                bpr_loss += self.inc_reg[0] * (reduced_user_reg + reduced_item_reg)

                u_diff_mat = tf.squared_difference(self.user_embeddings[:self.old_num_user],
                                                   self.old_user_embedding[:self.old_num_user])
                all_user_reg = tf.reduce_sum(u_diff_mat, 1) * self.u_mse_coef_dist_score
                i_diff_mat = tf.squared_difference(self.item_embeddings[:self.old_num_item],
                                                   self.old_item_embedding[:self.old_num_item])
                all_item_reg = tf.reduce_sum(i_diff_mat, 1) * self.i_mse_coef_dist_score

                self.mse_user_reg = all_user_reg
                self.mse_item_reg = all_item_reg

            if self.inc_reg[1] > 0:
                if self.local_distill_mode == 'LSP_s':
                    # ============== LSP_s ============
                    u_i_emb = tf.nn.embedding_lookup(self.item_embeddings[:self.old_num_item],
                                                     self.old_user_bl_idx)  # u * 10 * 128
                    i_u_emb = tf.nn.embedding_lookup(self.user_embeddings[:self.old_num_user],
                                                     self.old_item_bl_idx)  # i * 10 * 128

                    u_i_prob = tf.nn.softmax(
                        tf.reduce_sum(tf.multiply(tf.expand_dims(self.user_embeddings[:self.old_num_user], 1), u_i_emb),
                                      axis=2), axis=1)  # u * 10
                    i_u_prob = tf.nn.softmax(
                        tf.reduce_sum(tf.multiply(tf.expand_dims(self.item_embeddings[:self.old_num_item], 1), i_u_emb),
                                      axis=2), axis=1)  # i * 10

                    u_ia_kl = tf.reduce_mean(
                        tf.reduce_sum(tf.multiply(u_i_prob, tf.log(tf.divide(u_i_prob, self.old_user_bl_ls))), 1))
                    i_ua_kl = tf.reduce_mean(
                        tf.reduce_sum(tf.multiply(i_u_prob, tf.log(tf.divide(i_u_prob, self.old_item_bl_ls))), 1))

                    self.user_reg = self.inc_reg[1] * u_ia_kl
                    self.item_reg = self.inc_reg[1] * i_ua_kl
                    bpr_loss += self.inc_reg[1] * (u_ia_kl + i_ua_kl)

                elif self.local_distill_mode == 'local_anchor':
                    # ============== local anchor distillation -=================
                    u_1hop_means = tf.nn.embedding_lookup_sparse(self.item_embeddings, self.old_u_i_adj_mat, None,
                                                                 combiner="mean")  # u * 128
                    u_i_prod_new = tf.reduce_sum(u_1hop_means * self.user_embeddings[:self.old_num_user],
                                                 axis=1)  # u * 1
                    # u_i_prod_new = tf.norm(u_1hop_means - self.user_embeddings[:self.old_num_user], axis=1) # u * 1
                    loss_ui = tf.reduce_mean(tf.squared_difference(u_i_prod_new, self.ui_dist))

                    i_1hop_means = tf.nn.embedding_lookup_sparse(self.user_embeddings, self.old_i_u_adj_mat, None,
                                                                 combiner="mean")
                    i_u_prod_new = tf.reduce_sum(i_1hop_means * self.item_embeddings[:self.old_num_item], axis=1)
                    # i_u_prod_new = tf.norm(i_1hop_means - self.item_embeddings[:self.old_num_item], axis=1)
                    loss_iu = tf.reduce_mean(tf.squared_difference(i_u_prod_new, self.iu_dist))

                    self.user_reg = self.inc_reg[1] * loss_ui
                    self.item_reg = self.inc_reg[1] * loss_iu
                    bpr_loss += self.inc_reg[1] * (loss_ui + loss_iu)


            # ========= k clusters ============
            if self.global_distillation_flag:

                u_anchors = tf.nn.embedding_lookup_sparse(self.user_embeddings[:self.old_num_user],
                                                          self.old_u_cluster_mat, None, combiner="mean")  # k * 128
                i_anchors = tf.nn.embedding_lookup_sparse(self.item_embeddings[:self.old_num_item],
                                                          self.old_i_cluster_mat, None, combiner="mean")  # k * 128

                u_u_a_gs_matrix = tf.nn.softmax(
                    tf.matmul(self.user_embeddings[:self.old_num_user], tf.transpose(u_anchors)) / self.tau, axis=1)
                u_i_a_gs_matrix = tf.nn.softmax(
                    tf.matmul(self.user_embeddings[:self.old_num_user], tf.transpose(i_anchors)) / self.tau, axis=1)

                u_ua_kl = tf.reduce_mean(tf.reduce_sum(tf.multiply(u_u_a_gs_matrix, tf.log(
                    tf.divide(u_u_a_gs_matrix, self.old_user_gs[:, :self.k_centroids[0]]))), 1))
                u_ia_kl = tf.reduce_mean(tf.reduce_sum(tf.multiply(u_i_a_gs_matrix, tf.log(
                    tf.divide(u_i_a_gs_matrix, self.old_user_gs[:, self.k_centroids[0]:]))), 1))

                i_u_a_gs_matrix = tf.nn.softmax(
                    tf.matmul(self.item_embeddings[:self.old_num_item], tf.transpose(u_anchors)) / self.tau, axis=1)
                i_i_a_gs_matrix = tf.nn.softmax(
                    tf.matmul(self.item_embeddings[:self.old_num_item], tf.transpose(i_anchors)) / self.tau, axis=1)

                i_ua_kl = tf.reduce_mean(tf.reduce_sum(tf.multiply(i_u_a_gs_matrix, tf.log(
                    tf.divide(i_u_a_gs_matrix, self.old_item_gs[:, :self.k_centroids[0]]))), 1))
                i_ia_kl = tf.reduce_mean(tf.reduce_sum(tf.multiply(i_i_a_gs_matrix, tf.log(
                    tf.divide(i_i_a_gs_matrix, self.old_item_gs[:, self.k_centroids[0]:]))), 1))

                self.user_reg = self.inc_reg[2] * (u_ua_kl + i_ia_kl)
                self.item_reg = self.inc_reg[2] * (u_ia_kl + i_ua_kl)
                bpr_loss += self.inc_reg[2] * (u_ia_kl + i_ua_kl) + self.inc_reg[2] * (u_ua_kl + i_ia_kl)



            reg_loss = self.l2_embed * (tf.nn.l2_loss(user_rep) + tf.nn.l2_loss(item_rep))







        return bpr_loss, reg_loss, 0.002 * (user_user_distance + pos_item_item_dist)




    def graphconv(self, scope, central_ids, user_embeddings, item_embeddings, tag):
        with tf.compat.v1.variable_scope(scope):
            if tag == 'user':
                agg_funcs = self.user_agg_funcs
                self_agg_funcs = self.u_u_agg_func
                self_embeddings = user_embeddings
                neigh_embeddings = item_embeddings
                self_adj_info_ph = self.u_adj_info_ph
                neigh_adj_info_ph = self.v_adj_info_ph
                self_graph_info = self.u_u_graph_ph
                embed = tf.gather(self_embeddings, central_ids)
            else:
                agg_funcs = self.item_agg_funcs
                self_agg_funcs = self.v_v_agg_func
                self_embeddings = item_embeddings
                neigh_embeddings = user_embeddings
                self_adj_info_ph = self.v_adj_info_ph
                neigh_adj_info_ph = self.u_adj_info_ph
                self_graph_info = self.v_v_graph_ph
                embed = tf.gather(self_embeddings, central_ids)
                if tag != 'pos_item':
                    central_ids = tf.reshape(central_ids,
                                             [tf.shape(central_ids)[0] * central_ids.get_shape()[1]])


            central_ids = tf.cast(central_ids, tf.int32)
            unique_nodes, unique_idx = tf.unique(central_ids)

            self_id_at_layers = [unique_nodes]
            neigh_id_at_layers = []

            # == Bipartite GCN # =================================================================================

            for i in range(self.num_layers - 1):
                neigh_id_at_layer_i = uniform_sample(self_id_at_layers[i],
                                                     self_adj_info_ph if i % 2 == 0 else neigh_adj_info_ph,
                                                     self.num_samples[i])
                neigh_id_at_layers.append(neigh_id_at_layer_i)
                if i + 1 < self.num_layers - 1:
                    self_id_at_layers.append(tf.reshape(neigh_id_at_layers[i - 1], [-1]))

            self_matrix_at_layers = [tf.gather(self_embeddings if i % 2 == 0 else neigh_embeddings,
                                               self_id_at_layers[i]) for i in range(self.num_layers - 1)]

            neigh_matrix_at_layers = [tf.gather(neigh_embeddings if i % 2 == 0 else self_embeddings,
                                                neigh_id_at_layers[i]) for i in range(self.num_layers - 1)]

            self_outputs = [self_matrix_at_layers[0]]
            for i in range(self.num_layers - 2, -1, -1):
                output1 = agg_funcs[i](self_matrix_at_layers[i], neigh_matrix_at_layers[i])

                if i > 0:
                    neigh_matrix_at_layers[i - 1] = tf.reshape(output1, [tf.shape(self_matrix_at_layers[i - 1])[0],
                                                                         self.num_samples[i - 1], -1])

                self_outputs.append(output1)


                # == MGE layer # =====================================================================================
                self_graph_neighs = uniform_sample(unique_nodes, self_graph_info, self.num_self_neigh)
                self_graph_neighs = tf.cast(self_graph_neighs, tf.int32)
                # self_graph_neighs = tf.nn.embedding_lookup(self.u_u_graph_ph, unique_nodes)
                self_graph_neighs_matrix = tf.gather(self_embeddings, self_graph_neighs)
                output2 = self_agg_funcs(self_matrix_at_layers[0], self_graph_neighs_matrix)
                # ====================================================================================================

                output1 = tf.nn.embedding_lookup(output1, unique_idx)
                output2 = tf.nn.embedding_lookup(output2, unique_idx)
                output = tf.concat([output1, output2], 1)
                output = tf.nn.tanh(output)
                if tag != "pos_item" and tag != "user":
                    output_shape = output.get_shape().as_list()
                    output = tf.reshape(output, [tf.shape(embed)[0], self.neg_item_num, output_shape[-1]])
            return output, embed, self_outputs, self_id_at_layers



    def predict(self, batch_user_idx, item_idx, test_n_user, new_user_embedding=None, new_item_embedding=None):

        # === process item rep
        # # get item_rep for existing items in the training set
        n_item = len(item_idx)
        item_idx = tf.convert_to_tensor(item_idx, dtype=tf.int32)

        # === process new item rep
        if test_n_user > self.num_user:
            assert new_user_embedding.shape[0] == test_n_user - self.num_user
            user_embed = tf.concat([self.user_embeddings, new_user_embedding], axis=0)
        else:
            user_embed = self.user_embeddings
        # === process new item rep
        if n_item > self.num_item:
            assert new_item_embedding.shape[0] == n_item - self.num_item
            item_embed = tf.concat([self.item_embeddings, new_item_embedding], axis=0)
        else:
            item_embed = self.item_embeddings

        batch_user_idx = tf.convert_to_tensor(batch_user_idx)
        batch_user_idx = tf.cast(batch_user_idx, tf.int32)
        batch_user_rep, batch_user_embed, _, _ = self.graphconv('user_gcn', batch_user_idx, user_embed, item_embed, 'user')
        batch_user_rep = tf.concat([batch_user_rep, batch_user_embed], 1)

        batch_item_rep, batch_item_embed, _, _ = self.graphconv('item_gcn', item_idx, user_embed, item_embed, 'pos_item')
        batch_item_rep = tf.concat([batch_item_rep, batch_item_embed], 1)

        rating_score = tf.matmul(batch_user_rep, batch_item_rep, transpose_b=True)

        return rating_score, batch_user_rep, batch_item_rep