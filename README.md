# End-2-end Incremental Graph-based Recommender System

## Environment requirement
  See requirement.txt
  
## Log file directory

Will clarify
## Data Processing
  We introduced a new way to split val/test set. Under the `-data_split [0.6, 3, 0.1]` default setting, we will end up having four blocks. 
  The base block contains the first 60% of data, and 4 incremental blocks each contains the next 10% of data. Taking incremental block_1 as 
  the training set for example, the val set is the first half of the incremental block_2, and the test set is the second half of the incremental 
  block_2. Note the difference comparing to the experiment setting in the CIKM repo is that the val/test set is also split by absolute time.
### Data blocks and data split

When the data is processed by the *Data* class, it can be accessed through the dict variable *self.blocks* of the *Data* class.
*self.blocks* has the following structure:  
*self.blocks* = {0: {similar to CIKM version for base block},  
      1:{'train': training_set of the current block,  
       'n_user_train': # of users in the training set of current block,    
       'n_item_train': # of items in the training set of current block,   
       'val': val_set of the current block (by default the following 5% of data after training set),  
       'n_user_val': # of users in the val set of current block,  
       'n_item_val', # of items in the val set of current block,  
       'test': test_set of the current block (by default the following 5% of data after val set),,  
       'n_user_test': # of users in the test set of current block,  
       'n_item_test': # of items in the test set of current block,  
       'latest_reservoir': some data before the training set, size is determined by `-replay_ratio` (only used if `-sampling_mode lastest` is set),   
       'acc_train': accumulated training_data from block 0,  
       'acc_train_plus_val': accumulated training_data from block 0 + val_set,  
       'acc_train_plus_val_test': accumulated training_data from block 0 + val_set + test_set,  
       'train_matrix': the adj matrix of accumulated training_data (csr format),  
       'val_matrix': the adj matrix of acc_train_plus_val (csr format),  
       'test_matrix': the adj matrix of acc_train_plus_val_test (csr format),  
       'sliding_lists': [optional] this only exists if `-rs sliding` is set, it contains data of the sliding window,  
       'sliding_matrix':[optional] this only exists if `-rs sliding` is set, it contains data of the sliding window in adj matrix (csr format)}, ...}

## Command examples and explanation

* **Training the first block**  
  > `python3 -u run_mgccf.py -d Gowalla-20 -de 0 -e 100 -block 0 -train_mode sep -log_folder test -log test_log_0 -save_cp b0_100e -b_eval 1e5`

* **Training incremental blocks**  

 > fine-tune:  
  `python3 -u run_mgccf.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 1 -train_mode sep -log_folder test -log test_log_1 -load_cp b0_100e`  


  > fine-tune with weighted mse regularization:   
  `python3 -u run_mgccf.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 1 -train_mode sep -log_folder test -log test_log_1 -load_cp b0_100e -mse 100`  
  
  > fine-tune with local distillation:    
  `python3 -u run_mgccf.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 1 -train_mode sep -log_folder test -log test_log_1 -load_cp b0_100e -local_distill 1e7`  
  
  > fine-tune with global distillation:    
  `python3 -u run_mgccf.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 1 -train_mode sep -log_folder test -log test_log_1 -load_cp b0_100e -global_distill 1e4 -global_k 10,10 -global_tau 1`  
  
  > example command for reported gowalla experiment in GraphSAIL paper:    
  `python3 -u run_mgccf.py -d Gowalla-10 -de 0 -e 25 -block 1 -n_inc 4 -train_mode sep -log_folder test -log test_log_1 -load_cp b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4`
  
  Following is the new experiments for layer-wise contrastive loss.
  > layer-wise:   
  `python3 -u run_mgccf.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 1 -train_mode sep -log_folder test -log test_log_1 -load_cp b0_100e -layer_wise 1 -lambda_contrastive 5e2` 
  
 > SGCT:
 `python3 -u run_mgccf.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 1 -train_mode sep -log_folder test -log test_log_1 -load_cp b0_100e -layer_wise 0 -contrastive_mode 'Single' -lambda_contrastive 5e2` 

> MGCT:
 `python3 -u run_mgccf.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 1 -train_mode sep -log_folder test -log test_log_1 -load_cp b0_100e -layer_wise 0 -contrastive_mode 'Multi' -lambda_contrastive 5e2`

> LWC-KD:
 `python3 -u run_mgccf.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 1 -train_mode sep -log_folder test -log test_log_1 -load_cp b0_100e -layer_wise 1 -contrastive_mode 'Multi' -lambda_contrastive 5e2`  
  
  
  
 Following is the new experiments commands including reservoir.
  > **fine-tune with random sampled u-i pair from entire history**  
  `python3 -u run_mgccf.py -d gowalla_60 -e 20 -train_mode sep -lr 5e-4 -patience 2 -log_folder test -block 1 -n_inc 3 -log test -load_cp b0_test -rs full -union_mode snu -replay_ratio 0.2 -sampling_mode uniform -de 0`  
       `-rs`: controls the scope of reservoir.   
        - `full`: sample the subset of old data from entire historical data.   
        - `sliding`: sample the subset of old data from a sliding window. When this option is used, `-sliding_ratio` is used to determine the size of sliding window.   
        - `reservoir_sampling` sample the subset of old data from a reservoir, which is constructed using [reservoir_sampling](https://en.wikipedia.org/wiki/Reservoir_sampling). [GAG](https://arxiv.org/pdf/2007.02747.pdf) used reservoir sampling in their reservoir method(potential baseline). When this option is used, `-sliding_ratio` is used to determine the size of reservoir.    
       `-union_mode`: `snu` means first sample the subset of the reservoir, and then union the current block training data and the sampled subset; `uns` means means first union the current block training data and the reservoir, then sample from the union.  
       `-replay_ratio`: the size of the sampled subset. `0.2` means sample a subset with size of 20% of the entire dataset. By default, each incremental block has size of 10% of entire dataset, setting `-replay_ratio 0.2` means the old data has the size two times of the the size of training data of current block.  
       `-sampling_mode`: the sampling strategy used to sample from reservoir. For detailed explaination of each strategy, refer to the overleaf document.  
        - `uniform`: random sampling  
        - `prop_deg`: weighted sampling with higher weight for edges connects two high-degree nodes, vice versa  
        - `inverse_deg`: weighted sampling with higher weight for edges connects two low-degree nodes, vice versa  
        - `adp_inverse_deg`: adapt `inverse_deg` strategy a little bit according to the change in node degree across incremental blocks  
        - `boosting_inner_product`: do a pre-training inference at the begining of each block, weighted sampling with higher weight for positive u-i pairs with lower prediction score  
        - `boosting_wasserstein`: do a pre-training inference at the begining of each block, weighted sampling with higher weight for positive u-i pairs whose user's prediction has a larger wasserstein distance to the ground truth  
        - `mse_distill_score`: weighted sampling with higher weight for u-i pairs with a larger mse_distillation_score from the incremental training of last block. Note current implementation uses a top-k selection, i.e. u-i pairs with largest K mse_distillation_score is selected.     
        - `item_embedding`: select u-i pairs according to a hard threshold. Refer to the overleaf document for detail.   
        - `latest`: do not sample, directly use the most recent historical data as the subset. With `-replay_ratio 0.2`, it means select the most recent 20% data before the current training block.  
       `-lr 5e-4`: for incremental blocks, we usually use a smaller lr than the full-batch training  
       `-block 1 -n_inc 3`: this indicates the program will train for 3 consecutive blocks starting from block_1
      
    > **fine-tune with using inverse-deg sampling from a sliding window**  
    `python3 -u run_mgccf.py -d gowalla_60 -e 20 -train_mode sep -lr 5e-4 -patience 2 -log_folder test -block 1 -n_inc 3 -log test -load_cp b0_test -rs sliding -union_mode snu -replay_ratio 0.2 -sampling_mode inverse_deg -sliding_ratio 0.6 -de 0`  
       `-sliding_ratio`: controls the size of the sliding window. Here it means instead of sampling from the entire history, it samples from the most recent 60% of data before the current block. By default, 60% is the size of the base block(block_0), it means for the block_1, it still samples from the entire history, which is the first 60% of the entire dataset. Starting from block_2, it samples from the 60% of data before block_2, which means it do not sample from the very first 10% of the entire dataset.    
       `-inc_agg`: default value is `0`. In our implementation, although we only use the new u-i pair to optimize the BPR loss, we use all historical data to build the graph for the convolution operation(neighbour aggregation). If set to `1`, it uses the sliding window to build graph, meaning that it discards some earliest edges from the graph which is used for the convolution operation(neighbour aggregation).  

    > **fine-tune with using inverse-deg sampling from all history plus some knowledge distillation**  
    `python3 -u run_mgccf.py -d gowalla_60 -e 20 -train_mode sep -lr 5e-4 -patience 2 -log_folder test -block 1 -n_inc 3 -log test -load_cp b0_test -rs full -union_mode snu -replay_ratio 0.2 -sampling_mode inverse_deg -mse 100 -new_node_init 2hop_mean -de 0 `  
       `-mse 100`: this indicates it also applies some MSE distillation together with the usage of reservoir. The reservoir method and KD can be used together, simply add the corresponding arugments used in GraphSAIL.  
       `-new_node_init`: this controls the way to initialize new nodes. For each incremental block, some new nodes will be introduced, the default value `` means to initialize these new node randomly; in this example, `2hop_mean` means to initialize the embedding of these new nodes by using the mean embedding of their 2-hop neighbours.  
      

# GraphSANE
ALL GRAPH_SAIL
taskset -c "0-11" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files neg_graphsail_gowalla20 -load_cp gowalla_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4 --neg_res



## LWCKD

taskset -c "0-11" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files neg_lwc_gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res

taskset -c "12-23" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 1 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files neg_lwc_gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res

taskset -c "0-11" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files lwc_gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5

-layer_wise 1 -contrastive_mode 'Multi' -lambda_contrastive 5e2

taskset -c "0-11" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files ft_gowalla20 -load_cp gowalla_b0_100e -min_epoch 5 -patience 5

## ALL SGCT

### SANE

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files neg_sgct_gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 --neg_res

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d taobao1 -de 1 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao1 -log taobao1 -log_files neg_sgct_taobao1 -load_cp taobao1_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 --neg_res

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d taobao2_buy -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao2_buy -log taobao2_buy -log_files neg_sgct_taobao2_buy -load_cp taobao2_buy_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 --neg_res

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d yelp_5yrs_60 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testyelp_5yrs_60 -log yelp_5yrs_60 -log_files neg_sgct_yelp_5yrs_60 -load_cp yelp_5yrs_60_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,100 --neg_res

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d netflix -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testnetflix -log netflix -log_files neg_sgct_netflix -load_cp netflix_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,100 --neg_res

### NO-SANE

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,100

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d taobao1 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao1 -log taobao1 -log_files sgct_taobao1 -load_cp taobao1_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,100 --neg_res

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d taobao2_buy -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao2_buy -log taobao2_buy -log_files sgct_taobao2_buy -load_cp taobao2_buy_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,100 

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d yelp_5yrs_60 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testyelp_5yrs_60 -log yelp_5yrs_60 -log_files sgct_yelp_5yrs_60 -load_cp yelp_5yrs_60_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,100

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d netflix -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testnetflix -log netflix -log_files sgct_netflix -load_cp netflix_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,100




## ALL LWC-KD

### NO SANE
```
taskset -c "13-24" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files lwc_gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,100

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d taobao1 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao1 -log taobao1 -log_files lwc_taobao1 -load_cp taobao1_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,100

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d taobao2_buy -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao2_buy -log taobao2_buy -log_files lwc_taobao2_buy -load_cp taobao2_buy_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,100

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d yelp_5yrs_60 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testyelp_5yrs_60 -log yelp_5yrs_60 -log_files lwc_yelp_5yrs_60 -load_cp yelp_5yrs_60_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,100

taskset -c "13-24" python3 -u run_mgccf_negative3.py -d netflix -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testnetflix -log netflix -log_files lwc_netflix -load_cp netflix_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 2 -lr 5e-4 -con_ratios 1.5,1,1.5,1,1,1,1 -layer_wise 0 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,100
```


## SANE
```
taskset -c "0-4" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files neg_graphsail_gowalla20 -load_cp gowalla_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4 --neg_res

taskset -c "5-9" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 1 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files neg_sgct_gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01  -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res

taskset -c "10-14" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 2 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files neg_lwc_gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01  -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res

taskset -c "15-19" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 3 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files graphsail_gowalla20 -load_cp gowalla_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4

taskset -c "20-24" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 4 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files sgct_gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 

taskset -c "25-29" python3 -u run_mgccf_negative3.py -d Gowalla-20 -de 5 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testgowalla20 -log gowalla20 -log_files lwc_gowalla20 -load_cp gowalla_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 
```

# YELP
```
taskset -c "0-4" python3 -u run_mgccf_negative3.py -d yelp_5yrs_60 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testyelp_5yrs_60 -log yelp_5yrs_60 -log_files neg_graphsail_yelp_5yrs_60 -load_cp yelp_5yrs_60_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4 --neg_res

taskset -c "5-9" python3 -u run_mgccf_negative3.py -d yelp_5yrs_60 -de 1 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testyelp_5yrs_60 -log yelp_5yrs_60 -log_files neg_sgct_yelp_5yrs_60 -load_cp yelp_5yrs_60_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res

taskset -c "10-14" python3 -u run_mgccf_negative3.py -d yelp_5yrs_60 -de 2 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testyelp_5yrs_60 -log yelp_5yrs_60 -log_files neg_lwc_yelp_5yrs_60 -load_cp yelp_5yrs_60_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res


taskset -c "15-19" python3 -u run_mgccf_negative3.py -d yelp_5yrs_60 -de 3 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testyelp_5yrs_60 -log yelp_5yrs_60 -log_files graphsail_yelp_5yrs_60 -load_cp yelp_5yrs_60_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4

taskset -c "20-24" python3 -u run_mgccf_negative3.py -d yelp_5yrs_60 -de 4 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testyelp_5yrs_60 -log yelp_5yrs_60 -log_files sgct_yelp_5yrs_60 -load_cp yelp_5yrs_60_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 

taskset -c "25-29" python3 -u run_mgccf_negative3.py -d yelp_5yrs_60 -de 5 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testyelp_5yrs_60 -log yelp_5yrs_60 -log_files lwc_yelp_5yrs_60 -load_cp yelp_5yrs_60_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 
```



# TAOBAO1
```
taskset -c "0-4" python3 -u run_mgccf_negative3.py -d taobao1 -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao1 -log taobao1 -log_files neg_graphsail_taobao1 -load_cp taobao1_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4 --neg_res

taskset -c "5-9" python3 -u run_mgccf_negative3.py -d taobao1 -de 1 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao1 -log taobao1 -log_files neg_sgct_taobao1 -load_cp taobao1_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res

taskset -c "10-14" python3 -u run_mgccf_negative3.py -d taobao1 -de 2 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao1 -log taobao1 -log_files neg_lwc_taobao1 -load_cp taobao1_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res


taskset -c "15-19" python3 -u run_mgccf_negative3.py -d taobao1 -de 3 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao1 -log taobao1 -log_files graphsail_taobao1 -load_cp taobao1_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4

taskset -c "20-24" python3 -u run_mgccf_negative3.py -d taobao1 -de 4 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao1 -log taobao1 -log_files sgct_taobao1 -load_cp taobao1_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 

taskset -c "25-29" python3 -u run_mgccf_negative3.py -d taobao1 -de 5 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao1 -log taobao1 -log_files lwc_taobao1 -load_cp taobao1_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 
```

# TAOBAO2
```
taskset -c "0-4" python3 -u run_mgccf_negative3.py -d taobao2_buy -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao2_buy -log taobao2_buy -log_files neg_graphsail_taobao2_buy -load_cp taobao2_buy_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4 --neg_res

taskset -c "5-9" python3 -u run_mgccf_negative3.py -d taobao2_buy -de 1 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao2_buy -log taobao2_buy -log_files neg_sgct_taobao2_buy -load_cp taobao2_buy_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res

taskset -c "10-14" python3 -u run_mgccf_negative3.py -d taobao2_buy -de 0,1 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao2_buy -log taobao2_buy -log_files neg_lwc_taobao2_buy -load_cp taobao2_buy_b0_100e_mini -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res --embedded_dimension '[10, 10, 10]'


taskset -c "15-19" python3 -u run_mgccf_negative3.py -d taobao2_buy -de 3 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao2_buy -log taobao2_buy -log_files graphsail_taobao2_buy -load_cp taobao2_buy_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4

taskset -c "20-24" python3 -u run_mgccf_negative3.py -d taobao2_buy -de 4 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao2_buy -log taobao2_buy -log_files sgct_taobao2_buy -load_cp taobao2_buy_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 

taskset -c "25-29" python3 -u run_mgccf_negative3.py -d taobao2_buy -de 2,3 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testtaobao2_buy -log taobao2_buy -log_files lwc_taobao2_buy -load_cp taobao2_buy_b0_100e_mini -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 --embedded_dimension '[10, 10, 10]'
```

# NETFLIX
```
taskset -c "0-4" python3 -u run_mgccf_negative3.py -d netflix -de 0 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testnetflix -log netflix -log_files neg_graphsail_netflix -load_cp netflix_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4 --neg_res

taskset -c "5-9" python3 -u run_mgccf_negative3.py -d netflix -de 1 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testnetflix -log netflix -log_files neg_sgct_netflix -load_cp netflix_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res --embedded_dimension '[10, 10, 10]'

taskset -c "10-14" python3 -u run_mgccf_negative3.py -d netflix -de 2 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testnetflix -log netflix -log_files neg_lwc_netflix -load_cp netflix_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 --neg_res --embedded_dimension '[10, 10, 10]'


taskset -c "15-19" python3 -u run_mgccf_negative3.py -d netflix -de 3 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testnetflix -log netflix -log_files graphsail_netflix -load_cp netflix_b0_100e -mse 100 -local_distill 1e4 -global_distill 1e4 -global_k 10,10 -global_tau 1 -patience 2 -lr 5e-4

taskset -c "20-24" python3 -u run_mgccf_negative3.py -d netflix -de 4 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testnetflix -log netflix -log_files sgct_netflix -load_cp netflix_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Single' -lambda_contrastive 100,10,1 -min_epoch 5 --embedded_dimension '[10, 10, 10]'

taskset -c "25-29" python3 -u run_mgccf_negative3.py -d netflix -de 5 -e 25 -block 1 -n_inc 3 -train_mode sep -log_folder testnetflix -log netflix -log_files lwc_netflix -load_cp netflix_b0_100e -lambda_soft 0.01 -global_k 10,10 -global_tau 0.5 -patience 5 -lr 5e-4 -con_ratios 1.5,1,1.5,2,1,2,1 -layer_wise 1 -ui_con_positive 15 -contrastive_mode 'Multi' -lambda_contrastive 100,10,1 -min_epoch 5 --embedded_dimension '[10, 10, 10]'
```


# Datasets

## Standard pre-processing steps
  - Remove all nodes has degree less than 10
  - Remove all duplicated u-i pairs (some u-i pairs appear more than once with different timestamp, we keep the last one)
  - pre-processing notebook: data_utils/preprocess_dataset.ipynb. Untested code. Please read and make sure you understand before use.

## TB2014 (User Behavior Data on Taobao App)

Source of [raw data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=46).

## TB2015 (Taobao/Tmall IJCAI16 Contest)

        Only keeps buy edges.

Source of [raw data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=53).

## Alimama

        Do not keep page view(pv) edges.

Source of [raw data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649). 

## Gowalla

Source of [raw data](https://snap.stanford.edu/data/loc-Gowalla.html). 

## Yelp_5yrs

        The time range of this dataset is too long. I chopped the last five years of data for the pre-processing.

Source of [raw data](https://www.yelp.com/dataset). 

## LastFM

        This dataset is too small. In the GraphSAIL paper, I didn't remove the duplicates.

Source of [raw data](https://grouplens.org/datasets/hetrec-2011/). 

To pre-process the datasets use the pre-processincg notebook located in `data_utils/preprocess_dataset.ipynb`.


