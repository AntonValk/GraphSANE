#! /bin/bash

# sed -i 's/\r//g' script.sh

python3 -u run_mgccf_negative.py -d taobao1 -de 0 -e 100 -block 1 -n_inc 3 \
         -train_mode sep -log_folder taobao14_logs \
         -log test_log_1 -load_cp base_taobao14 -layer_wise 0 -con_negative 5 -con_positive 15 \
         -global_k 10,10 -mse 100 -local_distill 1e4 -global_distill 1e4 \
         -global_tau 1 -patience 2 -lr 5e-4 \
         -load_save_path_prefix /media/data/yuening/graph_incremental_logs/log-files/reservoir2/ \
         -min_epoch 5 -log_files "/media/data/yuening/logs/graphsail_base" -adaptive_mode=''  \
