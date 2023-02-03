from multiprocessing import Pool, Process
import random
import time
import datetime
import subprocess
import os
import random

PREFIX = 'python3 -u run_mgccf.py -seed 1000'
DATASET = '-d gowalla_60 -e 100 -train_mode sep -data_split [0.6,3,0.1] -lr 5e-4 -patience 2'
INC = '-log_folder gowalla_60_kd_hp_tuning -block 1 -n_inc 1 -load_cp b0_2000e'
GRAPHSAIL = ''
# GRAPHSAIL = '-global_k 10,10 -global_tau 1,1'

# PREFIX = 'python3 -u run_mgccf.py -seed 1000'
# DATASET = '-d tb2014_60 -e 100 -train_mode sep -data_split [0.6,3,0.1] -lr 5e-4 -patience 2'
# INC = '-log_folder tb2014_60_kd_hp_tuning -block 1 -n_inc 1 -load_cp b0_2000e'
# GRAPHSAIL = ''

# PREFIX = 'python3 -u run_mgccf.py -seed 1000'
# DATASET = '-d tb2015_60 -e 100 -train_mode sep -data_split [0.6,3,0.1] -lr 5e-4 -patience 2'
# INC = '-log_folder tb2015_60_kd_hp_tuning -block 1 -n_inc 1 -load_cp b0_2000e'
# GRAPHSAIL = ''

# PREFIX = 'python3 -u run_mgccf.py -seed 1000'
# DATASET = '-d yelp_5yrs_60 -e 100 -train_mode sep -data_split [0.6,3,0.1] -lr 5e-4 -patience 2'
# INC = '-log_folder yelp_5yrs_60_kd -block 1 -n_inc 3 -load_cp b0_2000e'
# GRAPHSAIL = ''

# PREFIX = 'python3 -u run_mgccf.py -seed 1000'
# DATASET = '-d almm2017_60 -e 100 -train_mode sep -data_split [0.6,3,0.1] -lr 5e-4 -patience 2'
# INC = '-log_folder almm2017_60_full -block 1 -n_inc 3 -load_cp b0_2000e'
# GRAPHSAIL = ''


BASE_COMMAND = " ".join([PREFIX, DATASET, INC, GRAPHSAIL])

args = {
        # '': ['']}
        # '-rs':['full'],
        # '-union_mode':['snu'],
        # '-replay_ratio':[0.01],
        # '-sliding_ratio':[0.6],
        # '-sampling_mode':['uniform', 'inverse_deg', 'prop_deg', 'latest', 'boosting_inner_product', 'boosting_wasserstein']}
        # '-sampling_mode':['inverse_deg'],
        # '-new_node_init':['2hop_mean']}
        # '-adaptive_reservoir':['interest_shift']}
        # '-inc_agg': [1]}

        '-mse':[0],
        '-local_distill': [1,10,1000],
        '-global_distill': [0],
        '-global_k': ['10,10'],
        '-global_tau': ['1,1']}
# args = {}

def construct_command():
    commands = []
    for k in args.keys():
        if len(commands) == 0:
            for option in args[k]:
                arg = " ".join([k, str(option)])
                commands.append(arg)
        else: 
            cmds = []
            for cmd in commands:
                for option in args[k]:
                    arg = " ".join([cmd, k, str(option)])
                    cmds.append(arg)
            commands = cmds

    for i in range(len(commands)):
        log_arg = '-'.join(commands[i].split(' ')[1::2])
        commands[i] = " ".join([commands[i], '-log', log_arg])
        commands[i] = " ".join([BASE_COMMAND, commands[i]])
    return commands

def exec_command(cmd):
    if os.system(cmd) != 0:
        with open('batch_run_log.txt', 'a') as f:
            f.write(cmd + ',' + (datetime.datetime.utcnow() - datetime.timedelta(hours=4)).strftime("%b_%d_%H_%M_%S") + '\n')

if __name__ == '__main__':

    # commands = [BASE_COMMAND+' -log graphsail-cikm_only'] * 5
    
    commands = [cmd for cmd in construct_command() for i in range(5)] 

    for i in range(len(commands)):
    #     # commands[i] = commands[i] + '-graphsail_cikm-' + str(i%5+1)
    #     # commands[i] = commands[i] + '-flatten_5-' + str(i%5+1)
        commands[i] = commands[i] + '-kd_only'# + str(i%5+1)

    # fix kmeans OMP issue
    for i in range(len(commands)):
        commands[i] = commands[i].replace('[', '')
        commands[i] = commands[i].replace(']', '')

    # print(len(commands))
    # print(commands)
    # assert False

    a = [" ".join(['taskset -c 0-20', cmd, '-de 0']) for cmd in commands[0:4]]
    b = [" ".join(['taskset -c 0-20', cmd, '-de 1']) for cmd in commands[4:8]]
    c = [" ".join(['taskset -c 20-40', cmd, '-de 2']) for cmd in commands[8:12]]
    d = [" ".join(['taskset -c 20-40', cmd, '-de 3']) for cmd in commands[12:16]]

    pools = []
    try:
        for cmds in [a,b,c,d]:
            pool = Pool(2) #use all available cores, otherwise specify the number you want as an argument
            pools.append(pool)

            for cmd in cmds:
                pool.apply_async(exec_command, args=(cmd,))
                time.sleep(3)
            pool.close()
        for pool in pools:
            pool.join()
    except:
        for pool in pools:
            pool.terminate()
            pool.join()