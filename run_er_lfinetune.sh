#!/bin/bash
K1=2000
K2=200
ROOT_EXP_DIR='./exps'
CKPT_DIR="${ROOT_EXP_DIR}/zzz_ckpt"
GUMBEL_TEMP=10
DECAY=0.999
RESET=1500

EXP_DIR="${ROOT_EXP_DIR}/er_lfinetune_${DECAY}_${RESET}"
LOG_DIR=${EXP_DIR}

python iterated_learning.py -method gumbel -ckpt_dir ${CKPT_DIR} -generation_steps 50000 \
           -logdir ${LOG_DIR} \
        	-temperature ${GUMBEL_TEMP} -decay_rate 1. -s_lr 0.0001 -l_lr 0.0001 -steps 50000 -batch_size 50 \
        	-log_steps 500 --use_ema --ema_decay ${DECAY} --reset_steps ${RESET} --reset_ema --reset_listener "finetune" -same_opt