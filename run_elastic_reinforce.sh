#!/bin/bash
ROOT_EXP_DIR='./exps'
CKPT_DIR="${ROOT_EXP_DIR}/zzz_ckpt"
DECAY=0.995
RESET=5000
EMARESET=5000

EXP_DIR="${ROOT_EXP_DIR}/elastic_reinforce_${DECAY}_${RESET}_${EMARESET}"
LOG_DIR=${EXP_DIR}

python iterated_learning.py -method reinforce -ckpt_dir ${CKPT_DIR} -generation_steps 90000 \
    -logdir ${LOG_DIR} -s_lr 0.001 -l_lr 0.001 -steps 25000 -batch_size 64 \
    -log_steps 500 --use_ema --ema_decay ${DECAY} --reset_steps ${RESET} \
    --reset_ema --ema_reset_steps ${EMARESET}
