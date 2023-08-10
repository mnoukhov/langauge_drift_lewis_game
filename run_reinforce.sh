#!/bin/bash
ROOT_EXP_DIR='./exps'
CKPT_DIR="${ROOT_EXP_DIR}/zzz_ckpt"
EXP_DIR="${ROOT_EXP_DIR}/reinforce"
LOG_DIR=${EXP_DIR}

python iterated_learning.py -method reinforce -ckpt_dir ${CKPT_DIR} \
    -generation_steps 90000 -logdir ${LOG_DIR} \
    -s_lr 0.0005 -l_lr 0.0005 -steps 50000 -batch_size 64 \
    -log_steps 500 -same_opt -ent_coef 0.001
