#!/bin/bash
K1=2000
K2S=400
K2L=400
ROOT_EXP_DIR='./exps'
CKPT_DIR="${ROOT_EXP_DIR}/zzz_ckpt"
EXP_DIR="${ROOT_EXP_DIR}/reinforce_25k_iter_generation${K1}_transmission${K2S}-${K2L}_same_opt"
LOG_DIR=${EXP_DIR}

python iterated_learning.py -method reinforce -ckpt_dir ${CKPT_DIR} -generation_steps ${K1} \
    -s_transmission_steps ${K2S} -l_transmission_steps ${K2L} -logdir ${LOG_DIR} \
    -s_lr 0.001 -l_lr 0.001 -steps 25000 -batch_size 64 \
    -log_steps 500 -same_opt
