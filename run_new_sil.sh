#!/bin/bash
K1=2000
K2S=200
K2L=200
ROOT_EXP_DIR='./exps'
CKPT_DIR="${ROOT_EXP_DIR}/new_ckpt"
GUMBEL_TEMP=10
EXP_DIR="${ROOT_EXP_DIR}/new_iter_generation${K1}_transmission${K2S}-${K2L}_same_opt"
LOG_DIR=${EXP_DIR}

python iterated_learning.py -method gumbel -ckpt_dir ${CKPT_DIR} -generation_steps ${K1} \
          -s_transmission_steps ${K2S} -l_transmission_steps ${K2L} -logdir ${LOG_DIR} \
        	-temperature ${GUMBEL_TEMP} -decay_rate 1. -s_lr 0.0001 -l_lr 0.0001 -steps 50000 -batch_size 50 \
        	-log_steps 500 -same_opt
