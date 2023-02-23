#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

source $1
exp=$2
gpu=$3
ARGS=${@:4}

add_reversed_training_edges_flag=''
if [[ $add_reversed_training_edges = *"True"* ]]; then
    add_reversed_training_edges_flag="--add_reversed_training_edges"
fi

cmd="python3 -u -m src.main \
    --data_dir $data_dir \
    $exp \
    --emb_model $emb_model \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --emb_dropout_rate $emb_dropout_rate \
    --num_epochs $num_epochs \
    --num_peek_epochs $num_peek_epochs \
    --num_wait_epochs $num_wait_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --learning_rate $learning_rate \
    --grad_norm $grad_norm \
    --model_dir $model_dir \
    $add_reversed_training_edges_flag \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

#$cmd
#nohup $cmd > log_wn18rr_conve.txt 2>&1 &
#nohup $cmd > log_wn18rr_acre.txt 2>&1 &
nohup $cmd > log_wn18rr_complex.txt 2>&1 &
#nohup $cmd > log_wn18rr_distmult.txt 2>&1 &
#nohup $cmd > log_wn18rr_rotate.txt 2>&1 &

#nohup $cmd > log_fb15k-237_conve.txt 2>&1 &
#nohup $cmd > log_fb15k-237_complex.txt 2>&1 &
#nohup $cmd > log_fb15k-237_distmult.txt 2>&1 &
#nohup $cmd > log_fb15k-237_acre.txt 2>&1 &
