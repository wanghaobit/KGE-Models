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
grad_norm_flag=''
if [ $grad_norm ]; then
    grad_norm_flag="--grad_norm ${grad_norm}"
fi

# dropout rates
emb_dropout_rate_flag=''
if [ $emb_dropout_rate ]; then
    emb_dropout_rate_flag="--emb_dropout_rate ${emb_dropout_rate}"
fi
input_dropout_rate_flag=''
if [ $input_dropout_rate ]; then
    input_dropout_rate_flag="--input_dropout_rate ${input_dropout_rate}"
fi
hidden_dropout_rate_flag=''
if [ $hidden_dropout_rate ]; then
    hidden_dropout_rate_flag="--hidden_dropout_rate ${hidden_dropout_rate}"
fi
hidden_dropout_rate_2_flag=''
if [ $hidden_dropout_2_rate ]; then
    hidden_dropout_rate_2_flag="--hidden_dropout_rate_2 ${hidden_dropout_rate_2}"
fi
feature_dropout_rate_flag=''
if [ $feature_dropout_rate ]; then
    feature_dropout_rate_flag="--feature_dropout_rate ${feature_dropout_rate}"
fi

# Conv
num_out_channels_flag=''
if [ $num_out_channels ]; then
    num_out_channels_flag="--num_out_channels ${num_out_channels}"
fi
kernel_size_flag=''
if [ $kernel_size ]; then
    kernel_size_flag="--kernel_size ${kernel_size}"
fi

cmd="python3 -u -m src.main \
    --data_dir $data_dir \
    $exp \
    --emb_model $emb_model \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --num_epochs $num_epochs \
    --num_peek_epochs $num_peek_epochs \
    --num_wait_epochs $num_wait_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --learning_rate $learning_rate \
    --model_dir $model_dir \
    $add_reversed_training_edges_flag \
    $grad_norm \
    $emb_dropout_rate_flag \
    $input_dropout_rate_flag \
    $hidden_dropout_rate_flag \
    $hidden_dropout_rate_2_flag \
    $feature_dropout_rate_flag \
    $num_out_channels_flag \
    $kernel_size_flag \
    --gpu $gpu \
    $ARGS"

echo "Executing $cmd"

$cmd
