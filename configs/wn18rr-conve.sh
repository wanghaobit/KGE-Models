#!/usr/bin/env bash

data_dir="data/WN18RR"
emb_model="ConvE"
add_reversed_training_edges="True"

entity_dim=200
relation_dim=200
num_epochs=1000
num_peek_epochs=1
num_wait_epochs=100
batch_size=1024
train_batch_size=1024
dev_batch_size=128
learning_rate=0.00125

emb_dropout_rate=0.1
input_dropout_rate=0.1
hidden_dropout_rate=0.3
feature_dropout_rate=0.2
num_out_channels_flag=32
kernel_size=3

model_dir="save_model/wn18rr/conve"
