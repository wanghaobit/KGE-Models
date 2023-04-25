#!/usr/bin/env bash

data_dir="data/WN18RR"
emb_model="SACN"
add_reversed_training_edges="True"

entity_dim=100
relation_dim=200
num_epochs=1000
num_wait_epochs=100
batch_size=128
train_batch_size=128
dev_batch_size=128
learning_rate=0.002

input_dropout_rate=0.0
hidden_dropout_rate=0.25
feature_dropout_rate=0.25
num_out_channels_flag=200
kernel_size=5

model_dir="save_model/wn18rr/sacn"
