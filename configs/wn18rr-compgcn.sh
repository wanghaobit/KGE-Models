#!/usr/bin/env bash

data_dir="data/WN18RR"
emb_model="CompGCN"
add_reversed_training_edges="True"

entity_dim=200
relation_dim=200
num_epochs=500
num_wait_epochs=100
batch_size=512
train_batch_size=512
dev_batch_size=128
learning_rate=0.001

hidden_dropout_rate=0.3
hidden_dropout_rate_2=0.3
feature_dropout_rate=0.3
num_out_channels=200
kernel_size=7

model_dir="save_model/wn18rr/compgcn"
