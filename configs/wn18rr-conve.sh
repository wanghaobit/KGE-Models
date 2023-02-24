#!/usr/bin/env bash

data_dir="data/WN18RR"
emb_model="ConvE"
add_reversed_training_edges="True"

entity_dim=200
relation_dim=200
emb_dropout_rate=0.3
num_epochs=1000
num_peek_epochs=1
num_wait_epochs=100
batch_size=1024
train_batch_size=1024
dev_batch_size=128
learning_rate=0.003
grad_norm=5

model_dir="save_model/wn18rr/conve"
