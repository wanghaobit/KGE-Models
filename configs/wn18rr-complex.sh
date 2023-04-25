#!/usr/bin/env bash

data_dir="data/WN18RR"
emb_model="ComplEx"
add_reversed_training_edges="True"

entity_dim=200
relation_dim=200
num_epochs=1000
num_wait_epochs=100
batch_size=2048
train_batch_size=2048
dev_batch_size=512
learning_rate=0.003

emb_dropout_rate=0.2

model_dir="save_model/wn18rr/complex"