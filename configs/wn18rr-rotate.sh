#!/usr/bin/env bash

data_dir="data/WN18RR"
emb_model="RotatE"
add_reversed_training_edges="True"

entity_dim=1000
relation_dim=1000
num_epochs=1000
num_wait_epochs=100
batch_size=8
train_batch_size=8
dev_batch_size=4
learning_rate=0.0001

emb_dropout_rate=0

model_dir="save_model/wn18rr/rotate"