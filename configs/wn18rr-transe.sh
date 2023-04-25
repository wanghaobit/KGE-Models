#!/usr/bin/env bash

data_dir="data/WN18RR"
emb_model="TransE"
add_reversed_training_edges="True"

entity_dim=200
relation_dim=200
num_epochs=1000
num_wait_epochs=100
batch_size=128
train_batch_size=128
dev_batch_size=128
learning_rate=0.0001

model_dir="save_model/wn18rr/transe"
