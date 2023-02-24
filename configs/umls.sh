#!/usr/bin/env bash

data_dir="data/umls"
emb_model="TransE"
add_reversed_training_edges="True"

entity_dim=200
relation_dim=200
emb_dropout_rate=0.2
num_epochs=1000
num_peek_epochs=1
num_wait_epochs=100
batch_size=512
train_batch_size=512
dev_batch_size=64
learning_rate=0.003
grad_norm=0

model_dir="save_model/umls/TransE"
