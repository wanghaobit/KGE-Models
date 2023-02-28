import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import src.eval as eval
from src.parse_args import args
import src.data_loader as data_loader
import src.CompGCN.helper as CompGCN_helper
from src.models import TransE, RotatE
from src.models import DistMult, ComplEx, TuckER
from src.models import ConvE, AcrE
from src.models import CompGCN


class Runner(nn.Module):
    def __init__(self, args):
        super(Runner, self).__init__()
        self.args = args
        self.__print_all_settings()

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs

        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.label_smoothing_epsilon = args.label_smoothing_epsilon
        self.loss_fun = nn.BCELoss()
        self.optim = None

        self.entity2id, self.id2entity, self.num_entities, \
        self.relation2id, self.id2relation, self.num_relations = data_loader.load_graph_data(args.data_dir)
        self.dev_objects, self.all_objects = data_loader.load_all_answers(args.data_dir, add_reversed_edges=True)
        self.train_data, self.train_data_tuple,\
        self.dev_tail_data, self.dev_head_data,\
        self.test_tail_data, self.test_head_data = self.load_kg_data()

        # Translation-based Models
        if args.emb_model == 'TransE':
            self.kge = TransE(args, self.num_entities, self.num_relations)
        elif args.emb_model == "RotatE":
            self.kge = RotatE(args, self.num_entities, self.num_relations)
        # Semantic Matching-based Models
        elif args.emb_model == 'DistMult':
            self.kge = DistMult(args, self.num_entities, self.num_relations)
        elif args.emb_model == 'ComplEx':
            self.kge = ComplEx(args, self.num_entities, self.num_relations)
        elif args.emb_model == 'TuckER':
            self.kge = TuckER(args, self.num_entities, self.num_relations)
        # CNN-based Models
        elif args.emb_model == 'ConvE':
            self.kge = ConvE(args, self.num_entities, self.num_relations)
        elif args.emb_model == "AcrE":
            self.kge = AcrE(args, self.num_entities, self.num_relations)
        # GCN-based Models
        elif args.emb_model == "CompGCN":
            self.kge = CompGCN(args, self.num_entities, self.num_relations)
            edge_index, edge_type = CompGCN_helper.construct_adj(self.train_data_tuple, self.num_relations)
            self.kge.set_edge(edge_index, edge_type)
        else:
            raise NotImplementedError
        self.kge.cuda()

    def load_kg_data(self):
        train_path = data_loader.get_train_path(args.data_dir)
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        test_path = os.path.join(args.data_dir, 'test.triples')
        entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
        relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')

        train_data_group, _ = data_loader.load_triples(train_path, entity_index_path, relation_index_path,
                                                 add_reverse_relations=args.add_reversed_training_edges,
                                                 group_examples_by_query=True)
        train_data_tuple, _ = data_loader.load_triples(train_path, entity_index_path, relation_index_path,
                                                 add_reverse_relations=args.add_reversed_training_edges)
        if 'NELL' in args.data_dir:
            adj_list_path = os.path.join(args.data_dir, 'adj_list.pkl')
            seen_entities = data_loader.load_seen_entities(adj_list_path, entity_index_path)
        else:
            seen_entities = set()
        dev_tail_data, dev_head_data = data_loader.load_triples(dev_path, entity_index_path, relation_index_path,
                                                                seen_entities=seen_entities, inverse_triple=True)
        test_tail_data, test_head_data = data_loader.load_triples(test_path, entity_index_path, relation_index_path,
                                                                  seen_entities=seen_entities, inverse_triple=True)
        return train_data_group, train_data_tuple, dev_tail_data, dev_head_data, test_tail_data, test_head_data

    def run_train(self, start_epoch=0):
        train_data = self.train_data
        dev_tail_data = self.dev_tail_data
        dev_head_data = self.dev_head_data

        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []

        for epoch_id in range(start_epoch, self.num_epochs):
            print('Train Epoch {}...'.format(epoch_id))
            # Update model parameters
            self.train()
            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            batch_losses = []
            start_time = time.time()
            for example_id in range(0, len(train_data), self.batch_size):
                self.optim.zero_grad()
                mini_batch = train_data[example_id:example_id + self.batch_size]
                if len(mini_batch) < self.batch_size:
                    continue
                loss = self.loss(mini_batch)
                loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)
                self.optim.step()
                batch_losses.append(loss['print_loss'])
            # Check training statistics
            stdout_msg = 'Epoch {}: average training loss = {}, times: {}'.format(
                                epoch_id, np.mean(batch_losses), time.time()-start_time)
            print(stdout_msg)
            # Check dev set performance
            # if self.run_analysis or (epoch_id > 0 and epoch_id % self.num_peek_epochs == 0):
            if (epoch_id+1) % self.num_peek_epochs == 0:
                self.eval()
                self.batch_size = self.dev_batch_size
                with torch.no_grad():
                    dev_tail_scores = self.forward(dev_tail_data)
                    dev_head_scores = self.forward(dev_head_data)

                    print('Dev set performance: (correct evaluation)')
                    left_results = eval.sum_hits_and_ranks(dev_tail_data, dev_tail_scores, self.dev_objects)
                    right_results = eval.sum_hits_and_ranks(dev_head_data, dev_head_scores, self.dev_objects)
                    results = eval.avg_hits_and_ranks(left_results, right_results)
                    metrics = results['mrr']

                    print('Dev set performance: (include test set labels)')
                    left_results = eval.sum_hits_and_ranks(dev_tail_data, dev_tail_scores, self.all_objects)
                    right_results = eval.sum_hits_and_ranks(dev_head_data, dev_head_scores, self.all_objects)
                    eval.avg_hits_and_ranks(left_results, right_results, verbose=True)

                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.__save_checkpoint(epoch_id)
                    best_dev_metrics = metrics
                else:
                    # Early stop
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(dev_metrics_history[-self.num_wait_epochs:]):
                        break
                dev_metrics_history.append(metrics)

    # train
    def loss(self, mini_batch):
        # compute object training loss
        e1, e2, r = data_loader.format_batch(mini_batch, num_labels=self.num_entities)
        e2_label = ((1 - self.label_smoothing_epsilon) * e2) + (1.0 / e2.size(1))
        pred_scores = self.kge.forward(e1, r)
        loss = self.loss_fun(pred_scores, e2_label)
        loss_dict = {}
        loss_dict['model_loss'] = loss
        loss_dict['print_loss'] = float(loss)
        loss_dict['pred_scores'] = pred_scores
        return loss_dict

    # dev and test
    def forward(self, examples):
        pred_scores = []
        for example_id in range(0, len(examples), self.batch_size):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.__make_full_batch(mini_batch, self.batch_size)
            e1, e2, r = data_loader.format_batch(mini_batch)
            pred_score = self.kge.forward(e1, r)
            pred_scores.append(pred_score[:mini_batch_size])
        scores = torch.cat(pred_scores)
        return scores

    def forward_fact(self, examples):
        pred_scores = []
        for example_id in range(0, len(examples), self.batch_size):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.__make_full_batch(mini_batch, self.batch_size)
            e1, e2, r = data_loader.format_batch(mini_batch)
            pred_score = self.kge.forward_fact(e1, r, e2)
            pred_scores.append(pred_score[:mini_batch_size])
        return torch.cat(pred_scores)

    def test(self):
        dev_tail_data = self.dev_tail_data
        dev_head_data = self.dev_head_data
        test_tail_data = self.test_tail_data
        test_head_data = self.test_head_data

        epoch_id = self.__load_checkpoint()
        print("Best epoch id: {}".format(epoch_id))
        self.eval()
        self.batch_size = self.dev_batch_size
        with torch.no_grad():
            dev_tail_scores = self.forward(dev_tail_data)
            dev_head_scores = self.forward(dev_head_data)

            print('Dev set performance: (correct evaluation)')
            left_results = eval.sum_hits_and_ranks(dev_tail_data, dev_tail_scores, self.dev_objects)
            right_results = eval.sum_hits_and_ranks(dev_head_data, dev_head_scores, self.dev_objects)
            eval.avg_hits_and_ranks(left_results, right_results)

            print('Dev set performance: (include test set labels)')
            left_results = eval.sum_hits_and_ranks(dev_tail_data, dev_tail_scores, self.all_objects)
            right_results = eval.sum_hits_and_ranks(dev_head_data, dev_head_scores, self.all_objects)
            eval.avg_hits_and_ranks(left_results, right_results, verbose=True)

            print('Test set performance:')
            pred_tail_scores = self.forward(test_tail_data)
            pred_head_scores = self.forward(test_head_data)
            left_results = eval.sum_hits_and_ranks(test_tail_data, pred_tail_scores, self.all_objects)
            right_results = eval.sum_hits_and_ranks(test_head_data, pred_head_scores, self.all_objects)
            eval.avg_hits_and_ranks(left_results, right_results, verbose=True)

    def __save_checkpoint(self, epoch_id):
        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)
        checkpoint_dict = dict()
        checkpoint_dict['epoch_id'] = epoch_id
        checkpoint_dict['state_dict'] = self.kge.state_dict()
        best_path = os.path.join(self.args.model_dir, '{}_best.tar'.format(args.emb_model))
        torch.save(checkpoint_dict, best_path)
        print('=> best model updated \'{}\''.format(best_path))

    def __load_checkpoint(self):
        checkpoint_path = os.path.join(self.args.model_dir, '{}_best.tar'.format(args.emb_model))
        print('=> loading checkpoint \'{}\''.format(checkpoint_path))
        checkpoint_dict = torch.load(checkpoint_path, map_location="cuda:{}".format(self.args.gpu))
        epoch_id = checkpoint_dict['epoch_id']
        self.kge.load_state_dict(checkpoint_dict['state_dict'])
        return epoch_id

    def __make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = data_loader.dummy_e()
        dummy_r = data_loader.dummy_r()
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def __print_all_settings(self):
        args = vars(self.args)
        maxLen = max([len(ii) for ii in args.keys()])
        fmtString = '\t%' + str(maxLen) + 's : %s'
        print('Arguments:')
        # sorted()
        for keyPair in args.items():
            print(fmtString % keyPair)

    def continue_train_model(self):
        print("Load Model From Break Point...")
        epoch_id = self.__load_checkpoint()
        print("Last Best Performance...")
        self.test()
        print("Continue to Train Model...")
        self.run_train(epoch_id+1)


def run_experiments():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        print("GPU is available!")
    else:
        print("GPU is not available!")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.process_data:
        print("Preprocess Dataset...")
        train_path = data_loader.get_train_path(args.data_dir)
        dev_path = os.path.join(args.data_dir, 'dev.triples')
        test_path = os.path.join(args.data_dir, 'test.triples')
        data_loader.prepare_kb_envrioment(train_path, dev_path, test_path)
    elif args.train:
        print("Initial KGE Model...")
        runner = Runner(args)
        runner.cuda()
        print("Training KGE Model...")
        runner.run_train()
        print("Best Performance:")
        runner.test()
    elif args.continue_train:
        print("Initial KGE Model...")
        runner = Runner(args)
        runner.cuda()
        runner.continue_train_model()
        print("Best Performance:")
        runner.test()
    else:
        print("Initial KGE Model...")
        runner = Runner(args)
        runner.cuda()
        print("Best Performance:")
        runner.test()
    print("End of Program.")


if __name__ == "__main__":
    run_experiments()
