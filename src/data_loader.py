import os
import torch
import pickle
import collections
from torch.autograd import Variable

DUMMY_RELATION = 'PAD'
START_RELATION = 'DUMMY_START_RELATION'
NO_OP_RELATION = 'NO_OP'
UNK_RELATION = 'UNK'
DUMMY_ENTITY = 'PAD'
NO_OP_ENTITY = 'UNK'

DUMMY_RELATION_ID = 0
START_RELATION_ID = 1
NO_OP_RELATION_ID = 2
UNK_RELATION_ID = 3
DUMMY_ENTITY_ID = 0
NO_OP_ENTITY_ID = 1


def get_train_path(data_dir):
    if 'NELL' in data_dir:
        train_path = os.path.join(data_dir, 'train.dev.large.triples')
    else:
        train_path = os.path.join(data_dir, 'train.triples')
    return train_path


def load_triples(data_path, entity_index_path, relation_index_path, add_reverse_relations=False,
                 seen_entities=None, group_examples_by_query=False, verbose=False, inverse_triple=False):
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    # for e2 in label: y[e2] = 1.0

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    triples = []
    inv_triples = []
    # sr2o = collections.defaultdict(set)
    if group_examples_by_query:
        triple_dict = {}
        inv_triple_dict = {}
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            e1, e2, r = line.strip().split()
            if seen_entities and (not e1 in seen_entities or not e2 in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip()))
                continue
            if group_examples_by_query:
                e1_id, e2_id, r_id = triple2ids(e1, e2, r)
                if e1_id not in triple_dict:
                    triple_dict[e1_id] = {}
                if r_id not in triple_dict[e1_id]:
                    triple_dict[e1_id][r_id] = set()
                triple_dict[e1_id][r_id].add(e2_id)
                if add_reverse_relations:
                    r_inv = '_' + r
                    e2_id, e1_id, r_inv_id = triple2ids(e2, e1, r_inv)
                    if e2_id not in triple_dict:
                        triple_dict[e2_id] = {}
                    if r_inv_id not in triple_dict[e2_id]:
                        triple_dict[e2_id][r_inv_id] = set()
                    triple_dict[e2_id][r_inv_id].add(e1_id)
                # To predict head entity
                if inverse_triple:
                    r_inv = '_' + r
                    e2_id, e1_id, r_inv_id = triple2ids(e2, e1, r_inv)
                    if e2_id not in inv_triple_dict:
                        inv_triple_dict[e2_id] = {}
                    if r_inv_id not in inv_triple_dict[e2_id]:
                        inv_triple_dict[e2_id][r_inv_id] = set()
                    inv_triple_dict[e2_id][r_inv_id].add(e1_id)
            else:
                triples.append(triple2ids(e1, e2, r))
                if add_reverse_relations:
                    triples.append(triple2ids(e2, e1, '_' + r))
                if inverse_triple:
                    inv_triples.append(triple2ids(e2, e1, '_' + r))

                # sub, obj, rel = triple2ids(e1, e2, r)
                # sr2o[(sub, rel)].add(obj)
                # sub, obj, rel = triple2ids(e2, e1, '_' + r)
                # sr2o[(sub, rel)].add(obj)

    if group_examples_by_query:
        for e1_id in triple_dict:
            for r_id in triple_dict[e1_id]:
                triples.append((e1_id, list(triple_dict[e1_id][r_id]), r_id))
        if inverse_triple:
            for e1_id in inv_triple_dict:
                for r_id in inv_triple_dict[e1_id]:
                    inv_triples.append((e1_id, list(inv_triple_dict[e1_id][r_id]), r_id))
    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples, inv_triples


def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index


def load_seen_entities(adj_list_path, entity_index_path):
    _, id2entity = load_index(entity_index_path)
    with open(adj_list_path, 'rb') as f:
        adj_list = pickle.load(f)
    seen_entities = set()
    for e1 in adj_list:
        seen_entities.add(id2entity[e1])
        for r in adj_list[e1]:
            for e2 in adj_list[e1][r]:
                seen_entities.add(id2entity[e2])
    print('{} seen entities loaded...'.format(len(seen_entities)))
    return seen_entities


def format_batch(batch_data, num_labels=-1, num_tiles=1, isMul=False):
    def convert_to_binary_multi_subject(e1):
        e1_label = zeros_var_cuda([len(e1), num_labels])
        for i in range(len(e1)):
            e1_label[i][e1[i]] = 1
        return e1_label

    def convert_to_binary_multi_object(e2):
        e2_label = zeros_var_cuda([len(e2), num_labels])
        for i in range(len(e2)):
            e2_label[i][e2[i]] = 1
        return e2_label

    def tile_along_beam(v, beam_size, dim=0):
        if dim == -1:
            dim = len(v.size()) - 1
        v = v.unsqueeze(dim + 1)
        v = torch.cat([v] * beam_size, dim=dim + 1)
        new_size = []
        for i, d in enumerate(v.size()):
            if i == dim + 1:
                new_size[-1] *= d
            else:
                new_size.append(d)
        return v.view(new_size)

    batch_e1, batch_e2, batch_r = [], [], []
    for i in range(len(batch_data)):
        e1, e2, r = batch_data[i]
        batch_e1.append(e1)
        if isMul:
            batch_e2.append(e2[0])
        else:
            batch_e2.append(e2)
        batch_r.append(r)
    batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
    batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
    if type(batch_e2[0]) is list:
        batch_e2 = convert_to_binary_multi_object(batch_e2)
    elif type(batch_e1[0]) is list:
        batch_e1 = convert_to_binary_multi_subject(batch_e1)
    else:
        batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
    # Rollout multiple times for each example
    if num_tiles > 1:
        batch_e1 = tile_along_beam(batch_e1, num_tiles)
        batch_r = tile_along_beam(batch_r, num_tiles)
        batch_e2 = tile_along_beam(batch_e2, num_tiles)
    return batch_e1, batch_e2, batch_r


def prepare_kb_envrioment(train_path, dev_path, test_path, add_reverse_relations=True):
    data_dir = os.path.dirname(train_path)

    def get_type(e_name):
        if e_name == DUMMY_ENTITY:
            return DUMMY_ENTITY
        if 'nell-995' in data_dir.lower():
            if '_' in e_name:
                return e_name.split('_')[1]
            else:
                return 'numerical'
        else:
            return 'entity'

    def hist_to_vocab(_dict):
        return sorted(sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)

    # Create entity and relation indices
    entity_hist = collections.defaultdict(int)
    relation_hist = collections.defaultdict(int)
    type_hist = collections.defaultdict(int)
    with open(train_path) as f:
        train_triples = [l.strip() for l in f.readlines()]
    with open(dev_path) as f:
        dev_triples = [l.strip() for l in f.readlines()]
    with open(test_path) as f:
        test_triples = [l.strip() for l in f.readlines()]

    # if test_mode:
    #     keep_triples = train_triples + dev_triples
    #     removed_triples = test_triples
    # else:
    keep_triples = train_triples
    removed_triples = dev_triples + test_triples

    # Index entities and relations
    for line in set(keep_triples + removed_triples):
        e1, e2, r = line.strip().split()
        entity_hist[e1] += 1
        entity_hist[e2] += 1
        if 'nell-995' in data_dir.lower():
            t1 = e1.split('_')[1] if '_' in e1 else 'numerical'
            t2 = e2.split('_')[1] if '_' in e2 else 'numerical'
        else:
            t1 = get_type(e1)
            t2 = get_type(e2)
        type_hist[t1] += 1
        type_hist[t2] += 1
        relation_hist[r] += 1
        if add_reverse_relations:
            # inv_r = r + '_inv'
            inv_r = '_' + r
            relation_hist[inv_r] += 1
    # Save the entity and relation indices sorted by decreasing frequency
    with open(os.path.join(data_dir, 'entity2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_ENTITY, DUMMY_ENTITY_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_ENTITY, NO_OP_ENTITY_ID))
        for e, freq in hist_to_vocab(entity_hist):
            o_f.write('{}\t{}\n'.format(e, freq))
    with open(os.path.join(data_dir, 'relation2id.txt'), 'w') as o_f:
        o_f.write('{}\t{}\n'.format(DUMMY_RELATION, DUMMY_RELATION_ID))
        o_f.write('{}\t{}\n'.format(START_RELATION, START_RELATION_ID))
        o_f.write('{}\t{}\n'.format(NO_OP_RELATION, NO_OP_RELATION_ID))
        o_f.write('{}\t{}\n'.format(UNK_RELATION, UNK_RELATION_ID))
        for r, freq in hist_to_vocab(relation_hist):
            o_f.write('{}\t{}\n'.format(r, freq))
    with open(os.path.join(data_dir, 'type2id.txt'), 'w') as o_f:
        for t, freq in hist_to_vocab(type_hist):
            o_f.write('{}\t{}\n'.format(t, freq))
    print('{} entities indexed'.format(len(entity_hist)))
    print('{} relations indexed'.format(len(relation_hist)))
    print('{} types indexed'.format(len(type_hist)))
    entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
    relation2id, id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
    type2id, id2type = load_index(os.path.join(data_dir, 'type2id.txt'))

    removed_triples = set(removed_triples)
    adj_list = collections.defaultdict(collections.defaultdict)
    entity2typeid = [0 for i in range(len(entity2id))]
    num_facts = 0
    for line in set(keep_triples):
        e1, e2, r = line.strip().split()
        triple_signature = '{}\t{}\t{}'.format(e1, e2, r)
        e1_id = entity2id[e1]
        e2_id = entity2id[e2]
        t1 = get_type(e1)
        t2 = get_type(e2)
        t1_id = type2id[t1]
        t2_id = type2id[t2]
        entity2typeid[e1_id] = t1_id
        entity2typeid[e2_id] = t2_id
        if not triple_signature in removed_triples:
            r_id = relation2id[r]
            if not r_id in adj_list[e1_id]:
                adj_list[e1_id][r_id] = set()
            if e2_id in adj_list[e1_id][r_id]:
                print('Duplicate fact: {} ({}, {}, {})!'.format(
                    line.strip(), id2entity[e1_id], id2relation[r_id], id2entity[e2_id]))
            adj_list[e1_id][r_id].add(e2_id)
            num_facts += 1
            if add_reverse_relations:
                # inv_r = r + '_inv'
                inv_r = '_' + r
                inv_r_id = relation2id[inv_r]
                if not inv_r_id in adj_list[e2_id]:
                    adj_list[e2_id][inv_r_id] = set([])
                if e1_id in adj_list[e2_id][inv_r_id]:
                    print('Duplicate fact: {} ({}, {}, {})!'.format(
                        line.strip(), id2entity[e2_id], id2relation[inv_r_id], id2entity[e1_id]))
                adj_list[e2_id][inv_r_id].add(e1_id)
                num_facts += 1
    print('{} facts processed'.format(num_facts))
    # Save adjacency list
    adj_list_path = os.path.join(data_dir, 'adj_list.pkl')
    with open(adj_list_path, 'wb') as o_f:
        pickle.dump(dict(adj_list), o_f)
    with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'wb') as o_f:
        pickle.dump(entity2typeid, o_f)


def load_graph_data(data_dir):
    # Load indices
    entity2id, id2entity = load_index(os.path.join(data_dir, 'entity2id.txt'))
    print('Sanity check: {} entities loaded'.format(len(entity2id)))
    # type2id, id2type = load_index(os.path.join(data_dir, 'type2id.txt'))
    # print('Sanity check: {} types loaded'.format(len(type2id)))
    # with open(os.path.join(data_dir, 'entity2typeid.pkl'), 'rb') as f:
    #     entity2typeid = pickle.load(f)
    relation2id, id2relation = load_index(os.path.join(data_dir, 'relation2id.txt'))
    print('Sanity check: {} relations loaded'.format(len(relation2id)))
    return entity2id, id2entity, len(entity2id), relation2id, id2relation, len(relation2id)


def load_all_answers(data_dir, add_reversed_edges=False):
    def add_subject(e1, e2, r, d):
        if not e2 in d:
            d[e2] = {}
        if not r in d[e2]:
            d[e2][r] = set()
        d[e2][r].add(e1)

    def add_object(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not r in d[e1]:
            d[e1][r] = set()
        d[e1][r].add(e2)

    def triple2ids(e1, e2, r):
        return entity2id[e1], entity2id[e2], relation2id[r]

    def get_inv_relation_id(r_id):
        return relation2id['_' + id2relation[r_id]]

    entity_index_path = os.path.join(data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(data_dir, 'relation2id.txt')
    entity2id, _ = load_index(entity_index_path)
    relation2id, id2relation = load_index(relation_index_path)

    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    train_subjects, train_objects = {}, {}
    dev_subjects, dev_objects = {}, {}
    all_subjects, all_objects = {}, {}
    # include dummy examples
    de = dummy_e()
    dr = dummy_r()
    add_subject(de, de, dr, train_subjects)
    add_subject(de, de, dr, dev_subjects)
    add_subject(de, de, dr, all_subjects)
    add_object(de, de, dr, train_objects)
    add_object(de, de, dr, dev_objects)
    add_object(de, de, dr, all_objects)
    for file_name in ['train.triples', 'dev.triples', 'test.triples']:
        if 'NELL' in data_dir and file_name == 'train.triples':
            continue
        with open(os.path.join(data_dir, file_name)) as f:
            for line in f:
                e1, e2, r = line.strip().split()
                e1, e2, r = triple2ids(e1, e2, r)
                if file_name in ['train.triples']:
                    add_subject(e1, e2, r, train_subjects)
                    add_object(e1, e2, r, train_objects)
                    if add_reversed_edges:
                        add_subject(e2, e1, get_inv_relation_id(r), train_subjects)
                        add_object(e2, e1, get_inv_relation_id(r), train_objects)
                if file_name in ['train.triples', 'dev.triples']:
                    add_subject(e1, e2, r, dev_subjects)
                    add_object(e1, e2, r, dev_objects)
                    if add_reversed_edges:
                        add_subject(e2, e1, get_inv_relation_id(r), dev_subjects)
                        add_object(e2, e1, get_inv_relation_id(r), dev_objects)
                add_subject(e1, e2, r, all_subjects)
                add_object(e1, e2, r, all_objects)
                if add_reversed_edges:
                    add_subject(e2, e1, get_inv_relation_id(r), all_subjects)
                    add_object(e2, e1, get_inv_relation_id(r), all_objects)
    return dev_objects, all_objects


def ones_var_cuda(s, requires_grad=False):
    return Variable(torch.ones(s), requires_grad=requires_grad).cuda()


def zeros_var_cuda(s, requires_grad=False):
    return Variable(torch.zeros(s), requires_grad=requires_grad).cuda()


def int_fill_var_cuda(s, value, requires_grad=False):
    return int_var_cuda((torch.zeros(s) + value), requires_grad=requires_grad)


def int_var_cuda(x, requires_grad=False):
    return Variable(x, requires_grad=requires_grad).long().cuda()


def var_cuda(x, requires_grad=False):
    return Variable(x, requires_grad=requires_grad).cuda()


def var_to_numpy(x):
    return x.data.cpu().numpy()


def self_edge():
    return NO_OP_RELATION_ID


def self_e():
    return NO_OP_ENTITY_ID


def dummy_r():
    return DUMMY_RELATION_ID


def dummy_e():
    return DUMMY_ENTITY_ID


def dummy_start_r():
    return START_RELATION_ID
