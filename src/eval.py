import torch
import numpy as np

from src.data_loader import NO_OP_ENTITY_ID, DUMMY_ENTITY_ID, var_cuda


def hits_and_ranks1(examples, scores, all_answers, verbose=False):
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), 128))
    # top_k_scores, top_k_targets = torch.topk(scores, scores.size(1))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0
    for i, example in enumerate(examples):
        e1, e2, r = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1
            mrr += 1.0 / (pos + 1)

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)
    mrr = float(mrr) / len(examples)

    if verbose:
        print('Hits@1 = {:.3f}'.format(hits_at_1))
        print('Hits@3 = {:.3f}'.format(hits_at_3))
        print('Hits@5 = {:.3f}'.format(hits_at_5))
        print('Hits@10 = {:.3f}'.format(hits_at_10))
        print('MRR = {:.3f}'.format(mrr))

    return hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr


def sum_hits_and_ranks(examples, scores, all_answers):
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    batch_e2 = []
    for i, example in enumerate(examples):
        e1, e2, r = example
        batch_e2.append(e2)
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    obj = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
    b_range = var_cuda(torch.arange(scores.size()[0]), requires_grad=False)
    ranks = 1 + torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
    ranks = ranks.float()
    results = {}
    results['count'] = torch.numel(ranks) + results.get('count', 0.0)
    results['mr'] = round(torch.sum(ranks).item() + results.get('mr', 0.0), 5)
    results['mrr'] = round(torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0), 5)
    for k in range(10):
        results['hits@{}'.format(k + 1)] = round(torch.numel(ranks[ranks <= (k + 1)])
                                                  + results.get('hits@{}'.format(k + 1), 0.0), 5)
    return results


def avg_hits_and_ranks(left_results, right_results, verbose=False):
    results = {}
    results['left_mr'] = round(left_results['mr'] / left_results['count'], 4)
    results['left_mrr'] = round(left_results['mrr'] / left_results['count'], 4)
    results['right_mr'] = round(right_results['mr'] / right_results['count'], 4)
    results['right_mrr'] = round(right_results['mrr'] / right_results['count'], 4)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (left_results['count'] + right_results['count']), 4)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (left_results['count'] + right_results['count']), 4)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k+1)] / left_results['count'], 4)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k+1)] / right_results['count'], 4)
        results['hits@{}'.format(k + 1)] = round((left_results['hits@{}'.format(k+1)]
                        + right_results['hits@{}'.format(k+1)]) / (left_results['count'] + right_results['count']), 4)

    stdout_left = 'Left performance: \t Hits@1 = {:.3f}, \t Hits@10 = {:.3f}, \t MRR = {:.3f}'.format(
        results['left_hits@1'], results['left_hits@10'], results['left_mrr'])
    print(stdout_left)

    stdout_right = 'Right performance: \t Hits@1 = {:.3f}, \t Hits@10 = {:.3f}, \t MRR = {:.3f}'.format(
        results['right_hits@1'], results['right_hits@10'], results['right_mrr'])
    print(stdout_right)

    if verbose:
        print('Average performance:')
        print('Hits@1 = {:.3f}'.format(results['hits@1']))
        print('Hits@3 = {:.3f}'.format(results['hits@3']))
        print('Hits@5 = {:.3f}'.format(results['hits@5']))
        print('Hits@10 = {:.3f}'.format(results['hits@10']))
        print('MRR = {:.3f}'.format(results['mrr']))
        print('MR = {}'.format(int(results['mr'])))
    return results


def hits_and_ranks(examples, scores, all_answers, verbose=False):
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    batch_e2 = []
    for i, example in enumerate(examples):
        e1, e2, r = example
        batch_e2.append(e2)
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    obj = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
    b_range = var_cuda(torch.arange(scores.size()[0]), requires_grad=False)
    ranks = 1 + torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
    ranks = ranks.float()
    results = {}
    results['count'] = torch.numel(ranks) + results.get('count', 0.0)
    results['mr'] = round((torch.sum(ranks).item() + results.get('mr', 0.0)) / results['count'], 3)
    results['mrr'] = round((torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)) / results['count'], 3)
    for k in range(10):
        results['hits@{}'.format(k + 1)] = round((torch.numel(ranks[ranks <= (k + 1)])
                                                  + results.get('hits@{}'.format(k+1), 0.0)) / results['count'], 3)

    if verbose:
        print('Average performance:')
        print('Hits@1 \t = {:.3f}'.format(results['hits@1']))
        print('Hits@3 \t = {:.3f}'.format(results['hits@3']))
        print('Hits@5 \t = {:.3f}'.format(results['hits@5']))
        print('Hits@10  = {:.3f}'.format(results['hits@10']))
        print('MRR \t = {:.3f}'.format(results['mrr']))
        print('MR \t = {}'.format(int(results['mr'])))

    return results
