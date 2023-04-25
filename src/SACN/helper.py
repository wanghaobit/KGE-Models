import torch


def get_adjacencies(train_batcher, num_entities, num_relations):
    data = []
    rows = []
    columns = []

    for i, triple in enumerate(train_batcher):
        e1, e2, r = triple
        for j in range(len(e2)):
            if e2[j] != 0:
                data.append(r)
                rows.append(e1)
                columns.append(e2[j])
    # print()

    rows = rows + [i for i in range(num_entities)]
    columns = columns + [i for i in range(num_entities)]
    data = data + [num_relations for _ in range(num_entities)]

    indices = torch.LongTensor([rows, columns]).cuda()
    v = torch.LongTensor(data).cuda()
    adjacencies = [indices, v, num_entities]
    return adjacencies
