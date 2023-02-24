for split in range(1, 4):
    for data in ['train', 'valid', 'test']:
        triples = []
        with open('countries/countries_S{}/{}.txt'.format(split, data), 'r') as fr:
            for line in fr.readlines():
                e1, r, e2 = line.strip().split()
                triples.append((e1, e2, r))

        data = 'dev' if data == 'valid' else data
        with open('countries/countries_S{}/{}.triples'.format(split, data), 'w') as fw:
            for t in triples:
                e1, e2, r = t
                fw.write(e1 + '\t' + e2 + '\t' + r + '\n')
