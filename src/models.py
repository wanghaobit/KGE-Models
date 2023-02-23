import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class KGEModel(nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(KGEModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.emb_dropout_rate = args.emb_dropout_rate
        self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
        self.EDropout = nn.Dropout(self.emb_dropout_rate)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    def get_entity_embeddings(self, e):
        return self.EDropout(self.entity_embeddings(e))

    def get_relation_embeddings(self, r):
        return self.RDropout(self.relation_embeddings(r))

    def get_all_entity_embeddings(self):
        return self.EDropout(self.entity_embeddings.weight)

    def get_all_relation_embeddings(self):
        # return self.RDropout(self.relation_embeddings.weight)
        return self.relation_embeddings.weight


# Translation-based Models
class TransE(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(TransE, self).__init__(args, num_entities, num_relations)

    def forward(self, e1, r):
        E1 = self.get_entity_embeddings(e1)
        R = self.get_relation_embeddings(r)
        E2 = self.get_all_entity_embeddings()
        score = self.p.gamma - torch.norm((E1 + R).unsqueeze(1) - E2, p=1, dim=2)

        S = torch.sigmoid(score)
        return S

    def forward_fact(self, e1, r, e2):
        E1 = self.get_entity_embeddings(e1)
        R = self.get_relation_embeddings(r)
        E2 = self.get_entity_embeddings(e2)
        score = self.p.gamma - torch.norm((E1 + R).unsqueeze(1) - E2, p=1, dim=2)

        S = torch.sigmoid(score)
        return S


class RotatE(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(RotatE, self).__init__(args, num_entities, num_relations)
        self.epsilon = 2.0
        self.pi = 3.14159265358979323846
        gamma = args.gamma
        assert (self.entity_dim == self.relation_dim)
        hidden_dim = self.entity_dim  # 500
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        # self.entity_embeddings = nn.Parameter(torch.zeros(num_entities, self.entity_dim*2))
        # self.relation_embeddings = nn.Parameter(torch.zeros(num_relations, self.relation_dim))
        # nn.init.uniform_(tensor=self.entity_embeddings, a=-self.embedding_range.item(), b=self.embedding_range.item())
        # nn.init.uniform_(tensor=self.relation_embeddings, a=-self.embedding_range.item(), b=self.embedding_range.item())
        self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim * 2)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        nn.init.uniform_(tensor=self.entity_embeddings.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())
        nn.init.uniform_(tensor=self.relation_embeddings.weight, a=-self.embedding_range.item(), b=self.embedding_range.item())

    def get_entity_embeddings(self, e):
        # return torch.index_select(self.entity_embeddings, dim=0, index=e)
        return self.entity_embeddings(e)

    def get_relation_embeddings(self, r):
        # return torch.index_select(self.relation_embeddings, dim=0, index=r)
        return self.relation_embeddings(r)

    def get_all_entity_embeddings(self):
        return self.entity_embeddings.weight

    def forward(self, e1, r):
        # [b, 2*e]
        E1 = self.get_entity_embeddings(e1)
        # [b, e]
        R = self.get_relation_embeddings(r)
        # [n, 2*e]
        E2 = self.get_all_entity_embeddings()
        # [b, e]
        E1_real, E1_img = torch.chunk(E1, 2, dim=1)
        # [n, e]
        E2_real, E2_img = torch.chunk(E2, 2, dim=1)

        # Make phases of relations uniformly distributed in [-pi, pi]
        # [b, e]
        phase_relation = R / (self.embedding_range.item() / self.pi)
        R_real = torch.cos(phase_relation)
        R_img = torch.sin(phase_relation)

        # 复数乘法 (a+bi)(c+di)=(ac+bd)+(ad+bc)i
        # [b, e]
        # ac+bd
        re_score = E1_real * R_real - E1_img * R_img
        # ad+bc
        im_score = E1_real * R_img + E1_img * R_real
        # re_score [b, 1, e],  E2_real [1, n, e]  --> [b, n, e]
        # 实部运算 h*r-t
        re_score = re_score.unsqueeze(1) - E2_real.unsqueeze(0)
        # 虚部运算 h*r-t
        im_score = im_score.unsqueeze(1) - E2_img.unsqueeze(0)

        # [2, b, n, e]
        score = torch.stack([re_score, im_score], dim=0)
        # [b, n, e]
        score = score.norm(dim=0)
        # [b, n]
        score = self.gamma.item() - score.sum(dim=2)

        S = torch.sigmoid(score)
        return S

    def forward_fact(self, e1, r, e2):
        E1 = self.get_entity_embeddings(e1)
        R = self.get_relation_embeddings(r)
        E2 = self.get_entity_embeddings(e2)
        E1_real, E1_img = torch.chunk(E1, 2, dim=1)
        E2_real, E2_img = torch.chunk(E2, 2, dim=1)

        # Make phases of relations uniformly distributed in [-pi, pi]
        # [b, e]
        phase_relation = R / (self.embedding_range.item() / self.pi)

        R_real = torch.cos(phase_relation)
        R_img = torch.sin(phase_relation)

        # [b, e]
        re_score = E1_real * R_real - E1_img * R_img
        im_score = E1_real * R_img + E1_img * R_real
        re_score = re_score - E2_real
        im_score = im_score - E2_img

        # [2, b, e]
        score = torch.stack([re_score, im_score], dim=0)
        # [b, e]
        score = score.norm(dim=0)
        # [b, 1]
        score = self.gamma.item() - score.sum(dim=-1, keepdim=True)

        S = torch.sigmoid(score)
        return S


# Semantic Matching-based models
class DistMult(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(DistMult, self).__init__(args, num_entities, num_relations)

    def forward(self, e1, r):
        E1 = self.get_entity_embeddings(e1)
        R = self.get_relation_embeddings(r)
        E2 = self.get_all_entity_embeddings()
        S = torch.mm(E1 * R, E2.transpose(1, 0))
        # x += self.bias.expand_as(x)
        S = torch.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2):
        E1 = self.get_entity_embeddings(e1)
        R = self.get_relation_embeddings(r)
        E2 = self.get_entity_embeddings(e2)
        S = torch.sum(E1 * R * E2, dim=1, keepdim=True)
        # x += self.bias.expand_as(x)
        S = torch.sigmoid(S)
        return S


class ComplEx(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(ComplEx, self).__init__(args, num_entities, num_relations)
        self.entity_img_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
        self.relation_img_embeddings = nn.Embedding(self.num_relations, self.relation_dim)

        # 初始化问题？
        # nn.init.xavier_normal_(self.entity_img_embeddings.weight)
        # nn.init.xavier_normal_(self.relation_img_embeddings.weight)

    def get_relation_img_embeddings(self, r):
        return self.RDropout(self.relation_img_embeddings(r))

    def get_entity_img_embeddings(self, e):
        return self.EDropout(self.entity_img_embeddings(e))

    def get_all_entity_img_embeddings(self):
        return self.EDropout(self.entity_img_embeddings.weight)

    def forward(self, e1, r):
        def dist_mult(E1, R, E2):
            return torch.mm(E1 * R, E2.transpose(1, 0))

        E1_real = self.get_entity_embeddings(e1)
        R_real = self.get_relation_embeddings(r)
        E2_real = self.get_all_entity_embeddings()
        E1_img = self.get_entity_img_embeddings(e1)
        R_img = self.get_relation_img_embeddings(r)
        E2_img = self.get_all_entity_img_embeddings()

        rrr = dist_mult(R_real, E1_real, E2_real)
        rii = dist_mult(R_real, E1_img, E2_img)
        iri = dist_mult(R_img, E1_real, E2_img)
        iir = dist_mult(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir

        S = torch.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2):
        def dist_mult_fact(E1, R, E2):
            return torch.sum(E1 * R * E2, dim=1, keepdim=True)

        E1_real = self.get_entity_embeddings(e1)
        R_real = self.get_relation_embeddings(r)
        E2_real = self.get_entity_embeddings(e2)
        E1_img = self.get_entity_img_embeddings(e1)
        R_img = self.get_relation_img_embeddings(r)
        E2_img = self.get_entity_img_embeddings(e2)

        rrr = dist_mult_fact(R_real, E1_real, E2_real)
        rii = dist_mult_fact(R_real, E1_img, E2_img)
        iri = dist_mult_fact(R_img, E1_real, E2_img)
        iir = dist_mult_fact(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir

        S = torch.sigmoid(S)
        return S


class TuckER(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(TuckER, self).__init__(args, num_entities, num_relations)
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.W = torch.nn.Parameter(torch.tensor(
            np.random.uniform(-1, 1, (self.relation_dim, self.entity_dim, self.entity_dim)),
            dtype=torch.float, device="cuda", requires_grad=True))
        self.InputDropout = nn.Dropout(args.input_dropout_rate)
        self.HiddenDropout1 = nn.Dropout(args.hidden_dropout_rate_1)
        self.HiddenDropout2 = nn.Dropout(args.hidden_dropout_rate_2)
        self.bn0 = nn.BatchNorm1d(self.entity_dim)
        self.bn1 = nn.BatchNorm1d(self.relation_dim)

    def forward(self, e1, r):
        E1 = self.get_entity_embeddings(e1)
        R = self.get_relation_embeddings(r)
        E2 = self.get_all_entity_embeddings()

        x = self.bn0(E1)
        x = self.InputDropout(x)
        x = x.view(-1, 1, E1.size(1))

        W_mat = torch.mm(R, self.W.view(R.size(1), -1))
        W_mat = W_mat.view(-1, E1.size(1), E1.size(1))
        W_mat = self.HiddenDropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, E1.size(1))
        x = self.bn1(x)
        x = self.HiddenDropout2(x)
        x = torch.mm(x, E2.transpose(1, 0))

        S = torch.sigmoid(x)
        return S

    def forward_fact(self, e1, r, e2):
        E1 = self.get_entity_embeddings(e1)
        R = self.get_relation_embeddings(r)
        E2 = self.get_entity_embeddings(e2)

        x = self.bn0(E1)
        x = self.InputDropout(x)
        x = x.view(-1, 1, E1.size(1))

        W_mat = torch.mm(R, self.W.view(R.size(1), -1))
        W_mat = W_mat.view(-1, E1.size(1), E1.size(1))
        W_mat = self.HiddenDropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, E1.size(1))
        x = self.bn1(x)
        x = self.HiddenDropout2(x)
        x = torch.mm(x, E2.transpose(1, 0))

        S = torch.sigmoid(x)
        return S


# CNN-based Models
class ConvE(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(ConvE, self).__init__(args, num_entities, num_relations)
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        assert(args.emb_2D_d1 * args.emb_2D_d2 == args.entity_dim)
        assert(args.emb_2D_d1 * args.emb_2D_d2 == args.relation_dim)
        self.emb_2D_d1 = args.emb_2D_d1
        self.emb_2D_d2 = args.emb_2D_d2
        self.num_out_channels = args.num_out_channels
        self.w_d = args.kernel_size
        self.HiddenDropout = nn.Dropout(args.hidden_dropout_rate)
        self.FeatureDropout = nn.Dropout(args.feat_dropout_rate)

        # stride = 1, padding = 0, dilation = 1, groups = 1
        self.conv1 = nn.Conv2d(1, self.num_out_channels, (self.w_d, self.w_d), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm1d(self.entity_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        h_out = 2 * self.emb_2D_d1 - self.w_d + 1
        w_out = self.emb_2D_d2 - self.w_d + 1
        self.feat_dim = self.num_out_channels * h_out * w_out
        self.fc = nn.Linear(self.feat_dim, self.entity_dim)

    def forward(self, e1, r):
        E1 = self.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = self.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = self.get_all_entity_embeddings()

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        # stacked_inputs = self.InputDropout(stacked_inputs)
        X = self.conv1(stacked_inputs)
        # X = self.bn1(X)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.mm(X, E2.transpose(1, 0))
        X += self.b.expand_as(X)

        S = torch.sigmoid(X)
        return S

    def forward_fact(self, e1, r, e2):
        E1 = self.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = self.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = self.get_entity_embeddings(e2)

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        # stacked_inputs = self.InputDropout(stacked_inputs)
        X = self.conv1(stacked_inputs)
        # X = self.bn1(X)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.matmul(X.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)
        X += self.b[e2].unsqueeze(1)

        S = torch.sigmoid(X)
        return S


class AcrE(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(AcrE, self).__init__(args, num_entities, num_relations)

        self.inp_drop = torch.nn.Dropout(args.inp_drop)
        self.hidden_drop = torch.nn.Dropout(args.hid_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.channel)
        self.bn2 = torch.nn.BatchNorm1d(args.entity_dim)
        self.fc = torch.nn.Linear(args.channel * 400, args.entity_dim)
        self.padding = 0
        self.way = args.way
        self.bias = args.bias
        self.channel = args.channel
        self.first_atrous = args.first_atrous
        self.second_atrous = args.second_atrous
        self.third_atrous = args.third_atrous

        if self.way == 's':
            self.conv1 = torch.nn.Conv2d(1, self.channel, (3, 3), 1, self.first_atrous, bias=self.bias,
                                         dilation=self.first_atrous)
            self.conv2 = torch.nn.Conv2d(self.channel, self.channel, (3, 3), 1, self.second_atrous,
                                         bias=self.bias, dilation=self.second_atrous)
            self.conv3 = torch.nn.Conv2d(self.channel, self.channel, (3, 3), 1, self.third_atrous, bias=self.bias,
                                         dilation=self.third_atrous)
        else:
            self.conv1 = torch.nn.Conv2d(1, self.channel, (3, 3), 1, self.first_atrous, bias=self.bias,
                                         dilation=self.first_atrous)
            self.conv2 = torch.nn.Conv2d(1, self.channel, (3, 3), 1, self.second_atrous, bias=self.bias,
                                         dilation=self.second_atrous)
            self.conv3 = torch.nn.Conv2d(1, self.channel, (3, 3), 1, self.third_atrous, bias=self.bias,
                                         dilation=self.third_atrous)
            self.W_gate_e = torch.nn.Linear(1600, 400)

        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))

    def forward(self, e1, r):
        E1 = self.get_entity_embeddings(e1).view(-1, 1, 10, 20)
        R = self.get_relation_embeddings(r).view(-1, 1, 10, 20)
        E2 = self.get_all_entity_embeddings()
        comb_emb = torch.cat([E1, R], dim=2)
        stack_inp = self.bn0(comb_emb)
        x = self.inp_drop(stack_inp)
        res = x
        if self.way == 's':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x + res
        else:
            conv1 = self.conv1(x).view(-1, self.channel, 400)
            conv2 = self.conv2(x).view(-1, self.channel, 400)
            conv3 = self.conv3(x).view(-1, self.channel, 400)
            res = res.expand(-1, self.channel, 20, 20).view(-1, self.channel, 400)
            x = torch.cat((res, conv1, conv2, conv3), dim=2)
            x = self.W_gate_e(x).view(-1, self.channel, 20, 20)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # x[b, e] E2[n, e] --> x[b, n]
        x = torch.mm(x, E2.transpose(1, 0))
        x += self.b.expand_as(x)
        S = torch.sigmoid(x)
        return S

    def forward_fact(self, e1, r, e2):
        E1 = self.get_entity_embeddings(e1).view(-1, 1, 10, 20)
        R = self.get_relation_embeddings(r).view(-1, 1, 10, 20)
        E2 = self.get_entity_embeddings(e2)
        comb_emb = torch.cat([E1, R], dim=2)
        stack_inp = self.bn0(comb_emb)
        x = self.inp_drop(stack_inp)
        res = x
        if self.way == 's':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x + res
        else:
            conv1 = self.conv1(x).view(-1, self.channel, 400)
            conv2 = self.conv2(x).view(-1, self.channel, 400)
            conv3 = self.conv3(x).view(-1, self.channel, 400)
            res = res.expand(-1, self.channel, 20, 20).view(-1, self.channel, 400)
            x = torch.cat((res, conv1, conv2, conv3), dim=2)
            x = self.W_gate_e(x).view(-1, self.channel, 20, 20)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # x[b, e] E2[b, e] --> [b, e] --> [b, 1]
        x = torch.mul(x, E2).sum(dim=-1, keepdim=True)
        # x[b, e]->[b, 1, e]    E2[b, e]->[b, e, 1]  --> [b, 1, 1]  --> [b, 1]
        # x = torch.matmul(x.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)
        x += self.b[e2].unsqueeze(1)

        # pred_et.unsqueeze(1)
        # x = torch.matmul(x.unsqueeze(1), E2.permute(0, 2, 1)).squeeze(1)
        # x += self.b[e2]

        S = torch.sigmoid(x)
        return S


# GNN-based Models
class SACN(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(SACN, self).__init__(args, num_entities, num_relations)

    def forward(self, e1, r):
        S = torch.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2):
        S = torch.sigmoid(S)
        return S


class CompGCN(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(CompGCN, self).__init__(args, num_entities, num_relations)

    def forward(self, e1, r):
        S = torch.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2):
        S = torch.sigmoid(S)
        return S


# Add new model here
class NewModel(KGEModel):
    def __init__(self, args, num_entities, num_relations):
        super(NewModel, self).__init__(args, num_entities, num_relations)

    def forward(self, e1, r):
        E1 = self.get_entity_embeddings(e1)
        R = self.get_relation_embeddings(r)
        E2 = self.get_all_entity_embeddings()
        # Add your model function here
        # The forward function should operate on the embeddings e1 and rel
        # and output scores for all entities (you will need a projection layer
        # with output size num_relations (from constructor above)

        # generate output scores here
        S = torch.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2):
        E1 = self.get_entity_embeddings(e1)
        R = self.get_relation_embeddings(r)
        E2 = self.get_entity_embeddings(e2)
        # Add your model function here
        # The forward_fact function should operate on the embeddings e1, r, and e2
        # and output scores for this triple (e1, r, e2)

        # generate output scores here
        S = torch.sigmoid(S)
        return S
