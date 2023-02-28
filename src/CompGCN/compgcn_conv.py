import inspect
from torch_scatter import scatter_add

from src.CompGCN.helper import *


class MessagePassing(nn.Module):
    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':  # If arguments ends with _i then include indic
                tmp = kwargs[arg[:-2]]  # Take the front part of the variable | Mostly it will be 'x',
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])  # Lookup for head entities in edges
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]  # tmp = kwargs['x']
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])  # Lookup for tail entities in edges
            else:
                message_args.append(kwargs[arg])  # Take things from kwargs

        update_args = [kwargs[arg] for arg in self.update_args]  # Take update args from kwargs

        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)  # Aggregated neighbors for each vertex
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def update(self, aggr_out):  # pragma: no cover
        return aggr_out


class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x, args=None):
        super(self.__class__, self).__init__()

        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act

        self.w_loop = get_param((in_channels, out_channels))
        self.w_in = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))

        self.GCNDropout = nn.Dropout(args.gcn_dropout_rate)
        self.bn = nn.BatchNorm1d(out_channels)

        if args.bias:
            self.register_parameter('b', nn.Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed):
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).cuda()

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                  edge_norm=None, mode='loop')
        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                 edge_norm=self.out_norm, mode='out')
        out = self.GCNDropout(in_res) * (1 / 3) + self.GCNDropout(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.args.bias:
            out = out + self.b
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        # xj_rel = self.rel_transform(x_j, rel_emb)
        xj_rel = ccorr(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
