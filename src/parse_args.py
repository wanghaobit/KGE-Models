import os
import argparse


parser = argparse.ArgumentParser(description='CLM')

# Experiment Control
parser.add_argument('--process_data', action='store_true',
                    help='process knowledge graph (default: False)')
parser.add_argument('--train', action='store_true',
                    help='run path selection set_policy training (default: False)')
parser.add_argument('--continue_train', action='store_true',
                    help='continue to train model (default: False)')
parser.add_argument('--test', action='store_true',
                    help='test path selection set_policy training (default: False)')

parser.add_argument('--emb_model', type=str, default='conve',
                    help='knowledge graph pretrained embedding model (default: point)')
parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'),
                    help='directory where the knowledge graph data is stored (default: None)')
parser.add_argument('--model_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model'),
                    help='directory where the GAN model parameters are stored (default: None)')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu device (default: 0)')
parser.add_argument('--seed', type=int, default=543, metavar='S',
                    help='random seed (default: 543)')

# Optimization
parser.add_argument('--num_epochs', type=int, default=20,
                    help='maximum number of pass over the entire training set (default: 20)')
parser.add_argument('--num_wait_epochs', type=int, default=100,
                    help='number of epochs to wait before stopping training if dev set performance drops')
parser.add_argument('--num_peek_epochs', type=int, default=1,
                    help='number of epochs to wait for next dev set result check (default: 2)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='mini-batch size (default: 256)')
parser.add_argument('--train_batch_size', type=int, default=256,
                    help='mini-batch size during training (default: 256)')
parser.add_argument('--dev_batch_size', type=int, default=64,
                    help='mini-batch size during inferece (default: 64)')
# parser.add_argument('--margin', type=float, default=0,
#                     help='margin used for base MAMES training (default: 0)')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
# parser.add_argument('--learning_rate_decay', type=float, default=1.0,
#                     help='learning rate decay factor for the Adam optimizer (default: 1)')
# parser.add_argument('--adam_beta1', type=float, default=0.9,
#                     help='Adam: decay rates for the first movement estimate (default: 0.9)')
# parser.add_argument('--adam_beta2', type=float, default=0.999,
#                     help='Adam: decay rates for the second raw movement estimate (default: 0.999)')
parser.add_argument('--grad_norm', type=float, default=0,
                    help='norm threshold for gradient clipping (default 0)')
# parser.add_argument('--xavier_initialization', type=bool, default=True,
#                     help='Initialize all model parameters using xavier initialization (default: True)')
# parser.add_argument('--random_parameters', type=bool, default=False,
#                     help='Inference with random parameters (default: False)')


# KGE Models' Share Parameters
parser.add_argument('--entity_dim', type=int, default=200, metavar='E',
                    help='entity embedding dimension (default: 200)')
parser.add_argument('--relation_dim', type=int, default=200, metavar='R',
                    help='relation embedding dimension (default: 200)')
parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1,
                    help='epsilon used for label smoothing')
parser.add_argument('--add_reversed_training_edges', action='store_true',
                    help='add reversed edges to extend training set (default: False)')

parser.add_argument('--emb_dropout_rate', type=float, default=0.0,
                    help='Knowledge graph embedding dropout rate (default: 0.3)')
parser.add_argument('--input_dropout_rate', type=float, default=0.2,
                    help='Dropout for Input layer')
parser.add_argument('--hidden_dropout_rate', type=float, default=0.3,
                    help='First hidden layer dropout rate (default: 0.3)')
parser.add_argument('--hidden_dropout_rate_2', type=float, default=0.3,
                    help='Second hidden layer dropout rate (default: 0.3)')
parser.add_argument('--feature_dropout_rate', type=float, default=0.2,
                    help='Feature dropout rate (default: 0.2)')
parser.add_argument('--bias', action='store_true',
                    help='Whether to use bias in the model')

# TransE
parser.add_argument('--gamma', type=float, default=24.0,
                    help="RotatE's moving average weight")

# ConvE
parser.add_argument('--emb_2D_d1', type=int, default=10,
                    help='ConvE embedding 2D shape dimension 1 (default: 10)')
parser.add_argument('--emb_2D_d2', type=int, default=20,
                    help='ConvE embedding 2D shape dimension 2 (default: 20)')
parser.add_argument('--num_out_channels', type=int, default=32,
                    help='ConvE number of output channels of the convolution layer (default: 32)')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='ConvE kernel size (default: 3)')
parser.add_argument('--theta', type=float, default=0.2,
                    help='Threshold for sifting high-confidence facts (default: 0.2)')

# AcrE
parser.add_argument("--way", type=str, default='t',
                    help='Serial or Parallel')
parser.add_argument("--first_atrous", type=int, default=1,
                    help="First layer expansion coefficient")
parser.add_argument("--second_atrous", type=int, default=2,
                    help="Second layer expansion coefficient")
parser.add_argument("--third_atrous", type=int, default=5,
                    help="Third layer expansion coefficient")

# CompGCN
parser.add_argument('-gcn_dim', type=int, default=200,
                    help='Number of hidden units in GCN')
parser.add_argument('--num_gcn_layer', type=int, default=1,
                    help='Number of GCN Layers to use')
parser.add_argument('--gcn_dropout_rate', type=float, default=0.1,
                    help='GCN dropout rate (default: 0.1)')

# SACN
parser.add_argument('-gc1_emb_size', type=int, default=150,
                    help='The hidden dimension of GCN layer')

args = parser.parse_args()
