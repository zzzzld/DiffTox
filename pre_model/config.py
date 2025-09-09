import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="./output/", help="the path of output model")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="the max number of epoch 2000")
    parser.add_argument("-s", "--seed", type=int, default=1, help="random seed")

    parser.add_argument('--hidden_feats', type=list, default=[64,384], help="the size of node representations after the i-th GAT layer")
    parser.add_argument('--dropout', type=float, default=0.6, help="dropout probability")
    parser.add_argument('--lr', type=float, default=0.001,  help='learning rate0.0001')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--batch_size', type=float, default=64)#cav 64 nav 256 herg 1024
    parser.add_argument('--sample_neighbor', type=bool, default=False, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=False, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=False, help='whether perform structure learning')
    parser.add_argument('--patience', type=int, default=50, help="Patience for early stopping")
    parser.add_argument('--head', type=int, default=10, help="the head size of attention")
    return parser.parse_args()