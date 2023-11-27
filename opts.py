import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--disable', type=bool, default=False)
    parser.add_argument('--sample_path', type=str, default='./samples')

    parser.add_argument('--data_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model_name', type=str, default='vgg16')

    parser.add_argument('--attack_name', type=str, default='standard', choices=['standard'])
    parser.add_argument('--trigger', type=str, default='4', help='trigger size for standard trigger')
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.02)

    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--early_epoch', type=int, default=5)
    parser.add_argument('--score', type=str, default='BAS',choices='[forgettingscore, gradnorm, BAS]')
    parser.add_argument('--alpha', type=float, default=0.5)

    parser.add_argument('--samples_idx', type=str, default='test_samples')

    opts = parser.parse_args()
    return opts
