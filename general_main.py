from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import torch
from utils.names_match_torch import methods
import os
import numpy as np
from utils.common import create_code_snapshot


def main(args):
    # print args recap
    print(args, end="\n\n")
    loss = torch.nn.CrossEntropyLoss()

    method = methods[args.method](args, loss, args.use_cuda)
    valid_acc, elapsed, ram_usage, ext_mem_sz, preds = method.train_model(tune=False)

    # directory with the code snapshot to generate the results
    sub_dir = 'submissions/' + args.sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(valid_acc), elapsed, np.average(ram_usage),
            np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for pred in preds:
            wf.write(str(pred) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="multi-task-nc",
                        choices=['ni', 'multi-task-nc', 'nic'])


    # Model
    parser.add_argument('--cls', type=str, default='densenet161',
                        choices=['resnet18', 'densenet161', 'efficientnetb5', 'efficientnetb4', 'densenet161complex', 'dense_freeze_till3', 'dense_freeze_till4', 'resnext101'])
    parser.add_argument('--method', type=str, default='task_mem',
                        choices=['task_mem', 'EWC', 'Tiny', 'multi_models', 'multi_models_smallFC']
                        )


    # Optimization
    parser.add_argument('--optimizer', dest='optimizer', default='SGD',
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--nesterov', type=bool, default=False,
                        help='nesterov')
    parser.add_argument('--amsgrad', type=bool, default=False,
                        help='amsgrad')
    parser.add_argument('--momentum', type=float, default=0,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')

    # Continual Learning

    parser.add_argument('--replay_examples', type=int, default=0,
                        help='data examples to keep in memory for each batch '
                             'for replay.')
    parser.add_argument('--replay_used', type=int, default=0,
                        help='data examples in memory to use for each batch '
                             'for replay.')
    parser.add_argument('--replay_epochs', type=int, default=1,
                        help='number of epochs for each batch, each epoch draw different memory  '
                             'for replay.')
    parser.add_argument('--review_size', type=int, default=20000,
                        help='number of mem samples to review by the end of training')
    parser.add_argument('--review_epoch', type=int, default=0,
                        help='number of epoch to review by the end of training')
    parser.add_argument('--review_lr_factor', type=float, default=0.5,
                        help='review lr decay factor')

    # ER
    parser.add_argument('--separate', dest='seperate', default=False, type=bool,
                        help='Train current data and memory separately (default: %(default)s)')

    parser.add_argument('--eps_mem_batch', dest='eps_mem_batch', default=10,
                        type=int,
                        help='Episode memory per batch (default: %(default)s)')

    # Misc
    parser.add_argument('--sub_dir', type=str, default="multi-task-nc",
                        help='directory of the submission file for this exp.')
    parser.add_argument('--verbose', type=bool, default=False,
                       help='print information or not')
    parser.add_argument('--preload_data', type=bool, default=False)

    #aug
    parser.add_argument('--aug', type=bool, default=True,
                        help='data augmentation')
    parser.add_argument('--aug_type', type=str, default='center_224',
                        help='data augmentation type')
    args = parser.parse_args()
    args.n_classes = 50

    args.use_cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if args.use_cuda else 'cpu'

    main(args)
