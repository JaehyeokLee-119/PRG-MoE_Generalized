import argparse
import json
import os
import random
import csv
from typing import List

import numpy as np
import torch

from module.trainer import LearningEnv


def set_random_seed(seed: int):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='This code is for ECPE task.')

    # Training Environment
    parser.add_argument('--gpus', default=[1])
    parser.add_argument('--num_process', default=int(os.cpu_count() * 0.8), type=int)
    parser.add_argument('--num_worker', default=6, type=int)
    parser.add_argument('--port', default=1234, type=int)

    parser.add_argument('--pretrained_model', default=None)
    parser.add_argument('--test', default=False)
    
    # Model Setting
    parser.add_argument('--encoder_name', default='bert-base-cased', type=str)
    parser.add_argument('--model_name', default='PRG_MoE')
    parser.add_argument('--unfreeze', default=10, type=int)    # 0: no freeze, n>0: # of unfreezed layers
    
    # Parameters
    parser.add_argument('--training_iter', default=15, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--learning_rate', default=5e-6, type=float)
    parser.add_argument('--patience', help='patience for Early Stopping', default=None, type=int)
    
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--n_speaker', help='the number of speakers', default=2, type=int)
    parser.add_argument('--n_emotion', help='the number of emotions', default=7, type=int)
    parser.add_argument('--n_cause', help='the number of causes', default=2, type=int)
    parser.add_argument('--n_expert', help='the number of causes', default=4, type=int)
    parser.add_argument('--guiding_lambda',help='the mixing ratio', default=0.6, type=float)
    parser.add_argument('--max_seq_len', help='the max length of each tokenized utterance', default=75, type=int)
    parser.add_argument('--contain_context', help='While tokenizing, previous utterances are contained or not', default=False)

    # Logging Setting
    parser.add_argument('--split_directory', default=None)
    parser.add_argument('--train_data', default="data/data_fold/data_0/dailydialog_train.json")
    parser.add_argument('--valid_data', default="data/data_fold/data_0/dailydialog_valid.json")
    parser.add_argument('--test_data', default="data/data_fold/data_0/dailydialog_test.json")
    parser.add_argument('--log_directory', default='logs', type=str)
    parser.add_argument('--data_label', help='the label that attaches to saved model', default='dailydialog_fold_0')

    # wandb setting
    parser.add_argument('--use_wandb', default=False)
    parser.add_argument('--project_name', default='log-large-type4-scheduler-type3', type=str)
    
    return parser.parse_args()

def test_preconditions(args: argparse.Namespace):
    if args.test:
        assert args.pretrained_model is not None, "For test, you should load pretrained model."


def main():
    args = parse_args()
    test_preconditions(args)
    set_random_seed(77)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in args.gpus])

    # # Original Dataset (With 5 folds)
    # train_data_list = [
    #     'data/data_fold/data_0/dailydialog_train.json',
    #     * [f'data/data_fold/data_{fold_}/data_{fold_}_train.json' for fold_ in range(1, 5)]
    # ]
    # valid_data_list = [
    #     'data/data_fold/data_0/dailydialog_valid.json',
    #     * [f'data/data_fold/data_{fold_}/data_{fold_}_valid.json' for fold_ in range(1, 5)]
    # ]
    # test_data_list = [
    #     'data/data_fold/data_0/dailydialog_test.json',
    #     * [f'data/data_fold/data_{fold_}/data_{fold_}_test.json' for fold_ in range(1, 5)]
    # ]

    # Mini Dataset (동작 테스트)
    train_data_list = ['data/data_mini/dailydialog_train.json']
    valid_data_list = ['data/data_mini/dailydialog_valid.json']
    test_data_list = ['data/data_mini/dailydialog_test.json']

    # # Original Dataset (1 fold)
    # train_data_list = ['data/data_fold/data_0/dailydialog_train.json']
    # valid_data_list = ['data/data_fold/data_0/dailydialog_valid.json']
    # test_data_list = ['data/data_fold/data_0/dailydialog_test.json']

    data_label = ['-original_fold']

    model_name = 'PRG_MoE_General'

    # encoder_name_list = ['bert-base-cased',
    #                      'j-hartmann/emotion-english-distilroberta-base',
    #                      'j-hartmann/emotion-english-roberta-large']
    # log_directory_list = ['logs/train-testing_General(5e-5, 40ep, bert-base-cased)',
    #                       'logs/train-testing_General(5e-5, 40ep, distillroberta-emo)',
    #                       'logs/train-testing_General(5e-5, 40ep, large-emo)']

    encoder_name_list = ['j-hartmann/emotion-english-roberta-large',
                         'j-hartmann/emotion-english-roberta-large',
                         'j-hartmann/emotion-english-roberta-large',
                         ]
    
    lrs = [4e-4, 5e-5, 8e-6]

    for tr, va, te, dl in zip(train_data_list, valid_data_list, test_data_list, data_label):
        args.train_data, args.valid_data, args.test_data, args.data_label = tr, va, te, dl

        for en, lr in zip(encoder_name_list, lrs):
            args.encoder_name = en
            args.learning_rate = lr
            args.model_name = model_name
            log_d = f'logs/testing-train-unfreeze[{args.unfreeze}](lr {str(lr)}, {args.training_iter} ep, large-emo)'
            args.log_directory = log_d + dl

            trainer = LearningEnv(**vars(args))
            trainer.run(**vars(args))

            del trainer

    '''
    train_data_list = [
        f'data_fold_test_IEMOCAP/data_{fold_}/data_{fold_}_train.json' for fold_ in range(0, 5)]
    valid_data_list = [
        f'data_fold_test_IEMOCAP/data_{fold_}/data_{fold_}_valid.json' for fold_ in range(0, 5)]
    test_data_list = [
        f'data_fold_test_IEMOCAP/data_{fold_}/data_{fold_}_test.json' for fold_ in range(0, 5)]
    data_label = [f'-data_{fold_}_IEMOCAP' for fold_ in range(0, 5)]

    for tr, va, te, dl in zip(train_data_list, valid_data_list, test_data_list, data_label):
        args.train_data, args.valid_data, args.test_data, args.data_label = tr, va, te, dl

        for en, log_d in zip(encoder_name_list, log_directory_list):
            args.encoder_name = en
            args.model_name = model_name
            args.log_directory = log_d + dl

            trainer = LearningEnv(**vars(args))
            trainer.run(**vars(args))

            del trainer
    '''


if __name__ == "__main__":
    main()
