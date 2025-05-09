# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import label_data, encode_datasets, train_downstream_model
import sys
sys.path.append('../keyclass/')
import utils
import wandb
from datetime import datetime

if __name__ == "__main__":
    parser_cmd = argparse.ArgumentParser()
    parser_cmd.add_argument('--config',
                            default='../config_files/config_dbpedia.yml',
                            help='Configuration file')
    parser_cmd.add_argument('--random_seed',
                            default=0,
                            type=int,
                            help="Random Seed")
    parser_cmd.add_argument('--use_wandb',
                            default=0,
                            type=int,
                            help="If set to 1 will use Weights and Biases to log the runs"
                            )
    parser_cmd.add_argument('--skip_self_training',
                        default=0,
                        type=int,
                        help="If set to 1 the script will bypass the self-training runs"
                        )
    args_cmd = parser_cmd.parse_args()

    print(f'Loading configuration file: {args_cmd.config}')

    args = utils.Parser(config_file_path=args_cmd.config).parse()

    if args['dataset'] == 'mimic' or args['problem_type'] == 'multi_label':
        print(f"Original implementation of Keyclass does not include multi-label support. Use run_all_multilabel.py (experimental)")
        sys.exit(1)

    run = None
    use_wandb = True if args_cmd.use_wandb == 1 else False
    skip_self_training = True if args_cmd.skip_self_training == 1 else False

    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")

    # Define the experiment name
    if args['label_model'] == 'data_programming':
        experiment_name = f"{args['dataset']}_lr_{args['end_model_lr']}_b_{args['end_model_batch_size']}_lf_{args['topk']}_dp_{timestamp}"
    elif args['label_model'] == 'majority_vote':
        experiment_name = f"{args['dataset']}_lr_{args['end_model_lr']}_b_{args['end_model_batch_size']}_lf_{args['topk']}_mv_{timestamp}"
    else:
        print(f"Unsupported label model in config file: {args['label_model']}")
        sys.exit(1)

    notes = f"Dataset: {args['dataset']}\nLearning Rate: {args['end_model_lr']}\nBatch Size: {args['end_model_batch_size']}\nLabeling Functions: {args['topk']}\nLabel Model: {args['label_model']}"

    # Weights & Biases setup 
    if use_wandb:
        tag_dataset = args['dataset']
        tag_lr = f"lr_{args['end_model_lr']}"
        tag_batch_size = f"batch_size_{args['end_model_batch_size']}"
        tag_number_lf = f"label_functions_{args['topk']}"
        tag_label_model = args['label_model']

        if skip_self_training:
            experiment_tags =  [tag_dataset, tag_lr, tag_batch_size, tag_number_lf, tag_label_model, "no_self_training" ]
        else:
            experiment_tags =  [tag_dataset, tag_lr, tag_batch_size, tag_number_lf, tag_label_model ]

        run = wandb.init(
            project = 'dl4h-reproduce-keyclass',
            name = experiment_name,
            notes = notes,
            tags = experiment_tags,
            config={
                "dataset": args['dataset'],
                "label_model": args['label_model'],
                "label_model_lr": args['label_model_lr'],
                "label_model_n_epochs": args['label_model_n_epochs'],
                "end_model_lr": args['end_model_lr'],
                "end_model_patience": args['end_model_patience'],
                "end_model_batch_size": args['end_model_batch_size'],
                "self_train_batch_size": args['self_train_batch_size'],
                "self_train_lr": args['self_train_lr'],
                "self_train_patience": args['self_train_patience'],
                "topk_label_functions": args['topk'],
                "base_encoder": args['base_encoder']
            }
        )

    # Set up logging
    log_file = f"{args['log_path']}{experiment_name}.log"
    logger = utils.setup_logging(log_file)

    print(f"Experiment ID: {experiment_name}")
    
    print(f"Key configuration parameters:\n{notes}")

    print("Encoding Dataset")
    encode_datasets.run(args_cmd, use_wandb, run, experiment_name)

    print("Labeling Data")
    label_data.run(args_cmd, use_wandb, run, experiment_name)

    print("Training Model")
    results = train_downstream_model.train(args_cmd, use_wandb, run, experiment_name, skip_self_training)

    if skip_self_training:
        print("Model Results (end-model WITHOUT self-training):")
    else:
        print("Model Results (self-trained end-model):")

    print(results)

    if use_wandb:
        run.finish()
