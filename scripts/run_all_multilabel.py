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
import label_data_multilabel, encode_datasets_multilabel, train_downstream_model_multilabel
import sys
sys.path.append('../keyclass_multilabel/')
import utils
import wandb
from datetime import datetime
import torch 

if __name__ == "__main__":
    parser_cmd = argparse.ArgumentParser()
    parser_cmd.add_argument('--config',
                            default='../config_files/config_mimic_unfiltered.yml',
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

    run = None
    use_wandb = True if args_cmd.use_wandb == 1 else False
    skip_self_training = True if args_cmd.skip_self_training == 1 else False

    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    target_descriptions = 'unfiltered' if 'unfiltered' in args_cmd.config else 'filtered'
    ngram_desc = f"{args['ngram_range'][0]}_{args['ngram_range'][1]}"

    # Define the experiment name
    if args['label_model'] == 'data_programming':
        # Example experiment name structure: mimic_filtered_lr_0.001_b_128_lf_300_dp_20230415_103000
        experiment_name = f"{args['dataset']}_{target_descriptions}_lr_{args['end_model_lr']}_b_{args['end_model_batch_size']}_lf_{args['topk']}_ngram_{ngram_desc}_dp_{timestamp}"
    elif args['label_model'] == 'majority_vote':
        # Example experiment name structure: mimic_filtered_lr_0.001_b_128_lf_300_mv_20230415_103000
        experiment_name = f"{args['dataset']}_{target_descriptions}_lr_{args['end_model_lr']}_b_{args['end_model_batch_size']}_lf_{args['topk']}_ngram_{ngram_desc}__mv_{timestamp}"
    else:
        print(f"Unsupported label model in config file: {args['label_model']}")
        sys.exit(1)

    # Define notes for logging
    notes = (
        f"Dataset: {args['dataset']} ({args.get('problem_type', 'single_label')})\n"
        f"Learning Rate: {args['end_model_lr']}\n"
        f"Batch Size: {args['end_model_batch_size']}\n"
        f"Labeling Functions (TopK): {args['topk']}\n"
        f"Label Model: {args['label_model']}\n"
        f"Evaluation Averaging: {args['average']}\n"
        f"Target Descriptions: {target_descriptions}\n"
        f"N-gram Range: {args['ngram_range']}"
    )

    # Weights & Biases setup (optional)
    if use_wandb:
        tag_dataset = args['dataset']
        tag_problem_type = args.get('problem_type', 'single_label')
        tag_lr = f"lr_{args['end_model_lr']}"
        tag_batch_size = f"batch_size_{args['end_model_batch_size']}"
        tag_number_lf = f"label_functions_{args['topk']}"
        tag_label_model = args['label_model']
        tag_average = f"avg_{args['average']}" 
        tag_target_descriptions = f"target_descriptions_{target_descriptions}"
        tag_ngram = f"ngram_{ngram_desc}"

        experiment_tags = [
            tag_dataset, tag_problem_type, tag_lr, tag_batch_size,
            tag_number_lf, tag_label_model, tag_average, tag_target_descriptions, tag_ngram
        ]
        if skip_self_training:
            experiment_tags.append("no_self_training")

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
                "base_encoder": args['base_encoder'],
                "average": args['average'],
                "problem_type": tag_problem_type,
                "target_descritions": target_descriptions,
                "ngram_range": args['ngram_range'],
            }
        )

    # Set up logging
    log_file = f"{args['log_path']}{experiment_name}.log"
    logger = utils.setup_logging(log_file)

    print(f"Experiment ID: {experiment_name}")
    
    print(f"Key configuration parameters:\n{notes}")

    print("Encoding Dataset")
    encode_datasets_multilabel.run(args_cmd, use_wandb, run, experiment_name)

    print("Labeling Data")
    try:
        label_data_multilabel.run(args_cmd, use_wandb, run, experiment_name)
    except torch.cuda.OutOfMemoryError as e:
        print(f"ERROR: CUDA Out of Memory during Label Model training!")
        print(e)
        if use_wandb and run:
            run.log({"status": "failed", "error": f"CUDA out of memory error during label model training: {e}"})
            run.finish(exit_code=1)
        sys.exit(1)  

    print("Training Model")
    results = train_downstream_model_multilabel.train(args_cmd, use_wandb, run, experiment_name, skip_self_training)

    if skip_self_training:
        print("Model Results (end-model WITHOUT self-training):")
    else:
        print("Model Results (self-trained end-model):")

    print(results)

    if use_wandb:
        run.finish()
