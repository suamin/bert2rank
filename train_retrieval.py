# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import MarginRankingLoss
from tensorboardX import SummaryWriter
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from knrm import BertKnrm
from data_utils import (
    processors,
    convert_train_examples_to_features, 
    convert_eval_examples_to_features
)
from data_utils import read_bioasq_json_file
import pickle
import json
import time
import config


logger = logging.getLogger(__name__)


def features(args, processor, set_type, tokenizer, test_batch_num=None):
    """Prepare features for training, evaluation or predictions
    with BERT.
    
    Parameters
    ----------
    args : argparse.ArgumentParser
        Argument parser object as initialized with BERT.
    
    processor : DataProcessor
        Data processing class for converting texts to
        required format.
    
    set_type : str
        Data set type, one of ["train", "dev", "test"].
    
    test_batch_num : int, optional
        Test batch number of BioASQ in 1 to 5. Required only
        when ``set_type`` is "test".
    
    Returns
    -------
    all_input_ids : a torch.LongTensor
        Batch of input sequences converted to their ids.
        Shape of tensor depends on ``set_type``, if "train"
        then shape is [batch size x 2 x sequence length]
        (2 is because each example is a pair of <<query `i`, 
        relevant document `j`>, <query `i`, irrelevant document
        `k`>> to rightly pass for ``MarginRankingLoss``).
        In case of "dev" and "test" it is simply [batch size x
        sequence length] since we only have <query, document>
        pairs at inference.
    
    all_segment_ids : a torch.LongTensor
        Types indices selected in [0, 1]. Type 0 corresponds 
        to a `sentence A` and type 1 corresponds to a `sentence B` 
        token (see BERT paper for more details). Shapes follow
        same rules as ``all_input_ids``.
    
    all_input_mask : a torch.LongTensor
        Mask with values in [0, 1]. Shapes follow same rules 
        as ``all_input_ids``.
    
    """
    # Prepare data loader
    
    if set_type == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif set_type == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        # should be from 1 to 5
        if not test_batch_num:
            raise ValueError("test batch number required")
        examples = processor.get_test_examples(args.data_dir, test_batch_num)
    
    cache_fname_base = '{0}_{1}_{2}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(),
        str(args.max_seq_length),
        str("alinger")
    )
    if set_type == "test":
        cache_fname_base += "_6b{}".format(test_batch_num)
    cached_features_file = os.path.join(
        args.data_dir, 
        "{}_{}".format(set_type, cache_fname_base)
    )
    
    try:
        with open(cached_features_file, "rb") as reader:
            if set_type == "train":
                features = pickle.load(reader)
            else:
                features, guids = pickle.load(reader)
    except:
        if set_type == "train":
            features = convert_train_examples_to_features(
                examples, args.max_seq_length, tokenizer
            )
        else:
            features, guids = convert_eval_examples_to_features(
                examples, args.max_seq_length, tokenizer
            )
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info(
                "  Saving %s features into cached file %s" 
                % (set_type, cached_features_file)
            )
            with open(cached_features_file, "wb") as writer:
                if set_type == "train":
                    pickle.dump(features, writer)
                else:
                    pickle.dump((features, guids), writer)
    
    #
    # feature set in case of *train* is composed of:
    # - <question, +ve doc>, <question, -ve doc>
    #   all_* : -1 x 2 x max_seq_length    
    # 
    if set_type == "train":
        all_input_ids = torch.cat([
            to_tensor([f.input_ids for f in f_set], torch.long).unsqueeze(0) 
            for f_set in features
        ], dim=0)
        all_input_mask = torch.cat([
            to_tensor([f.input_mask for f in f_set], torch.long).unsqueeze(0) 
            for f_set in features
        ], dim=0)
        all_segment_ids = torch.cat([
            to_tensor([f.segment_ids for f in f_set], torch.long).unsqueeze(0)
            for f_set in features
        ], dim=0)
        return (all_input_ids, all_segment_ids, all_input_mask), len(examples)
    else:
        all_input_ids = to_tensor([f.input_ids for f in features], torch.long)
        all_input_mask = to_tensor([f.input_mask for f in features], torch.long)
        all_segment_ids = to_tensor([f.segment_ids for f in features], torch.long)
        ids_map = {}
        for idx, guid in enumerate(guids):
            ids_map[guid] = idx
        all_ids = to_tensor([ids_map[guid] for guid in guids], torch.long)
        return (all_input_ids, all_segment_ids, all_input_mask, all_ids, ids_map), len(examples)


def features(args, processor, set_type, tokenizer, test_batch_num=None):
    # Prepare data loader
    
    if set_type == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif set_type == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        # should be from 1 to 5
        if not test_batch_num:
            raise ValueError("test batch number required")
        examples = processor.get_test_examples(args.data_dir, test_batch_num)
    
    cache_fname_base = '{0}_{1}_{2}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(),
        str(args.max_seq_length),
        str("alinger")
    )
    if set_type == "test":
        cache_fname_base += "_6b{}".format(test_batch_num)
    cached_features_file = os.path.join(
        args.data_dir, 
        "{}_{}".format(set_type, cache_fname_base)
    )
    
    try:
        with open(cached_features_file, "rb") as reader:
            if set_type == "train":
                features = pickle.load(reader)
            else:
                features, guids = pickle.load(reader)
    except:
        if set_type == "train":
            features = convert_train_examples_to_features(
                examples, args.max_seq_length, tokenizer
            )
        else:
            features, guids = convert_eval_examples_to_features(
                examples, args.max_seq_length, tokenizer
            )
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info(
                "  Saving %s features into cached file %s" 
                % (set_type, cached_features_file)
            )
            with open(cached_features_file, "wb") as writer:
                if set_type == "train":
                    pickle.dump(features, writer)
                else:
                    pickle.dump((features, guids), writer)
    
    #
    # feature set in case of *train* is composed of:
    # - <question, +ve doc>, <question, -ve doc>
    #   all_* : -1 x 2 x max_seq_length    
    # 
    if set_type == "train":
        all_input_ids = torch.cat([
            to_tensor([f.input_ids for f in f_set], torch.long).unsqueeze(0) 
            for f_set in features
        ], dim=0)
        all_input_mask = torch.cat([
            to_tensor([f.input_mask for f in f_set], torch.long).unsqueeze(0) 
            for f_set in features
        ], dim=0)
        all_segment_ids = torch.cat([
            to_tensor([f.segment_ids for f in f_set], torch.long).unsqueeze(0)
            for f_set in features
        ], dim=0)
        return (all_input_ids, all_segment_ids, all_input_mask), len(examples)
    else:
        all_input_ids = to_tensor([f.input_ids for f in features], torch.long)
        all_input_mask = to_tensor([f.input_mask for f in features], torch.long)
        all_segment_ids = to_tensor([f.segment_ids for f in features], torch.long)
        ids_map = {}
        for idx, guid in enumerate(guids):
            ids_map[guid] = idx
        all_ids = to_tensor([ids_map[guid] for guid in guids], torch.long)
        return (all_input_ids, all_segment_ids, all_input_mask, all_ids, ids_map), len(examples)


def to_tensor(array, dtype):
    return torch.tensor(array, dtype=dtype)


def train(args, model, processor, tokenizer, device, n_gpu):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    data, num_examples = features(args, processor, "train", tokenizer)
    data = TensorDataset(*data)
    
    if args.local_rank == -1:
        sampler = RandomSampler(data)
    else:
        sampler = DistributedSampler(data)
    
    data_loader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
    step_size = args.gradient_accumulation_steps * args.num_train_epochs
    num_train_optimization_steps = len(data_loader) // step_size
    
    # Prepare optimizer
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer 
                if not any(nd in n for nd in no_decay)
            ], 
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer 
                if any(nd in n for nd in no_decay)
            ], 
            'weight_decay': 0.0
        }
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from "
                "https://www.github.com/nvidia/apex to use "
                "distributed and fp16 training."
            )
        
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)
    
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    
    model.train()
    loss_fct = MarginRankingLoss(margin=args.margin)
    ckpt_num = 0
    eval_results_history = []
    best = 0.
    best_props = {}
    eval_result = None
    no_improvement = 0
    t = time.time()
    
    try:
        for num_epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            if no_improvement > args.tolerance:
                logger.info("No improvement in last %d evaluations, early stopping")
                logger.info("epoch: {} | nb_tr_steps: {} | global_step: {} | tr_loss: {}".format(
                    num_epoch, nb_tr_steps, global_step, tr_loss))
            
            for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
                print(nb_tr_steps)
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, mask_ids = batch
                
                # <question, +ve doc> pairs
                input_ids_qp, segment_ids_qp, input_mask_qp = \
                input_ids[:, 0, :], segment_ids[:, 0, :], mask_ids[:, 0, :]
                # <question, -ve doc> pairs
                input_ids_qn, segment_ids_qn, input_mask_qn = \
                input_ids[:, 1, :], segment_ids[:, 1, :], mask_ids[:, 1, :]
                
                pos_scores = model(input_ids_qp, segment_ids_qp, input_mask_qp)
                neg_scores = model(input_ids_qn, segment_ids_qn, input_mask_qn)
                
                # y all 1s to indicate positive should be higher
                y = torch.ones(len(pos_scores)).float().to(device)
                loss = loss_fct(pos_scores, neg_scores, y)
                if nb_tr_steps % 10 == 0 and nb_tr_steps != 0:
                    logger.info("+ve scores : %r" % pos_scores)
                    logger.info("-ve scores : %r" % neg_scores)
                    logger.info("Train step loss : %0.5f" % loss.item())
                    if global_step > 0:
                        logger.info("Train total loss : %0.5f" % (tr_loss/global_step))
                
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles 
                        # this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step, args.warmup_proportion
                        )
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0]:
                        tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', loss.item(), global_step)
                
                if nb_tr_steps % config.eval_every_step == 0 and nb_tr_steps != 0:
                    eval_result = eval(
                        args, model, processor, 
                        tokenizer, device, 
                        tr_loss, global_step
                    )
                    if eval_result["f1"] >= best:
                        save(
                            model, "%s_%0.3f_%0.3f_%0.3f" % (
                                args.model_name,
                                eval_result["precision"], 
                                eval_result["recall"],
                                eval_result["f1"]
                            ) , 
                            args, tokenizer, ckpt_num
                        )
                        best = eval_result["f1"]
                        best_props["num_epoch"] = num_epoch
                        best_props["nb_tr_steps"] = nb_tr_steps
                        best_props["tr_loss"] = tr_loss/global_step
                        best_props["ckpt_num"] = ckpt_num
                        best_props["global_step"] = global_step
                        best_props["eval_result"] = eval_result
                        with open(os.path.join(config.output_dir, "best.json"), "w") as wf:
                            json.dump(best_props, wf, indent=2)
                        
                        # make predictions with best model
                        for i in range(1, 6):
                            predict(args, model, processor, tokenizer, device, i)
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    
                    ckpt_num += 1
                    eval_results_history.append((ckpt_num, eval_result))
       
    except KeyboardInterrupt:
        logger.info("Training interrupted!")
        if eval_result is not None:
            save(
                model, "%s_%0.3f_%0.3f_%0.3f_interrupted" % (
                    args.model_name,
                    eval_result["precision"], 
                    eval_result["recall"],
                    eval_result["f1"]
                ) , 
                args, tokenizer, ckpt_num
            )
    
    t = time.time() - t
    logger.info("Training took %0.3f seconds" % t)
    loss = tr_loss / global_step
    logger.info("Final training loss %0.5f" % loss)
    logger.info("Best F1-score on eval set : %0.3f" % best)
    logger.info("***** Eval best props *****")
    for key in sorted(best_props.keys()):
        if key != "eval_result":
            logger.info("  %s = %s", key, str(best_props[key]))
        else:
            for eval_key in sorted(best_props[key].keys()):
                logger.info("  %s = %s", eval_key, str(best_props[key][eval_key]))
    
    with open(os.path.join(config.output_dir, "eval_results_history.pkl"), "wb") as wf:
        pickle.dump(eval_results_history, wf)


def eval(args, model, processor, tokenizer, device, tr_loss=None, global_step=None):
    data, num_examples = features(args, processor, "dev", tokenizer)
    all_input_ids, all_segment_ids, all_input_mask, all_ids, ids_map = data
    data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_ids)
    
    if args.local_rank == -1:
        sampler = RandomSampler(data)
    else:
        sampler = DistributedSampler(data)
    
    data_loader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)
    
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    preds = []
    preds_ids = []
    with open(os.path.join(config.data_dir, "dev_id2rels.pkl"), "rb") as rf:
        id2true = pickle.load(rf)
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids, segment_ids, mask_ids, ids = batch
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        mask_ids = mask_ids.to(device)
        
        with torch.no_grad():
            scores = model(input_ids, segment_ids, mask_ids)
            if len(preds) == 0:
                preds.append(scores.detach().cpu().numpy())
                preds_ids.append(ids.numpy())
            else:
                preds[0] = np.append(preds[0], scores.detach().cpu().numpy(), axis=0)
                preds_ids[0] = np.append(preds_ids[0], ids.numpy(), axis=0)
    
    preds = preds[0]
    preds_ids = preds_ids[0]
    id2preds = {}
    rev_idsmap = {v:k for k, v in ids_map.items()}
    for i, j in zip(preds, preds_ids):
        question_id, doc_id = rev_idsmap[j].split("_")
        if i <= 0:
            continue
        if question_id in id2preds:
            id2preds[question_id].append(doc_id)
        else:
            id2preds[question_id] = [doc_id]
    # take top-10 only
    id2preds = {k:v[:10] for k, v in id2preds.items()}
    all_ps = []
    all_rs = []
    all_f1s = []
    for qid in id2preds:
        if not qid in id2true:
            continue
        y_pred = set([i for i in id2preds[qid]])
        y_true = set(id2true[qid])
        common = y_pred.intersection(y_true)
        diff = y_pred - y_true
        tps = len(common)
        fps = len(diff)
        all_tps = len(y_true)
        if (tps + fps) == 0:
            p = 0
        else:
            p = tps / (tps + fps)
        all_ps.append(p)
        if all_tps == 0:
            r = 0
        else:
            r = tps / all_tps
        all_rs.append(r)
        if (p + r) == 0:
            f1 = 0.
        else:
            f1 = 2 * ((p * r) / (p + r))
        all_f1s.append(f1)
    result = {}
    result["precision"] = sum(all_ps) / len(all_ps)
    result["recall"] = sum(all_rs) / len(all_rs)
    result["f1"] = sum(all_f1s) / len(all_f1s)
        
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    
    return result


def predict(args, model, processor, tokenizer, device, test_batch_num):
    data, num_examples = features(args, processor, "test", tokenizer, test_batch_num)
    all_input_ids, all_segment_ids, all_input_mask, all_ids, ids_map = data
    data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_ids)
    
    if args.local_rank == -1:
        sampler = RandomSampler(data)
    else:
        sampler = DistributedSampler(data)
    
    data_loader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)
    
    logger.info("***** Running predictions *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    preds = []
    preds_ids = []
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        input_ids, segment_ids, mask_ids, ids = batch
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        mask_ids = mask_ids.to(device)
        
        with torch.no_grad():
            scores = model(input_ids, segment_ids, mask_ids)
            if len(preds) == 0:
                preds.append(scores.detach().cpu().numpy())
                preds_ids.append(ids.numpy())
            else:
                preds[0] = np.append(preds[0], scores.detach().cpu().numpy(), axis=0)
                preds_ids[0] = np.append(preds_ids[0], ids.numpy(), axis=0)
    
    preds = preds[0]
    preds_ids = preds_ids[0]
    id2preds = {}
    rev_idsmap = {v:k for k, v in ids_map.items()}
    for i, j in zip(preds, preds_ids):
        question_id, doc_id = rev_idsmap[j].split("_")
        doc_id = "http://www.ncbi.nlm.nih.gov/pubmed/" + doc_id
        if i <= 0:
            continue
        if question_id in id2preds:
            id2preds[question_id].append(doc_id)
        else:
            id2preds[question_id] = [doc_id]
    
    # open template file and write predictions
    batch_name = "6b{}".format(test_batch_num)
    test_qas = read_bioasq_json_file(
        os.path.join(config.data_dir, "test_template_{}.json".format(batch_name))
    )
    for idx, qas in enumerate(test_qas):
        if qas["id"] in id2preds:
            qas["documents"] = id2preds[qas["id"]]
    test_qas = {
        "questions": test_qas
    }
    wf = open(
        os.path.join(config.output_dir, "test_predictions_{}.json".format(batch_name)),
        "w",
        encoding="utf-8",
        errors="ignore"
    )
    json.dump(test_qas, wf, indent=2)
    wf.close()


def save(model, model_name, args, tokenizer, ckpt_num):
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(
        args.output_dir, model_name + "_" + str(ckpt_num)
    )
    output_config_file = os.path.join(
        args.output_dir, model_name + "_" + str(ckpt_num) + "_" + "config.json"
    )
    torch.save(model.state_dict(), output_model_file)
    model.bert.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the model (e.g. vanBioBERT).")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--use_knrm",
                        action='store_true',
                        help="Use K-NRM instead of BERT [CLS] for retrieval.")
    parser.add_argument("--tolerance",
                        default=3,
                        type=int,
                        help="Number of no improvement evaluation cycles to stop training.")
    parser.add_argument("--margin",
                        default=1.0,
                        type=float,
                        help="Margin to use in MarginRankingLoss.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = torch.device("cuda:0")
    n_gpu = 1
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict` must be True.")
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    
    task_name = args.task_name.lower()
    
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    
    processor = processors[task_name]()
    
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, 
        do_lower_case=args.do_lower_case
    )
    model = BertKnrm.from_pretrained(args.bert_model, use_knrm=args.use_knrm,
        last_layer_only=False, N=12, method="selfattn")
    
    if args.local_rank == 0:
        torch.distributed.barrier()
    
    if args.fp16:
        model.half()
    
    model.to(device)
    if args.use_knrm:
        model.to_device(device)
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
       
    if args.do_train:
        try:
            train(args, model, processor, tokenizer, device, n_gpu)
        except KeyboardInterrupt:
            sys.exit(1)
    
    ### Saving best-practices: if you use defaults names for the model, you 
    ### can reload it using from_pretrained()
    ### Example:
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        save(model, args.model_name +"_end", args, tokenizer, "_end")
    else:
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, 
            do_lower_case=args.do_lower_case
        )
        model = BertKnrm.from_pretrained(args.bert_model)
    
    model.to(device)
    
    ### Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval(args, model, processor, tokenizer, device)
    
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        for i in range(1, 6):
            predict(args, model, processor, tokenizer, device, i)


if __name__ == "__main__":
    main()

"""

export BIO_BERT=/raid/data/saam01/pretrained/bert/biobert_v1.0_pubmed_pmc
export BIO_BERT=/raid/data/saam01/bioasq6b_phaseA/output/pubmed_pmc_470k
export DATA_DIR=/raid/data/saam01/bioasq6b_phaseA/data
export OUT_DIR=/raid/data/saam01/bioasq6b_phaseA/output


export BIO_BERT=/home/mlt/saad/projects/bioasq-task7b/tmp/bert_knrm/output/biobert_squad_tuned
export DATA_DIR=/home/mlt/saad/projects/bioasq-task7b/tmp/bert_knrm/data
export OUT_DIR=/home/mlt/saad/projects/bioasq-task7b/tmp/bert_knrm/output/biobert_squad_tuned

# train

python train_retrieval.py \
    --bert_model $BIO_BERT \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --overwrite_output_dir \
    --task_name retrieval \
    --model_name vanBioBERT \
    --max_seq_length 256 \
    --learning_rate 2e-5 \
    --train_batch_size 6 \
    --num_train_epochs 1.0 \
    --margin 1.0 \
    --use_knrm \
    --tolerance 2 \
    --do_train \
    --eval_batch_size 128 \
    --warmup_proportion 0.1 \
    --seed 2019


6B1     AP    Recall     F      MAP    GMAP
=============================================
*     0.4034  0.6613  0.4237  0.2340  0.0160  (+)
**    0.3535  0.6617  0.3964  0.2344  0.0152
***   0.3876  0.6591  0.4119  0.2385  0.0164
^        -       -       -       -       -

6B2     AP    Recall     F      MAP    GMAP
=============================================
*     0.4328  0.6664  0.4226  0.2411  0.0332
**    0.3825  0.6628  0.3931  0.2250  0.0292
***   0.4561  0.6460  0.4317  0.2417  0.0286
^     0.5091  0.5530  0.4493  0.2070  0.0508  (+)

6B3     AP    Recall     F      MAP    GMAP
=============================================
*     0.4341  0.6110  0.4394  0.3564  0.0152
**    0.4422  0.6121  0.4456  0.3553  0.0159
***   0.4174  0.6156  0.4250  0.3666  0.0153
^     0.5926  0.5169  0.5000  0.2189  0.0563  (+)

6B4     AP    Recall     F      MAP    GMAP
=============================================
*     0.3211  0.6540  0.3739  0.2187  0.0086
**    0.3203  0.6527  0.3707  0.2268  0.0091
***   0.3014  0.6507  0.3493  0.2214  0.0085
^     0.4245  0.5180  0.4052  0.1615  0.0150  (+)

6B5     AP    Recall     F      MAP    GMAP
=============================================
*     0.3977  0.6321  0.3947  0.2216  0.0195
**    0.4219  0.6314  0.4021  0.2578  0.0208  (+)
***   0.3985  0.6399  0.3875  0.2628  0.0213
^     0.2702  0.3591  0.2552  0.1781  0.0077

Average of 6B2 - 6B5 (we exclude 6B1 where AUEB-NLP-5 did not
submitted to make comparison fair).

AVG     AP    Recall     F      MAP    GMAP
=============================================
*     0.3964  0.6408  0.4076  0.2594  0.0191
**    0.3917  0.6397  0.4028  0.2244  0.0187
***   0.3933  0.6380  0.3983  0.2217  0.0184
^     0.4491  0.4867  0.4016  0.1250  0.0324
---------------------------------------------
W       ^       *       *       *       ^
---------------------------------------------
OW                   * 

Where,
*   = vanilla BioBERT intialized with SQuAD 1.1 fine-tuning
**  = vanilla BioBERT intialized with SQuAD 2.0 fine-tuning
*** = vanilla BioBERT
^   = AUEB-NLP-5
(+) = Best performing system in terms of F measure
W   = Winner per metric
OW  = Overall winning system based on number of times it won per metric


Layer-wise performance evaluation on test batch 3 using **
----------------------------------------------------------

L#      AP    Recall     F      MAP    GMAP
============================================
L05 - 0.3637  0.2341  0.2322  0.1449  0.0010
L06 - 0.3809  0.5946  0.3910  0.3207  0.0142
L07 - 0.3142  0.6125  0.3585  0.2779  0.0122
L08 - 0.3267  0.6077  0.3675  0.2972  0.0132
L09 - 0.3114  0.6075  0.3494  0.2911  0.0126
L10 - 0.3977  0.6079  0.4076  0.3454  0.0150
L11 - 0.4451  0.6080  0.4454  0.3505  0.0157
L12 - 0.4422  0.6121  0.4456  0.3530  0.0164


"""