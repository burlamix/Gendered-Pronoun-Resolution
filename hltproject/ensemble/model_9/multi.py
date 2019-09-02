from __future__ import absolute_import, division, print_function

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

from pathlib import Path

from bson import ObjectId

import pdb


import collections
import json
import logging
import math
import os
import random
import sys
from io import open

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)

from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertForMultipleChoice, BertForPreTraining, BertConfig, BertModel, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


#test_class_labels = [get_class_label(aco, bco) for aco, bco in zip(test_df['A-coref'], test_df['B-coref'])]


def get_class_label(a_coref, b_coref):
    if a_coref:
        return 0
    elif b_coref:
        return 1
    return 2

def get_gender(pronoun):
    gender_mapping = {'he': 0, 'his': 0, 'him': 0, 
                      'she': 1, 'her': 1, 'hers': 1}
    return gender_mapping.get(pronoun.lower(), 1)

class SwagExample(object):
    def __init__(self,
                 swag_id,
                 context_sentence,
                 ending_0,
                 ending_1,
                 ending_2,
                 a_offset,
                 b_offset,
                 pronoun_offset,
                 label = None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.endings = [
            ending_0,
            ending_1,
            ending_2
        ]
        self.a_offset = a_offset
        self.b_offset = b_offset
        self.pronoun_offset = pronoun_offset
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "swag_id: {}".format(self.swag_id),
            "context_sentence: {}".format(self.context_sentence),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2])
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)

class SwagInputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'pronoun_ids': pronoun_ids
            }
            for _, input_ids, input_mask, segment_ids, pronoun_ids in choices_features
        ]
        self.label = label

# A good chunk of the code below is borrowed from: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_swag.py
# TODO: "token_pronoun_embeddings" aren't being used here. That is left for future work!
class BertSwagRunner:
    def __init__(self, dev_df, val_df, test_df, bert_model = 'bert-large-uncased', do_lower_case = True, learning_rate = 1e-5,  
                  num_train_epochs = 2, max_seq_length = 300, train_batch_size = 4, predict_batch_size = 4, warmup_proportion = 0.1,
                                  num_choices=3):
        #self.dev_df = self.extract_target(dev_df)
        #self.val_df = self.extract_target(val_df)
        #self.test_df = test_df #self.extract_target(test_df)

        # Custom parameters
        self.bert_model = bert_model
        self.do_lower_case = do_lower_case
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size

        # Default parameters
        self.predict_batch_size = predict_batch_size
        
        self.seed = 42
        self.warmup_proportion = warmup_proportion

        self.local_rank = -1
        self.gradient_accumulation_steps = 1
        self.loss_scale = 0
        self.version_2_with_negative = False
        self.fp16 = False
        self.no_cuda = False
        self.verbose_logging = False
        
        self.num_choices = num_choices
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")

        logger.info("device: {} distributed training: {}, 16-bits training: {}".format(
            self.device, bool(self.local_rank != -1), self.fp16))

        self.train_batch_size = self.train_batch_size // self.gradient_accumulation_steps

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case, never_split=("[A]", "[B]", "[P]"))
        # These tokens are not actually used, so we can assign arbitrary values.
        self.tokenizer.vocab["[A]"] = -1
        self.tokenizer.vocab["[B]"] = -1
        self.tokenizer.vocab["[P]"] = -1
        
    def extract_target(self, df):
        df['target'] = [get_class_label(aco, bco) for aco, bco in zip(df['A-coref'], df['B-coref'])]
        df['gender'] = df['Pronoun'].transform(get_gender)

        return df
        
    def row_to_swag_example(self, row, is_training):
        json_dict = {}

        label = None
        if is_training:
            if row['A-coref']:
                label = 0
            elif row['B-coref']:
                label = 1
            else:
                label = 2
        pronoun = row['Pronoun']
        swag_id = str(ObjectId())
        context_sentence = row['Text'] + " " + f"{pronoun} is"

        return SwagExample(
                swag_id = swag_id,
                context_sentence = context_sentence,
                ending_0 = row['A'],
                ending_1 = row['B'],
                ending_2 = 'other',
                a_offset = row['A-offset'],
                b_offset = row['B-offset'],
                pronoun_offset = row['Pronoun-offset'],
                label = label
            )
    
    def insert_tag(self, text, a_offset, b_offset, pronoun_offset):
        """Insert custom tags to help us find the position of A, B, and the pronoun after tokenization."""
        to_be_inserted = sorted([
            (a_offset, " [A] "),
            (b_offset, " [B] "),
            (pronoun_offset, " [P] ")
        ], key=lambda x: x[0], reverse=True)
        text = text
        for offset, tag in to_be_inserted:
            text = text[:offset] + tag + text[offset:]
        return text

    def tokenize_with_offsets(self, text, tokenizer):
        """Returns a list of tokens and the positions of A, B, and the pronoun."""
        entries = {}
        final_tokens = []
        for token in tokenizer.tokenize(text):
            if token in ("[A]", "[B]", "[P]"):
                entries[token] = len(final_tokens)
                continue
            final_tokens.append(token)
        return final_tokens, (entries["[A]"], entries["[B]"], entries["[P]"])

    def convert_examples_to_features(self, examples, is_training):
        """Loads a data file into a list of `InputBatch`s."""

        # Swag is a multiple choice task. To perform this task using Bert,
        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Swag example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        features = []
        for example_index, example in enumerate(examples):
            context_tokens, (a_offset, b_offset, pronoun_offset) = self.tokenize_with_offsets(self.insert_tag(example.context_sentence, example.a_offset,
                                                                                                   example.b_offset, example.pronoun_offset), self.tokenizer)
            #start_ending_tokens = tokenizer.tokenize(example.start_ending)

            choices_features = []
            for ending_index, ending in enumerate(example.endings):
                # We create a copy of the context tokens in order to be
                # able to shrink it according to ending_tokens
                context_tokens_choice = context_tokens[:]
                ending_tokens = self.tokenizer.tokenize(ending)#start_ending_tokens + tokenizer.tokenize(ending)
                # Modifies `context_tokens_choice` and `ending_tokens` in
                # place so that the total length is less than the
                # specified length.  Account for [CLS], [SEP], [SEP] with
                # "- 3"
                self._truncate_seq_pair(context_tokens_choice, ending_tokens, self.max_seq_length - 3)

                tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
                segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

                # Account for the [CLS] token
                a_offset += 1
                b_offset += 1
                pronoun_offset += 1
                pronoun_ids = np.array([3] * (len(segment_ids)))
                if a_offset < len(pronoun_ids):
                    n_a_tokens = len(self.tokenizer.tokenize(example.endings[0]))
                    pronoun_ids[a_offset: a_offset + n_a_tokens] = 0
                if b_offset < len(pronoun_ids):
                    n_b_tokens = len(self.tokenizer.tokenize(example.endings[1]))
                    pronoun_ids[b_offset: b_offset + n_b_tokens] = 1

                #print(f"ei: {example_index}, po: {pronoun_offset}, pil: {len(pronoun_ids)}")
                #print("*" * 50)
                if pronoun_offset < len(pronoun_ids):
                    pronoun_ids[pronoun_offset] = 2
                pronoun_ids = list(pronoun_ids)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (self.max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                pronoun_ids += padding

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length
                assert len(pronoun_ids) == self.max_seq_length

                choices_features.append((tokens, input_ids, input_mask, segment_ids, pronoun_ids))

            label = example.label

            features.append(
                SwagInputFeatures(
                    example_id = example.swag_id,
                    choices_features = choices_features,
                    label = label
                )
            )

        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def select_field(self, features, field):
        return [
            [
                choice[field]
                for choice in feature.choices_features
            ]
            for feature in features
        ]
    
    def evaluate(self, model, eval_examples, is_test=False):
        eval_features = self.convert_examples_to_features(eval_examples, False)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", self.predict_batch_size)
        all_input_ids = torch.tensor(self.select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(self.select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(self.select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_pronoun_ids = torch.tensor(self.select_field(eval_features, 'pronoun_ids'), dtype=torch.long)
        all_label = None
        eval_data = None
        if not is_test:
            all_label = torch.tensor([f.label for f in eval_examples], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pronoun_ids, all_label)
        else:
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pronoun_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.predict_batch_size)

        model.eval()
        if is_test:
            eval_loss = None
        else:
            eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0

        test_logits = []
        #for input_ids, input_mask, segment_ids, pronoun_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = tuple(t.to(self.device) for t in batch)
            if not is_test:
                input_ids, input_mask, segment_ids, pronoun_ids, label_ids = batch
            else:
                input_ids, input_mask, segment_ids, pronoun_ids = batch

            with torch.no_grad():
                if not is_test:
                    label_ids = label_ids.to(self.device)
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    eval_loss += tmp_eval_loss.mean().item()
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            test_logits.extend(logits)
            
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            
        if not is_test:
            eval_loss = eval_loss / nb_eval_steps
        
        eval_probas = torch.softmax(torch.from_numpy(np.array(test_logits)), 1).cpu().numpy().clip(1e-3, 1-1e-3)
        
        return eval_loss, eval_probas
    
    def run_k_fold(self):
        kfold_data = pd.concat([self.dev_df, self.val_df])
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        test_examples = self.test_df.apply(lambda x: self.row_to_swag_example(x, False), axis=1).tolist()
        val_preds, test_preds, val_losses = [], [], []
        for train_index, valid_index in kf.split(kfold_data, kfold_data["gender"]):
            print("=" * 20)
            print(f"Fold {len(val_preds) + 1}")
            print("=" * 20)
            kf_train = kfold_data.iloc[train_index]
            kf_val = kfold_data.iloc[valid_index]

            train_examples = kf_train.apply(lambda x: self.row_to_swag_example(x, True), axis=1).tolist()
            val_examples = kf_val.apply(lambda x: self.row_to_swag_example(x, True), axis=1).tolist()
            
            num_train_optimization_steps = int(
                len(train_examples) / self.train_batch_size / self.gradient_accumulation_steps) * self.num_train_epochs
            
            # Prepare model
            model = BertForMultipleChoice.from_pretrained(self.bert_model,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(self.local_rank)),
                num_choices=self.num_choices)
            
            def children(m):
                return m if isinstance(m, (list, tuple)) else list(m.children())

            def set_trainable_attr(m, b):
                m.trainable = b
                for p in m.parameters():
                    p.requires_grad = b


            def apply_leaf(m, f):
                c = children(m)
                if isinstance(m, nn.Module):
                    f(m)
                if len(c) > 0:
                    for l in c:
                        apply_leaf(l, f)


            def set_trainable(l, b):
                apply_leaf(l, lambda m: set_trainable_attr(m, b))
                
            set_trainable(model.bert, False)
            set_trainable(model.bert.pooler, True)
            #set_trainable(model.bert.embeddings.token_pronoun_embeddings, True)

            for i in range(12,24):
                set_trainable(model.bert.encoder.layer[i], True)
                
            total_params = sum(p.numel() for p in model.parameters())

            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"Total parameters: {total_params}, trainable parameters: {total_trainable_params}")
            
            model.to(self.device)
            
            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            optimizer = BertAdam(optimizer_grouped_parameters,
                                         lr=self.learning_rate,
                                         warmup=self.warmup_proportion,
                                         t_total=num_train_optimization_steps)
            global_step = 0
            train_features = self.convert_examples_to_features(train_examples, True)
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            all_input_ids = torch.tensor(self.select_field(train_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(self.select_field(train_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(self.select_field(train_features, 'segment_ids'), dtype=torch.long)
            all_pronoun_ids = torch.tensor(self.select_field(train_features, 'pronoun_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pronoun_ids, all_label)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)

            model.train()
            for _ in trange(int(self.num_train_epochs), desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, pronoun_ids, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    
                    print(f"loss: {loss}")
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
            val_loss, val_probas = self.evaluate(model, val_examples, is_test=False)
            test_loss, test_probas = self.evaluate(model, test_examples, is_test=True)
            val_class_labels = kf_val['target'].tolist()#[self.get_class_label(aco, bco) for aco, bco in zip(kf_val['A-coref'], kf_val['B-coref'])]
            val_class_labels_2 = [get_class_label(aco, bco) for aco, bco in zip(kf_val['A-coref'], kf_val['B-coref'])]

            val_preds.append(val_probas)

            val_losses.append(log_loss(val_class_labels, val_probas,labels=[0,1,2]))
            logger.info("Confirm val loss: %.4f", val_losses[-1])
            test_preds.append(test_probas)

            del model
            
        return val_preds, test_preds, val_losses

    def train(self,train_set,validation_set,weight_folder_path,n_splits=2):


        train_set = pd.read_csv(train_set, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
        validation_set = pd.read_csv(validation_set, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")

        os.makedirs(weight_folder_path, exist_ok=True)
        train_set = self.extract_target(train_set)
        validation_set = self.extract_target(validation_set)

        #kfold_data = pd.concat([self.dev_df, self.val_df])
        kfold_data = pd.concat([train_set,validation_set])
        kf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
        #test_examples = self.test_df.apply(lambda x: self.row_to_swag_example(x, False), axis=1).tolist()
        val_preds, test_preds, val_losses = [], [], []
        zi=0

        for train_index, valid_index in kf.split(kfold_data, kfold_data["gender"]):
            print("=" * 20)
            print(f"\nFold {len(val_preds) + 1}")
            print("=" * 20)
            kf_train = kfold_data.iloc[train_index]
            kf_val = kfold_data.iloc[valid_index]

            train_examples = kf_train.apply(lambda x: self.row_to_swag_example(x, True), axis=1).tolist()
            val_examples = kf_val.apply(lambda x: self.row_to_swag_example(x, True), axis=1).tolist()
            
            num_train_optimization_steps = int(
                len(train_examples) / self.train_batch_size / self.gradient_accumulation_steps) * self.num_train_epochs
            
            # Prepare model
            model = BertForMultipleChoice.from_pretrained(self.bert_model,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(self.local_rank)),
                num_choices=self.num_choices)
            
            def children(m):
                return m if isinstance(m, (list, tuple)) else list(m.children())

            def set_trainable_attr(m, b):
                m.trainable = b
                for p in m.parameters():
                    p.requires_grad = b


            def apply_leaf(m, f):
                c = children(m)
                if isinstance(m, nn.Module):
                    f(m)
                if len(c) > 0:
                    for l in c:
                        apply_leaf(l, f)


            def set_trainable(l, b):
                apply_leaf(l, lambda m: set_trainable_attr(m, b))
                
            set_trainable(model.bert, False)
            set_trainable(model.bert.pooler, True)
            #set_trainable(model.bert.embeddings.token_pronoun_embeddings, True)

            for i in range(12,24):
                set_trainable(model.bert.encoder.layer[i], True)
                
            total_params = sum(p.numel() for p in model.parameters())

            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"Total parameters: {total_params}, trainable parameters: {total_trainable_params}")
            
            model.to(self.device)
            
            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            optimizer = BertAdam(optimizer_grouped_parameters,
                                         lr=self.learning_rate,
                                         warmup=self.warmup_proportion,
                                         t_total=num_train_optimization_steps)
            global_step = 0
            train_features = self.convert_examples_to_features(train_examples, True)
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            all_input_ids = torch.tensor(self.select_field(train_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(self.select_field(train_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(self.select_field(train_features, 'segment_ids'), dtype=torch.long)
            all_pronoun_ids = torch.tensor(self.select_field(train_features, 'pronoun_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pronoun_ids, all_label)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)

            model.train()
            for _ in trange(int(self.num_train_epochs), desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, pronoun_ids, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    
                    print(f"loss: {loss}")
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
            #questo potrei anche saltarlo per fare più veloce....
            #val_loss, val_probas = self.evaluate(model, val_examples, is_test=False)
            #test_loss, test_probas = self.evaluate(model, test_examples, is_test=True)



            #val_class_labels = kf_val['target'].tolist()#[self.get_class_label(aco, bco) for aco, bco in zip(kf_val['A-coref'], kf_val['B-coref'])]
            #val_class_labels = [get_class_label(aco, bco) for aco, bco in zip(kf_val['A-coref'], kf_val['B-coref'])]

            #val_preds.append(val_probas)


            #val_losses.append(log_loss(val_class_labels, val_probas,labels=[0,1,2]))
            #logger.info("Confirm val loss: %.4f", val_losses[-1])
            #test_preds.append(test_probas)

            with open(weight_folder_path+"/weights_fold_"+str(zi)+".mw", 'wb') as output:
                pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
            zi=zi+1
            del model
            
        #return val_preds, val_losses
        return None, None
       

    def my_evaluate(self, eval_examples_name, weight_folder_path, is_test=False):

        eval_examples_df = pd.read_csv(eval_examples_name, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")


        eval_examples = eval_examples_df.apply(lambda x: self.row_to_swag_example(x, not is_test), axis=1).tolist()
        val_preds =[]
        zi=0
        for filename in os.listdir(weight_folder_path):
            if filename.endswith(".mw") : 
                 # print(os.path.join(directory, filename))
                with open(weight_folder_path+"/weights_fold_"+str(zi)+".mw", "rb") as handle:
                    model = pickle.load(handle)
                zi=zi+1

                val_loss, val_probas = self.evaluate(model, eval_examples, is_test)

                val_preds.append(val_probas)

        final_preds = np.mean(val_preds, axis=0)

        return final_preds

    #val_examples = kf_val.apply(lambda x: self.row_to_swag_example(x, True), axis=1).tolist()



class model_9(model):

    def __init__(self):

        #problema cerca di inizializzare l'oggetto senza dover istanziare i dataset, occupo memoria per nulla.
        swag_runner = BertSwagRunner(None, None, None, num_train_epochs=1, bert_model='bert-large-uncased')
        self.runner = swag_runner

    def train(self, train_set, vallidation_set, weight_folder_path ):

        self.runner.train( train_set, vallidation_set, weight_folder_path, n_splits=4)


    #forse qui sarebbe meglio riuscire a salvare i pvari pesi tutti nello stesso pickle 
    def evaluate(self, val_df, weight_folder_path="model_9" ):

        return  self.runner.my_evaluate( val_df, weight_folder_path, is_test=False)


'''


#per trainare e testare più velocemente, sono solo 5 esempi
test_path = "../datasets/gap-light.tsv"
dev_path = "../datasets/gap-light.tsv"
val_path = "../datasets/gap-light.tsv"




print("\n\n\n\n         building model         \n\n")
model_9_inst = model_9()



print("\n\n\n\n         training model         \n\n")
model_9_inst.train(dev_path,val_path,"model_9")



print("\n\n\n\n         evaluating         \n\n")
val_probas = model_9_inst.evaluate( test_path,"model_9")

print("val_probas")
print(val_probas)


submission_df = pd.DataFrame([test_df_prod.ID, val_probas[:,0], val_probas[:,1], val_probas[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()

submission_df.to_csv('stage2_swag_only.csv', index=False)


'''