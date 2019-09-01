# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import absolute_import, division, print_function

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


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

from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

from helperbot import (
    TriangularLR, BaseBot, WeightDecayOptimizerWrapper,
    GradualWarmupScheduler
)

import gc

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# Any results you write to the current directory are saved as output.
'''
test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
dev_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv"
val_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv"
'''

test_path = "../datasets/gap-light.tsv"
dev_path = "../datasets/gap-light.tsv"
val_path = "../datasets/gap-light.tsv"


dev_df = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
test_df = pd.read_csv(dev_path, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")
val_df = pd.read_csv(val_path, delimiter="\t")

test_df_prod = test_df.copy()




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

test_class_labels = [get_class_label(aco, bco) for aco, bco in zip(test_df['A-coref'], test_df['B-coref'])]





class Head(nn.Module):
    """The MLP submodule"""
    def __init__(self, bert_hidden_size: int):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        self.span_extractor = SelfAttentiveSpanExtractor(bert_hidden_size * 3)
        # EndpointSpanExtractor below also gives similar results
#         self.span_extractor = EndpointSpanExtractor(
#             bert_hidden_size * 3, "x,y,x*y"
#         )
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_hidden_size * 3 * 3, 512),           
            nn.ReLU(),
            nn.Linear(512, 3)
        )
                
    def forward(self, bert_outputs, offsets):
        assert bert_outputs.size(2) == self.bert_hidden_size * 3
        spans_contexts = self.span_extractor(
            bert_outputs, 
            offsets[:, :4].reshape(-1, 2, 2)
        )
        spans_contexts = spans_contexts.reshape(offsets.size()[0], -1)
        return self.fc(torch.cat([
            spans_contexts,
            torch.gather(
                bert_outputs, 1,
                offsets[:, [4]].unsqueeze(2).expand(-1, -1, self.bert_hidden_size * 3)
            ).squeeze(1)
        ], dim=1))

class GAPDataset(Dataset):
    """Custom GAP Dataset class"""
    def __init__(self, df, tokenizer, tokenize_fn, labeled=True):
        self.labeled = labeled
        if labeled:
            self.y = df.target.values.astype("uint8")
        
        self.offsets, self.tokens = [], []
        for _, row in df.iterrows():
            tokens, offsets = tokenize_fn(row, tokenizer)
            self.offsets.append(offsets)
            self.tokens.append(tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + tokens + ["[SEP]"]))
        
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.offsets[idx], self.y[idx]
        return self.tokens[idx], self.offsets[idx], None

class GAPModel(nn.Module):
    """The main model."""
    def __init__(self, bert_model: str, device: torch.device, use_layer: int = -1):
        super().__init__()
        self.device = device
        self.use_layer = use_layer
        if bert_model in ("bert-base-uncased", "bert-base-cased"):
            self.bert_hidden_size = 768
        elif bert_model in ("bert-large-uncased", "bert-large-cased"):
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")
        self.bert = BertModel.from_pretrained(bert_model).to(device)
        self.head = Head(self.bert_hidden_size).to(device)
    
    def forward(self, token_tensor, offsets):
        token_tensor = token_tensor.to(self.device)
        bert_outputs, _ =  self.bert(
            token_tensor, attention_mask=(token_tensor > 0).long(), 
            token_type_ids=None, output_all_encoded_layers=True)
        concat_bert = torch.cat((bert_outputs[-1],bert_outputs[-2],bert_outputs[-3]),dim=-1)
        #head_outputs = self.head(bert_outputs[self.use_layer], offsets.to(self.device))
        head_outputs = self.head(concat_bert, offsets.to(self.device))
        return head_outputs            
    
class GAPBot(BaseBot):
    def __init__(self, model, train_loader, val_loader, *, optimizer, clip_grad=0,
        avg_window=100, log_dir="./cache/logs/", log_level=logging.INFO,
        checkpoint_dir="./cache/model_cache/", batch_idx=0, echo=False,
        device="cpu", use_tensorboard=False):
        super().__init__(
            model, train_loader, val_loader, 
            optimizer=optimizer, clip_grad=clip_grad,
            log_dir=log_dir, checkpoint_dir=checkpoint_dir, 
            batch_idx=batch_idx, echo=echo,
            device=device, use_tensorboard=use_tensorboard
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_format = "%.6f"
        
    def extract_prediction(self, tensor):
        return tensor
    
    def snapshot(self):
        """Override the snapshot method because Kaggle kernel has limited local disk space."""
        loss = self.eval(self.val_loader)
        loss_str = self.loss_format % loss
        self.logger.info("Snapshot loss %s", loss_str)
        self.logger.tb_scalars(
            "losses", {"val": loss},  self.step)
        target_path = (
            self.checkpoint_dir / "best.pth")        
        if not self.best_performers or (self.best_performers[0][0] > loss):
            torch.save(self.model.state_dict(), target_path)
            self.best_performers = [(loss, target_path, self.step)]
        self.logger.info("Saving checkpoint %s...", target_path)
        assert Path(target_path).exists()
        return loss

class BERTSpanExtractor:
    def __init__(self, dev_df, val_df, test_df, bert_model = 'bert-large-uncased', do_lower_case=True, learning_rate=1e-5, n_epochs=30,
                train_batch_size=10, predict_batch_size=32):
        self.dev_df = self.extract_target(dev_df)
        self.val_df = self.extract_target(val_df)
        self.test_df = self.extract_target(test_df)
        self.do_lower_case = do_lower_case
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size
        self.bert_model = bert_model
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case, 
                                                       never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"))
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = torch.device("cpu")
        
    def extract_target(self, df):
        df['target'] = [get_class_label(aco, bco) for aco, bco in zip(df['A-coref'], df['B-coref'])]
        df['gender'] = df['Pronoun'].transform(get_gender)

        return df
        
    def tokenize(self, row, tokenizer):
        break_points = sorted(
            [
                ("A", row["A-offset"], row["A"]),
                ("B", row["B-offset"], row["B"]),
                ("P", row["Pronoun-offset"], row["Pronoun"]),
            ], key=lambda x: x[0]
        )
        tokens, spans, current_pos = [], {}, 0
        for name, offset, text in break_points:
            tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))
            # Make sure we do not get it wrong
            assert row["Text"][offset:offset+len(text)] == text
            # Tokenize the target
            tmp_tokens = tokenizer.tokenize(row["Text"][offset:offset+len(text)])
            spans[name] = [len(tokens), len(tokens) + len(tmp_tokens) - 1] # inclusive
            tokens.extend(tmp_tokens)
            current_pos = offset + len(text)
        tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))
        assert spans["P"][0] == spans["P"][1]
        return tokens, (spans["A"] + spans["B"] + [spans["P"][0]])

    def collate_examples(self, batch, truncate_len=490):
        """Batch preparation.

        1. Pad the sequences
        2. Transform the target.
        """    
        transposed = list(zip(*batch))
        max_len = min(
            max((len(x) for x in transposed[0])),
            truncate_len
        )
        tokens = np.zeros((len(batch), max_len), dtype=np.int64)
        for i, row in enumerate(transposed[0]):
            row = np.array(row[:truncate_len])
            tokens[i, :len(row)] = row
        token_tensor = torch.from_numpy(tokens)
        # Offsets
        offsets = torch.stack([
            torch.LongTensor(x) for x in transposed[1]
        ], dim=0) + 1 # Account for the [CLS] token
        # Labels
        if len(transposed) == 2 or transposed[2][0] is None:
            return token_tensor, offsets, None
        labels = torch.LongTensor(transposed[2])
        return token_tensor, offsets, labels
    
    def run_k_fold(self):
        test_ds = GAPDataset(self.test_df, self.tokenizer, self.tokenize, labeled=True) #not great, but it's a hack needed so this "bot" thing doesn't crash
        test_loader = DataLoader(
            test_ds,
            collate_fn = self.collate_examples,
            batch_size=self.predict_batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False
        )

        kfold_data = pd.concat([self.dev_df, self.val_df])
        kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

        val_preds, test_preds, val_ys, val_losses = [], [], [], []
        for train_index, valid_index in kf.split(kfold_data, kfold_data["gender"]):
            print("=" * 20)
            print(f"Fold {len(val_preds) + 1}")
            print("=" * 20)
            train_ds = GAPDataset(kfold_data.iloc[train_index], self.tokenizer, self.tokenize)
            val_ds = GAPDataset(kfold_data.iloc[valid_index], self.tokenizer, self.tokenize)
            train_loader = DataLoader(
                train_ds,
                collate_fn = self.collate_examples,
                batch_size=self.train_batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=True,
                drop_last=False #True
            )
            val_loader = DataLoader(
                val_ds,
                collate_fn = self.collate_examples,
                batch_size=self.predict_batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=False
            )
            model = GAPModel(self.bert_model, self.device)
            
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
                
            # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
            set_trainable(model.bert, False)
            set_trainable(model.head, True)
            for i in range(12,24):
                set_trainable(model.bert.encoder.layer[i], True)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            bot = GAPBot(
                model, train_loader, val_loader,
                optimizer=optimizer, echo=True,
                avg_window=25
            )
            gc.collect()
            steps_per_epoch = len(train_loader) 
            n_steps = steps_per_epoch * self.n_epochs
            bot.train(
                n_steps,
                log_interval=steps_per_epoch // 2,
                snapshot_interval=steps_per_epoch,
                scheduler=TriangularLR(
                    optimizer, 20, ratio=2, steps_per_cycle=steps_per_epoch * 100)
            )
            # Load the best checkpoint
            bot.load_model(bot.best_performers[0][1])
            bot.remove_checkpoints(keep=0)    
            val_preds.append(torch.softmax(bot.predict(val_loader), -1).clamp(1e-4, 1-1e-4).cpu().numpy())
            val_ys.append(kfold_data.iloc[valid_index].target.astype("uint8").values)
            val_losses.append(log_loss(val_ys[-1], val_preds[-1]))
            bot.logger.info("Confirm val loss: %.4f", val_losses[-1])
            test_preds.append(torch.softmax(bot.predict(test_loader), -1).clamp(1e-4, 1-1e-4).cpu().numpy())
            del model
            
        return val_preds, test_preds, val_losses


print("end declaration")
bert_span_extractor = BERTSpanExtractor(dev_df, val_df, test_df_prod, train_batch_size=10, n_epochs=15, bert_model='bert-large-uncased')
print("running k-fold")

bert_span_val_preds, bert_span_test_preds, bert_span_val_losses = bert_span_extractor.run_k_fold()


span_test_probas = np.mean(bert_span_test_preds, axis=0)
submission_df = pd.DataFrame([test_df_prod.ID, span_test_probas[:,0], span_test_probas[:,1], span_test_probas[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()

submission_df.to_csv('stage1_span_only.csv', index=False)