# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import absolute_import, division, print_function

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

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# Any results you write to the current directory are saved as output.

test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
dev_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv"
val_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv"

dev_df = pd.read_csv(test_path, delimiter="\t")#pd.read_csv(dev_path, delimiter="\t")
test_df = pd.read_csv(dev_path, delimiter="\t")#pd.read_csv(test_path, delimiter="\t")
val_df = pd.read_csv(val_path, delimiter="\t")

test_df_prod = test_df.copy()
test_df_prod = test_df_prod[['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'B', 'B-offset', 'URL']]

print(len(dev_df), len(val_df), len(test_df))



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


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class SquadInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        
SquadRawResult = collections.namedtuple("SquadRawResult",
                                       ["unique_id", "start_logits", "end_logits"])

# A good chunk of this is borrowed from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_squad.py
class SquadRunner:

    def __init__(self, dev_df, val_df, test_df, bert_model = 'bert-large-uncased', do_lower_case = True, learning_rate = 1e-5, num_train_epochs = 2, max_seq_length = 300, doc_stride = 128, train_batch_size = 12, predict_batch_size = 8, warmup_proportion = 0.1,                n_best_size = 20, max_query_length = 50, max_answer_length = 50, output_dir = 'squad'):
        self.dev_df = self.extract_target(dev_df)
        self.val_df = self.extract_target(val_df)
        self.test_df = test_df
        #self.test_df = self.extract_target(test_df)

        # Custom parameters
        self.do_lower_case = do_lower_case
        if do_lower_case: 
            self.bert_model = 'bert-large-uncased'
        else:
            self.bert_model = 'bert-large-cased'
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.output_dir = output_dir
        self.train_batch_size = train_batch_size

        # Default parameters
        self.predict_batch_size = predict_batch_size
        
        self.seed = 42
        self.warmup_proportion = warmup_proportion
        self.n_best_size = n_best_size
        self.max_query_length = max_query_length
        self.max_answer_length = max_answer_length
        self.local_rank = -1
        self.gradient_accumulation_steps = 1
        self.loss_scale = 0
        self.version_2_with_negative = False
        self.fp16 = False
        self.no_cuda = False
        self.verbose_logging = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")

        logger.info("device: {} distributed training: {}, 16-bits training: {}".format(
            self.device, bool(self.local_rank != -1), self.fp16))

        self.train_batch_size = self.train_batch_size // self.gradient_accumulation_steps

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case)
        
    def extract_target(self, df):
        df["Neither"] = 0
        df.loc[~(df['A-coref'] | df['B-coref']), "Neither"] = 1
        df["target"] = 0
        df.loc[df['B-coref'] == 1, "target"] = 1
        df.loc[df["Neither"] == 1, "target"] = 2
        df['gender'] = df['Pronoun'].transform(get_gender)
        return df

    def read_squad_examples_from_data(self, input_data, is_training, version_2_with_negative):

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if is_training:
                        if version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)
        return examples

    def row_to_squad_example(self, row, is_training):
        json_dict = {}

        question_text = ""
        pronoun_offset = row['Pronoun-offset']
        n_chars_processed = 0
        words = row['Text'].split(" ")
        for i, w in enumerate(words):
            n_chars_processed += len(w) + 1
            if n_chars_processed >= pronoun_offset:
                question_text = " ".join(words[i:i+5])
                break 
        
        qas = None
        if is_training:
            answer_offset = row['A-offset'] if row['A-coref'] else row['B-offset']
            answer_text = row['A'] if row['A-coref'] else row['B']
            qas = [{'answers': [{'answer_start': answer_offset, 'text': answer_text}], 
                                 'question': question_text, 'id': str(ObjectId())}]
        else:
            qas = [{'question': question_text, 'id': str(ObjectId())}]
            
        json_dict['paragraphs'] = [{'context': row['Text'], 'qas': qas}]
        
        return json_dict 
        
    def convert_examples_to_features(self, examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000

        features = []
        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question_text)

            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                start_position = None
                end_position = None
                if is_training and not example.is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                if is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0


                if example_index < 20:
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (example_index))
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    logger.info("tokens: %s" % " ".join(tokens))
                    logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if is_training and example.is_impossible:
                        logger.info("impossible example")
                    if is_training and not example.is_impossible:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        logger.info("start_position: %d" % (start_position))
                        logger.info("end_position: %d" % (end_position))
                        logger.info(
                            "answer: %s" % (answer_text))

                features.append(
                    SquadInputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=example.is_impossible))
                unique_id += 1

        return features


    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,  orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)


    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index


    def write_predictions(self, all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case):
        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min mull score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = self._get_best_indexes(result.start_logits, n_best_size)
                end_indexes = self._get_best_indexes(result.end_logits, n_best_size)

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index]))
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = self.get_final_text(tok_text, orig_text, do_lower_case, verbose_logging=False)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = self._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            all_predictions[example.qas_id] = nbest_json[0]["text"]


        return all_predictions, all_nbest_json


    def get_final_text(self, pred_text, orig_text, do_lower_case, verbose_logging=False):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if verbose_logging:
                logger.info(
                    "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if verbose_logging:
                logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if verbose_logging:
                logger.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if verbose_logging:
                logger.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text


    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes


    def _compute_softmax(self, scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs
    
    #
    def evaluate(self, model, eval_examples, eval_features):
        logger.info("***** Running predictions *****")
        #logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", self.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(SquadRawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))

        all_predictions, all_nbest_json = self.write_predictions(eval_examples, eval_features, all_results,
                          self.n_best_size, self.max_answer_length,
                          self.do_lower_case)

        return all_predictions, all_results
    
    def get_class_label(self, a_coref, b_coref):
        if a_coref:
            return 0
        elif b_coref:
            return 1
        return 2
    
    def get_start_end_logit(self, example, feature, result, text, offset):
        of = 0
        orig_tok_idx_start = 0
        for t in example.doc_tokens:
            of += len(t) + 1
            if of > offset:
                break
            orig_tok_idx_start += 1

        orig_tok_idx_end = orig_tok_idx_start + len(whitespace_tokenize(text)) - 1

        start_logit = -100 #result.start_logits[0]
        end_logit = -100 #result.end_logits[0]

        for feat_idx, orig_idx in feature.token_to_orig_map.items():
            if orig_idx >= orig_tok_idx_start and orig_idx <= orig_tok_idx_end:
                start_logit = max(start_logit, result.start_logits[feat_idx])
                end_logit = max(end_logit, result.end_logits[feat_idx])
        return start_logit, end_logit

    def build_a_b_logits(self, examples, features, results, predictions, a_texts, b_texts, a_offsets, b_offsets):
        logits = {}
        for feature, result in zip(features, results):
            a_b_logit = []
            
            example = examples[feature.example_index]
            a_text = a_texts[feature.example_index]
            a_offset = a_offsets[feature.example_index]
            b_text =b_texts[feature.example_index]
            b_offset = b_offsets[feature.example_index]


            a_start_logit, a_end_logit = self.get_start_end_logit(example, feature, result, a_text, a_offset)
            b_start_logit, b_end_logit = self.get_start_end_logit(example, feature, result, b_text, b_offset)
            max_start_logit = max(result.start_logits)
            max_end_logit = max(result.end_logits)

            if feature.example_index in logits:
                a_start_logit = max(a_start_logit, logits[feature.example_index][0])
                a_end_logit = max(a_end_logit, logits[feature.example_index][1])
                b_start_logit = max(b_start_logit, logits[feature.example_index][2])
                b_end_logit = max(b_end_logit, logits[feature.example_index][3])
                max_start_logit = max(max_start_logit, logits[feature.example_index][4])
                max_end_logit = max(max_end_logit, logits[feature.example_index][5])

            a_b_logit.append(a_start_logit)
            a_b_logit.append(a_end_logit)
            a_b_logit.append(b_start_logit)
            a_b_logit.append(b_end_logit)
            a_b_logit.append(max_start_logit)
            a_b_logit.append(max_end_logit)

            prediction = predictions[example.qas_id]

            if prediction.lower() in a_text.lower() or a_text.lower() in prediction.lower():
                a_b_logit.append(0.)
            elif prediction.lower() in b_text.lower() or b_text.lower() in prediction.lower():
                a_b_logit.append(1.)
            else:
                a_b_logit.append(2.)

            logits[feature.example_index] = a_b_logit

        return list(logits.values())
    
    def run_k_fold(self):
        kfold_data = pd.concat([self.dev_df, self.val_df])
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        test_squad_format = self.test_df.apply(lambda x: self.row_to_squad_example(x, False), axis=1).tolist()
        test_examples = self.read_squad_examples_from_data(test_squad_format, False, False)
        test_features = self.convert_examples_to_features(
                    examples=test_examples,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_seq_length,
                    doc_stride=self.doc_stride,
                    max_query_length=self.max_query_length,
                    is_training=False)
        val_preds, test_preds, val_losses = [], [], []
        for train_index, valid_index in kf.split(kfold_data, kfold_data["gender"]):
            print("=" * 20)
            print(f"Fold {len(val_preds) + 1}")
            print("=" * 20)
            kf_train_unfiltered = kfold_data.iloc[train_index]
            kf_val_unfiltered = kfold_data.iloc[valid_index]
            kf_train = kf_train_unfiltered[kf_train_unfiltered['A-coref'] | kf_train_unfiltered['B-coref']]
            kf_val = kf_val_unfiltered[kf_val_unfiltered['A-coref'] | kf_val_unfiltered['B-coref']]

            train_squad = kf_train.apply(lambda x: self.row_to_squad_example(x, True), axis=1).tolist()
            val_squad = kf_val.apply(lambda x: self.row_to_squad_example(x, True), axis=1).tolist()
            train_examples = self.read_squad_examples_from_data(train_squad, True, False)
            val_examples = self.read_squad_examples_from_data(val_squad, False, False)

            num_train_optimization_steps = int(
                len(train_examples) / self.train_batch_size / self.gradient_accumulation_steps) * self.num_train_epochs

            # Prepare model
            model = BertForQuestionAnswering.from_pretrained(self.bert_model,
                        cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(self.local_rank)))

            # Freeze some weights
            model_children = list(model.children())
            bert_layers = list(model_children[0].children())
            bert_embeddings, bert_encoder, bert_pooler = bert_layers

            for param in bert_embeddings.parameters():
                param.requires_grad = False

            for child in list(bert_encoder.children())[0][:-12]:
                for param in child.parameters():
                    param.requires_grad = False

            total_params = sum(p.numel() for p in model.parameters())

            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"Total parameters: {total_params}, trainable parameters: {total_trainable_params}")

            model.to(self.device)
            model = torch.nn.DataParallel(model)

            # Prepare optimizer
            param_optimizer = list(model.named_parameters())

            # hack to remove pooler, which is not used
            # thus it produce None grad that break apex
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

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

            train_features = self.convert_examples_to_features(
                examples=train_examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training=True)

            logger.info("***** Running training *****")
            logger.info("  Num orig examples = %d", len(train_examples))
            logger.info("  Num split examples = %d", len(train_features))
            logger.info("  Batch size = %d", self.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                       all_start_positions, all_end_positions)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)

            model.train()
            for _ in trange(int(self.num_train_epochs), desc="Epoch"):
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(self.device) for t in batch) # multi-gpu does scattering it-self
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)

                    if self.gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                    print(f"loss: {loss}")
                    loss.backward()
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join('.', WEIGHTS_NAME)
            output_config_file = os.path.join('.', CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            self.tokenizer.save_vocabulary('.')

            train_squad_unfiltered = kf_train_unfiltered.apply(lambda x: self.row_to_squad_example(x, True), axis=1).tolist()
            val_squad_unfiltered = kf_val_unfiltered.apply(lambda x: self.row_to_squad_example(x, True), axis=1).tolist()
            train_examples_unfiltered = self.read_squad_examples_from_data(train_squad_unfiltered, True, False)
            val_examples_unfiltered = self.read_squad_examples_from_data(val_squad_unfiltered, False, False)

            train_features_unfiltered = self.convert_examples_to_features(
                examples=train_examples_unfiltered,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training=False)
            val_features_unfiltered = self.convert_examples_to_features(
                examples=val_examples_unfiltered,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training=False)

            # Train logits
            train_predictions, train_results = self.evaluate(model, train_examples_unfiltered, train_features_unfiltered)

            # Val logits
            val_predictions, val_results = self.evaluate(model, val_examples_unfiltered, val_features_unfiltered)

            # Test logits
            test_predictions, test_results = self.evaluate(model, test_examples, test_features)

            train_a_b_logits = self.build_a_b_logits(train_examples_unfiltered, train_features_unfiltered, train_results, train_predictions, kf_train_unfiltered['A'].tolist(),
                    kf_train_unfiltered['B'].tolist(), kf_train_unfiltered['A-offset'].tolist(), kf_train_unfiltered['B-offset'].tolist())
            val_a_b_logits = self.build_a_b_logits(val_examples_unfiltered, val_features_unfiltered, val_results, val_predictions, kf_val_unfiltered['A'].tolist(),
                            kf_val_unfiltered['B'].tolist(), kf_val_unfiltered['A-offset'].tolist(), kf_val_unfiltered['B-offset'].tolist())
            test_a_b_logits = self.build_a_b_logits(test_examples, test_features, test_results, test_predictions, self.test_df['A'].tolist(),
                            self.test_df['B'].tolist(), self.test_df['A-offset'].tolist(), self.test_df['B-offset'].tolist())

            scaler = StandardScaler().fit(train_a_b_logits)

            train_a_b_logits_scaled = scaler.transform(train_a_b_logits)
            val_a_b_logits_scaled = scaler.transform(val_a_b_logits)
            test_a_b_logits_scaled = scaler.transform(test_a_b_logits)

            train_class_labels = [self.get_class_label(aco, bco) for aco, bco in zip(kf_train_unfiltered['A-coref'], kf_train_unfiltered['B-coref'])]
            val_class_labels = [self.get_class_label(aco, bco) for aco, bco in zip(kf_val_unfiltered['A-coref'], kf_val_unfiltered['B-coref'])]

            logreg = LogisticRegression(C=0.1)
            logreg.fit(np.array(train_a_b_logits_scaled), train_class_labels)

            val_logreg_probas = logreg.predict_proba(val_a_b_logits_scaled)
            test_logreg_probas = logreg.predict_proba(test_a_b_logits_scaled)

            val_preds.append(val_logreg_probas)
            val_losses.append(log_loss(val_class_labels, val_logreg_probas))
            logger.info("Confirm val loss: %.4f", val_losses[-1])
            test_preds.append(test_logreg_probas)

            del model
            
            break
            
        return val_preds, test_preds, val_losses



squad_runner = SquadRunner(dev_df, val_df, test_df_prod, train_batch_size=12, num_train_epochs=2, do_lower_case=True)

squad_val_preds, squad_test_preds, squad_val_losses = squad_runner.run_k_fold()

squad_test_probas = np.mean(squad_test_preds, axis=0)


submission_df = pd.DataFrame([test_df_prod.ID, squad_test_probas[:,0], squad_test_probas[:,1], squad_test_probas[:,2]], index=['ID', 'A', 'B', 'NEITHER']).transpose()
submission_df.to_csv('stage1_squad_only.csv', index=False)