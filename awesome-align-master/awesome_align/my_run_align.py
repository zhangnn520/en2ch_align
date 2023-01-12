# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020, Zi-Yi Dou
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

import os
import torch
import logging
import shutil
import random
import argparse
import tempfile
import itertools
import numpy as np
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm, trange

from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer
from awesome_align.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)
base_dir = os.getcwd()


class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path, offsets=None):
        assert os.path.isfile(file_path)
        self.examples = []
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.offsets = offsets

    def process_line(self, worker_id, line):
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None

        src, tgt = line.split(' ||| ')
        if src.rstrip() == '' or tgt.rstrip() == '':
            return None

        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for
                                                                                      word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [
            self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                            max_length=self.tokenizer.max_len)['input_ids'], \
                           self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                            max_length=self.tokenizer.max_len)['input_ids']
        if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
            return None

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        return (worker_id, ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt)

    def __iter__(self):
        if self.offsets is not None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id
            offset_start = self.offsets[worker_id]
            offset_end = self.offsets[worker_id + 1] if worker_id + 1 < len(self.offsets) else None
        else:
            offset_start = 0
            offset_end = None
            worker_id = 0

        with open(self.file_path, encoding="utf-8") as f:
            f.seek(offset_start)
            line = f.readline()
            while line:
                processed = self.process_line(worker_id, line)
                if processed is None:
                    print(
                        f'Line "{line.strip()}" (offset in bytes: {f.tell()}) is not in the correct format. Skipping...')
                    empty_tensor = torch.tensor([self.tokenizer.cls_token_id, 999, self.tokenizer.sep_token_id])
                    empty_sent = ''
                    yield (worker_id, empty_tensor, empty_tensor, [-1], [-1], empty_sent, empty_sent)
                else:
                    yield processed
                if offset_end is not None and f.tell() >= offset_end:
                    break
                line = f.readline()


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    gold_path = args.eval_gold_file if evaluate else args.train_gold_file
    return LineByLineTextDataset(tokenizer, args, file_path=file_path, gold_path=gold_path)


def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args, langid_mask=None, lang_id=None) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    mask_type = torch.bool

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=mask_type), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    if langid_mask is not None:
        padding_mask = langid_mask.eq(lang_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).to(mask_type)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(mask_type) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(mask_type) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--output_file", default=os.path.join(base_dir, "../align_output/align_result.txt"),
        type=str, help="The output file."
    )
    parser.add_argument(
        "--output_prob_file", default=os.path.join(base_dir, "../align_output/output_prob_file.txt"),
        type=str, help="The output probability file."
    )
    parser.add_argument(
        "--output_word_file", default=os.path.join(base_dir, "../align_output/output_word_file.txt"), type=str,
        help='The output word file.'
    )
    parser.add_argument(
        "--data_file", default=os.path.join(base_dir, "../data/model_data/test.txt"),
        type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--checkpoint_model_name", default="checkpoint-18000",
        type=str, help="The input training data file (a text file)."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(base_dir, "../outputs"),
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--ignore_possible_alignments", default=True, action="store_true",
        help="Whether to ignore possible gold alignments"
    )
    parser.add_argument(
        "--gold_one_index", default=True, action="store_true", help="Whether the gold alignment files are one-indexed"
    )
    # Other parameters
    parser.add_argument("--cache_data", action="store_true", default=True, help='if cache the dataset')
    parser.add_argument("--align_layer", type=int, default=8, help="layer for alignment extraction")
    parser.add_argument(
        "--extraction", default='softmax', type=str, choices=['softmax', 'entmax'], help='softmax or entmax'
    )
    parser.add_argument(
        "--softmax_threshold", type=float, default=0.001
    )
    parser.add_argument(
        "--num_workers", action="store_true", default=0,
        help="对应有多个线程，win环境下如果该值不为0会报错"
    )

    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-multilingual-cased",
        type=str,
        help="论文中说这个模型名称的效果最好‘lm-mlm-100-1280,目前使用的是bert-base-multilingual-cased",
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="bert-base-multilingual-cased",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(base_dir, "../model"),
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", default=True,
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", default=True, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    modeling.PAD_ID = tokenizer.pad_token_id
    modeling.CLS_ID = tokenizer.cls_token_id
    modeling.SEP_ID = tokenizer.sep_token_id

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    # Evaluation
    checkpoint = os.path.join(args.output_dir, args.checkpoint_model_name)
    logger.info("Evaluate the following checkpoints: %s", args.checkpoint_model_name)
    model = model_class.from_pretrained(checkpoint)
    model.to(args.device)
    word_align(args, model, tokenizer)


def open_writer_list(filename, num_workers):
    writer = open(filename, 'w+', encoding='utf-8')
    writers = [writer]
    if num_workers > 1:
        writers.extend([tempfile.TemporaryFile(mode='w+', encoding='utf-8') for i in range(1, num_workers)])
    return writers


def find_offsets(filename, num_workers):
    if num_workers <= 1:
        return None
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_workers
        offsets = [0]
        for i in range(1, num_workers):
            f.seek(chunk_size * i)
            pos = f.tell()
            while True:
                try:
                    l = f.readline()
                    break
                except UnicodeDecodeError:
                    pos -= 1
                    f.seek(pos)
            offsets.append(f.tell())
    return offsets


def word_align(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    def collate(examples):
        worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

    offsets = find_offsets(args.data_file, args.num_workers)
    dataset = LineByLineTextDataset(tokenizer, file_path=args.data_file, offsets=offsets)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers
    )

    model.to(args.device)
    model.eval()
    tqdm_iterator = trange(0, desc="Extracting")

    writers = open_writer_list(args.output_file, args.num_workers)
    if args.output_prob_file is not None:
        prob_writers = open_writer_list(args.output_prob_file, args.num_workers)
    if args.output_word_file is not None:
        word_writers = open_writer_list(args.output_word_file, args.num_workers)

    for batch in dataloader:
        with torch.no_grad():
            worker_ids, ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch
            word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device,
                                                      0, 0, align_layer=args.align_layer, extraction=args.extraction,
                                                      softmax_threshold=args.softmax_threshold, test=True,
                                                      output_prob=(args.output_prob_file is not None))
            for worker_id, word_aligns, sent_src, sent_tgt in zip(worker_ids, word_aligns_list, sents_src, sents_tgt):
                output_str = []
                if args.output_prob_file is not None:
                    output_prob_str = []
                if args.output_word_file is not None:
                    output_word_str = []
                for word_align in word_aligns:
                    if word_align[0] != -1:
                        output_str.append(f'{word_align[0]}-{word_align[1]}')
                        if args.output_prob_file is not None:
                            output_prob_str.append(f'{word_aligns[word_align]}')
                        if args.output_word_file is not None:
                            output_word_str.append(f'{sent_src[word_align[0]]}<sep>{sent_tgt[word_align[1]]}')
                writers[worker_id].write(' '.join(output_str) + '\n')
                if args.output_prob_file is not None:
                    prob_writers[worker_id].write(' '.join(output_prob_str) + '\n')
                if args.output_word_file is not None:
                    word_writers[worker_id].write(' '.join(output_word_str) + '\n')
            tqdm_iterator.update(len(ids_src))

    merge_files(writers)
    if args.output_prob_file is not None:
        merge_files(prob_writers)
    if args.output_word_file is not None:
        merge_files(word_writers)


def merge_files(writers):
    if len(writers) == 1:
        writers[0].close()
        return

    for i, writer in enumerate(writers[1:], 1):
        writer.seek(0)
        shutil.copyfileobj(writer, writers[0])
        writer.close()
    writers[0].close()
    return


if __name__ == "__main__":
    main()
