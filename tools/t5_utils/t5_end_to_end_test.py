# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import torch
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../")

from transformers import PreTrainedTokenizerFast
from transformers import T5Tokenizer # transformers-4.10.0-py3
from utils.recover_bpe import recover_bpe
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from sacrebleu import corpus_bleu


def bleu_score(pred, ref):
    bleu = corpus_bleu(pred, [ref], force=True)
    print("       bleu score: {:6.2f}".format(bleu.score))
    print("       bleu counts: {}".format(bleu.counts))
    print("       bleu totals: {}".format(bleu.totals))
    print("       bleu precisions: {}".format(bleu.precisions))
    print("       bleu sys_len: {}; ref_len: {}".format(bleu.sys_len, bleu.ref_len))
    return bleu

def prepare_tensor(name, input):
    client_util = httpclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

class TranslationResult(object):
    def __init__(self, name, frame_work):
        self.name = name
        self.frame_work = frame_work # FT or HF
        self.file_name = name + ".txt"

        self.token_list = []
        self.batch_ids_list = []
        self.batch_seq_len_list = []
        self.batch_num = 0
        self.execution_time = 0.0  # seconds
        self.sentence_num = 0
        self.bleu_score = None
            
def translate(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    source_file = args_dict["source"]
    tgt_file = args_dict["target"]
    topk = args_dict['sampling_topk']
    topp = args_dict['sampling_topp']
    maximum_output_length = args_dict['maximum_output_length']

    tokenizer = T5Tokenizer.from_pretrained(args_dict['model'])
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(args_dict['model'])

    with open(source_file, 'r') as f:
        src_text = recover_bpe(f.readlines())
        src_text = ["translate English to German: " + line.strip() for line in src_text]

    with open(tgt_file, 'r') as f:
        tgt_text = recover_bpe(f.readlines())

    translation_result_list = []
    translation_result_list.append(TranslationResult("ft_triton", "FT"))
    client_util = httpclient
    for i in range(len(translation_result_list)):
        sys.stdout.flush()
        
        url = "localhost:8000"
        model_name = "fastertransformer"
        request_parallelism = 10
        verbose = False
        with client_util.InferenceServerClient(url,
                                               concurrency=request_parallelism,
                                               verbose=verbose) as client:
            prev = 0
            start_time = datetime.now()
            results = []
            while prev < len(src_text):
                input_texts = src_text[prev:prev+batch_size]
                prev += batch_size
                input_token = tokenizer(input_texts, return_tensors='pt', padding=True)
                
                input_ids = input_token.input_ids.numpy().astype(np.uint32)
                mem_seq_len = torch.sum(input_token.attention_mask, dim=1).numpy().astype(np.uint32)
                mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0], 1])

                # TODO(bhsueh) should be set to optional inputs in the future
                runtime_top_k = (topk * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
                runtime_top_p = topp * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                beam_search_diversity_rate = 0.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                temperature = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                len_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                repetition_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                random_seed = 0 * np.ones([input_ids.shape[0], 1]).astype(np.int32)
                is_return_log_probs = True * np.ones([input_ids.shape[0], 1]).astype(bool)
                max_output_len = (maximum_output_length * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
                bad_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
                stop_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
                beam_width = (args_dict['beam_width'] * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
                start_ids = 0 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
                end_ids = 1 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)

                inputs = [
                    prepare_tensor("input_ids", input_ids),
                    prepare_tensor("sequence_length", mem_seq_len),
                    prepare_tensor("runtime_top_k", runtime_top_k),
                    prepare_tensor("runtime_top_p", runtime_top_p),
                    prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
                    prepare_tensor("temperature", temperature),
                    prepare_tensor("len_penalty", len_penalty),
                    prepare_tensor("repetition_penalty", repetition_penalty),
                    prepare_tensor("random_seed", random_seed),
                    prepare_tensor("is_return_log_probs", is_return_log_probs),
                    prepare_tensor("max_output_len", max_output_len),
                    prepare_tensor("beam_width", beam_width),
                    prepare_tensor("start_id", start_ids),
                    prepare_tensor("end_id", end_ids),
                    prepare_tensor("bad_words_list", bad_words_ids),
                    prepare_tensor("stop_words_list", stop_words_ids),
                ]
                
                print("set request")
                result = client.infer(model_name, inputs)
                print("get request")
                results.append(result)
                
            for result in results:
                ft_decoding_outputs = result.as_numpy("output_ids")
                ft_decoding_seq_lens = result.as_numpy("sequence_length")
                cum_log_probs = result.as_numpy("cum_log_probs")
                output_log_probs = result.as_numpy("output_log_probs")
                
                translation_result_list[i].batch_ids_list.append(ft_decoding_outputs)
                translation_result_list[i].batch_seq_len_list.append(ft_decoding_seq_lens)
                
                translation_result_list[i].sentence_num += len(input_token)
                translation_result_list[i].batch_num += 1

        stop_time = datetime.now()
        translation_result_list[i].execution_time = (stop_time - start_time).total_seconds()
        
        for batch_token, batch_seq_len in zip(translation_result_list[i].batch_ids_list, translation_result_list[i].batch_seq_len_list):
            for j in range(len(batch_token)):
                translation_result_list[i].token_list.append(fast_tokenizer.decode(batch_token[j][0][:batch_seq_len[j][0]], skip_special_tokens=True))

        translation_result_list[i].bleu_score = bleu_score(translation_result_list[i].token_list, tgt_text[:len(translation_result_list[i].token_list)])
        with open(translation_result_list[i].name + ".txt", 'w') as f:
            for line in translation_result_list[i].token_list:
                f.write(line)
    
    for t in translation_result_list:
        if t.name.find("warmup") != -1: 
            continue
        print("[INFO] {} translates {} batches taking {:.2f} sec to translate {} tokens, BLEU score: {:.2f}, {:.0f} tokens/sec.".format(
                t.name, t.batch_num, t.execution_time, t.bleu_score.sys_len, t.bleu_score.score, t.bleu_score.sys_len / t.execution_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-max_output_len', '--maximum_output_length', type=int, default=128, metavar='NUMBER',
                        help='maximum output length (default: 128)')
    parser.add_argument("--source", default="../tools/t5_utils/test.en",
                        help="Path to the source file.")
    parser.add_argument("--target", default="../tools/t5_utils/test.de",
                        help="Path to the target file.")
    parser.add_argument('-model', '--model', type=str, default="t5-small", metavar='STRING',
                        help='T5 model size.', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"])
    parser.add_argument('-beam', '--beam_width', type=int, default=1, metavar='NUMBER',
                        help='Beam width for beam search. If setting 1, then using sampling.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    args = parser.parse_args()

    translate(vars(args))
