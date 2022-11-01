from google.protobuf import text_format
import model_config_pb2
import subprocess
import glob 
import os

conf_path = os.environ.get('CONF_PATH', "/model/fastertransformer/config.pbtxt")

# ./gpt_gemm <batch_size> <beam_width> <max_input_len> <head_number> <size_per_head> <inter_size> <vocab_size> <data_type> <tensor_para_size>

with open(conf_path) as f:
    conf : model_config_pb2.ModelConfig = model_config_pb2.ModelConfig()
    text_format.Parse(f.read(), conf)
    params = conf.parameters
    head_num = params['head_num'].string_value
    size_per_head = params['size_per_head'].string_value
    tensor_para_size = params['tensor_para_size'].string_value
    vocab_size = params['vocab_size'].string_value
    inter_size = params['inter_size'].string_value
    # max_batch_size = conf.max_batch_size
    # Data Type = 0 (FP32) or 1 (FP16) or 2 (BF16)
    data_type = os.environ.get('DATA_TYPE', 0) 
    batch_size = os.environ.get('BATCH_SZ', 1)
    max_input_len = os.environ.get('MAX_ILEN', 100)
    beam_width = os.environ.get('BEAM_W', 1)
    cmnd = f"/workspace/gpt_gemm {batch_size} {beam_width} {max_input_len} {head_num} {size_per_head} {inter_size} {vocab_size} {data_type} {tensor_para_size}"
    retcode = subprocess.call(cmnd.split())
    if retcode == 0:
        print("[TUNE] : tuning succesfull")
    else:
        print("[TUNE] : gpt_gemm returned with nonzero status")