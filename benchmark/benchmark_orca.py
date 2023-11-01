# pip install transformers; pip install SentencePiece; pip install huggingface_hub; pip install pandas
# export SAX_ROOT=gs://msingh-sax/sax-root

import tensorflow as tf
import os, sys
sys.path.append('/saxml/bazel-bin/saxml/client/python/')
import sax

import queue
import huggingface_hub
huggingface_hub.login(token="")
from transformers import LlamaTokenizer
import numpy as np
import pandas
import threading
import multiprocessing as mp
import time
import argparse

output_queue = queue.Queue()
thread_times_queue = queue.Queue()
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
option = sax.ModelOptions()
option.SetExtraInput("temperature", 0)
option.SetExtraInput("per_example_max_decode_steps", 128)

DATAFILE = "gs://msingh-sax-data/datasets/open_orca_gpt4_50k_filtered_tokenized_llama_prompt.pkl"

def get_dataset(file_name):
  dataset = tf.data.TFRecordDataset(file_name)
  count = sum([1 for _ in dataset])
  return dataset, count

def lm_predict(prompt):
  res = lm_model.Generate(prompt, option)
  return res

def register_sax_model(model_id):
  model = sax.Model(model_id)
  global lm_model
  lm_model = model.LM()

def process_data(batch):
  option = sax.ModelOptions()
  output_list = []
  option.SetExtraInput("per_example_max_decode_steps", 128)
  for prompt in batch:
    predicted = lm_model.Generate(prompt, option)
    output_list.append((prompt, predicted[0][0]))
  return output_list

def create_prompt_data(filename):
  df = pandas.read_pickle(filename)
  return df["input"].to_list()

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--num_batches', type=int, default=32)
  parser.add_argument('-b', '--batch_size', type=int, default=8)
  parser.add_argument('-t', '--num_threads', type=int, default=32)
  args = parser.parse_args()

  # Fetch model server and warm up.
  register_sax_model("/sax/test/llama33b")
  lm_predict("warm up")

  prompts = create_prompt_data(DATAFILE)
  num_prompts = args.num_batches * args.batch_size
  prompts = prompts[:num_prompts]
  output_queue = []
  start = time.time()
  batched_data = []
  for i in range(0, args.num_batches):
    batched_data.append(prompts[i:i+args.batch_size])

  total_input_tokens = 0
  total_output_tokens = 0
  with mp.pool.ThreadPool(processes=args.num_threads) as pool:
    for result in pool.map(process_data, batched_data):
      output_queue.append(result)
  total_time = time.time() - start

  req_count = 0
  for output_list in output_queue:
    for result in output_list:
      num_input_tokens = len(tokenizer.encode(result[0]))
      num_output_tokens = len(tokenizer.encode(result[1]))
      total_input_tokens += num_input_tokens
      total_output_tokens += num_output_tokens
      req_count +=1
      # print(f"Request {req_count}: promptlen={num_input_tokens},  predictedlen={num_output_tokens}")

  qps = total_time / len(prompts)
  output_tokens_per_sec = total_output_tokens / total_time
  batch_latency = total_time / len(batched_data)
  query_latency = total_time / len(prompts)
  avg_input_len = total_input_tokens / len(prompts)
  avg_output_len = total_output_tokens / len(prompts)


  print("ClientThreadBatchSize, NumBatches, Threads, Time, QPS, OutTokenPerSec, Batch Latency(s), Query Latency (s), AvgInputLen, AvgOutputLen")
  print(f"{args.batch_size}, {len(batched_data)}, {args.num_threads}, {total_time}, {qps}, {output_tokens_per_sec}, {batch_latency}, {query_latency}, {avg_input_len}, {avg_output_len}")

if __name__ == "__main__":
    main()
