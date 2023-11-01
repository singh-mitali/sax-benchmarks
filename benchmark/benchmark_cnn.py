# pip install transformers; pip install SentencePiece; pip install huggingface_hub
import tensorflow as tf
import os, sys
sys.path.append('/saxml/bazel-bin/saxml/client/python/')
import sax

import json
import threading
import datetime
import queue
import huggingface_hub
huggingface_hub.login(token="")
from transformers import LlamaTokenizer

original_filenames = [
  "gs://test-example-123/datasets/cnn_dailymail/3.4.0/cnn_dailymail-test.tfrecord-00000-of-00001"
]
output_queue = queue.Queue()
thread_times_queue = queue.Queue()
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
option = sax.ModelOptions()
option.SetExtraInput("temperature", 0)
option.SetExtraInput("per_example_max_decode_steps", 128)

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

def process_data(batch_idx, dataset):
    targets = []
    predictions = []
    all_res = []
    processed_data = 0
    local_predicted_list = []
    for i in range(len(dataset)):
      original_record = dataset[i]
      original_example_proto = tf.train.Example()
      original_example_proto.ParseFromString(original_record)

      res = {}

      for key, feature in original_example_proto.features.feature.items():
        if key not in ["article", "highlights"]:
          continue
        kind = feature.WhichOneof('kind')
        feature_value = getattr(feature, kind).value[0]
        if key == "article":
          strs_to_join = ['summarize:', feature_value.decode("utf-8")]
          prompt = " ".join(strs_to_join)
          res['article'] = prompt
          processed_data += 1
          predicted = lm_predict(prompt)
          local_predicted_list.append((prompt, predicted))

    output_queue.put(local_predicted_list)
    print("Thread idx {}: processed {} example".format(batch_idx, processed_data))

def main():
  # Fetch model server and warm up.
  register_sax_model("/sax/test/llama33b")
  lm_predict("warm up")

  count = 0
  N = 256
  for original_filename in original_filenames:
    count += 1
    original_dataset, num_original_examples = get_dataset(original_filename)
    print("Processing {} examples in the file: {}".format(num_original_examples, original_filename))
    original_dataset = original_dataset if not N else original_dataset.take(N)
    num_original_examples = sum([1 for _ in original_dataset])
    print(original_dataset)
    print(f"take {num_original_examples} samples from data")

    # Multi-thread
    num_threads = 32
    per_batch_samples = int(num_original_examples/num_threads)
    batch_datasets = list(original_dataset.batch(per_batch_samples).as_numpy_iterator())
    print(f"batch_datasets: {len(batch_datasets)}")
    threads = []
    start = datetime.datetime.now()
    print(f"start time: {start}")

    for i in range(len(batch_datasets)):
      t = threading.Thread(target=process_data, args=(i, batch_datasets[i]))
      t.start()
      threads.append(t)

    print("Waiting for threads to join...")
    for t in threads:
      t.join()

    end = datetime.datetime.now()
    print(f"end time: {end}")
    delta = end - start
    print(f"time cost: {delta.total_seconds()} seconds")

    thread_tokens = []
    while not output_queue.empty():
        thread_results = output_queue.get()
        num_t_tokens = 0
        num_req = 0
        num_req_tokens = 0
        for (p, r) in thread_results:
            num_req += 1
            num_prompt_tokens = len(tokenizer.encode(p))
            num_decode_tokens = len(tokenizer.encode(r[0][0]))
            num_t_tokens += num_decode_tokens
            print(f"Request {num_req}: num_prompt_tokens {num_prompt_tokens} num_decode_tokens {num_decode_tokens}")
        thread_tokens.append(num_t_tokens)

    print(f"Total decode length: {sum(thread_tokens)}")
    print(f"Average decode speed as tokens per sec: {sum(thread_tokens) / delta.total_seconds()}")

if __name__ == "__main__":
    main()
