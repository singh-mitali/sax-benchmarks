# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serving model parameters for lm_cloud."""

import os
from typing import List, cast

import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import tasks_lib
from paxml.tasks.lm.params import lm_cloud
from praxis import base_input
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.layers import activations
from praxis.layers import multi_query_attention
from saxml.server import servable_model_registry
from saxml.server.pax import quantization
from saxml.server.pax.lm.layers import LLaMARotaryEmbedding
from saxml.server.pax.lm.layers import ParallelTransformer
from saxml.server.pax.lm.params import template
#from saxml.server.pax.lm.experimental.params import llm_xla_flags

@template.make_servable()
class BaseLLaMA(base_experiment.BaseExperiment):
  """Base LLaMA Transformer LM configuration."""

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'
  SOS_ID = 1
  EOS_ID = 2

  # architecture related
  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.bfloat16
  MODEL_DTYPE = jnp.bfloat16
  USE_MQA = False

  ACTIVATION_CLS = activations.SiLU
  USE_GATED_ACTIVATION = True
  RMS_NORM_EPSILON = 1.0e-05



  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 1, 1]
  DCN_MESH_SHAPE = None
  DECODE_MESH_TRANSPOSE = None

  BATCH_SIZE = 1
  NUM_SAMPLES = 1
  ENABLE_GENERATE_STREAM = True
  STREAM_INTERVAL_STEPS = 16
  FPROP_FOR_PREFIX = True
  INPUT_SEQ_LEN = 4096
  BUCKET_KEYS = [128, 1024, 4096]
  MAX_DECODE_STEPS = [128, 512, 1024]
  EXTRA_INPUTS = {
      'temperature': 0.5,
      'per_example_max_decode_steps': 128,
      'per_example_top_k': 200,
      'per_example_top_p': 0.95,
  }

  def model(self):
    model_p = pax_fiddle.Config(layers.LanguageModel, name='xformer_lm')
    model_p.lm_tpl.packed_input = False
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE
    model_p.lm_tpl.position_emb_tpl = None
    model_p.lm_tpl.softmax_tpl = pax_fiddle.Config(
        layers.FullSoftmax,
        name='output',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
    model_p.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(
        layers.Embedding,
        name='tok_embeddings',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    ln_tpl = pax_fiddle.Config(
        layers.RmsNorm,
        name='norm',
        direct_scale=True,
        epsilon=self.RMS_NORM_EPSILON,
    )
    model_p.lm_tpl.final_ln_tpl = ln_tpl.clone()

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    transformer_layer_p = cast(
        pax_fiddle.Config[layers.Transformer],
        stacked_transformer_tpl.transformer_layer_params_tpl,
    )
    transformer_layer_p.norm_policy = 'pre'
    transformer_layer_p.ln_tpl = ln_tpl.clone()

    if self.USE_MQA:
      transformer_layer_p.tr_atten_tpl = pax_fiddle.Config(
          multi_query_attention.MultiQueryDotProductAttention,
          num_kv_heads=self.NUM_KV_HEADS,
      )
      transformer_layer_p.tr_atten_tpl.combine_qkv = False
    else:
      transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
      transformer_layer_p.tr_atten_tpl.combine_qkv = True

    transformer_layer_p.tr_atten_tpl.internal_enable_query_scale = True
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.rotary_position_emb_tpl = (
        pax_fiddle.Config(LLaMARotaryEmbedding)
    )
    transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    transformer_layer_p.tr_fflayer_tpl.has_bias = False
    transformer_layer_p.tr_fflayer_tpl.ln_tpl = ln_tpl.clone()
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION
    )

    model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    model_p.fprop_dtype = self.FPROP_DTYPE
    model_p.dtype = self.MODEL_DTYPE
    return model_p

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    return []

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='xformer_task')
    model_p = self.model()
    task_p.model = model_p

    # Set sharding
    task_p = template.set_decoding_sharding_hparams(
        task_p,
        mesh_shape=self.ICI_MESH_SHAPE,
        decode_mesh_transpose=self.DECODE_MESH_TRANSPOSE,
    )
    # Unused.
    lp = task_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = pax_fiddle.Config(
        optimizers.ShardedSgd,
        learning_rate=1e-3,
        lr_schedule=pax_fiddle.Config(schedules.Constant)
    )
    return task_p


@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuant(BaseLLaMA):
  # Chan these params with customer config
  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008

  NUM_KV_HEADS = 8
  USE_MQA = True

  ICI_MESH_SHAPE = [1, 1, 8]

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'

  INPUT_SEQ_LEN = 4096
  BUCKET_KEYS = [4096]
  MAX_DECODE_STEPS = 128
  ENABLE_GENERATE_STREAM = False

  BATCH_SIZE = 8
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS8ICI(LLaMA33BQuant):
  BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 8, 1]
  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS_8_16_32(LLaMA33BQuant):
  BATCH_SIZE = [8, 16, 32]
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS8BKT(LLaMA33BQuant):
  BATCH_SIZE = 8
  BUCKET_KEYS = [128, 1024, 4096]
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS8Xla(LLaMA33BQuant):
  BATCH_SIZE = 8
  # Compiler options.
  XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
    }
  MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    }
  def compiler_options(self) -> jax.stages.CompilerOptions:
    return self.XLA_TPU_FLAGS | self.MBLO_FLAGS

  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS8BktXla(LLaMA33BQuant):
  BATCH_SIZE = 8
  BUCKET_KEYS = [128, 1024, 4096]
  # Compiler options.
  XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
    }
  MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    }
  def compiler_options(self) -> jax.stages.CompilerOptions:
    return self.XLA_TPU_FLAGS | self.MBLO_FLAGS

  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS8DMT(LLaMA33BQuant):
  BATCH_SIZE = 8
  DECODE_MESH_TRANSPOSE = {
      'fprop_mdl': 'mdl',
      'mdl': 'fprop_mdl',
  }
  @classmethod
  def serving_mesh_shape(cls) -> list[int]:
    return [1, 1, 1, 1, 8]
    # replica, data_mdl2, mdl, fprop_data, fprop_mdl
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS8DMTXLA(LLaMA33BQuant):
  BATCH_SIZE = 8
  DECODE_MESH_TRANSPOSE = {
      'fprop_mdl': 'mdl',
      'mdl': 'fprop_mdl',
  }
  BUCKET_KEYS = [4096]
  # Compiler options.
  XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
    }
  MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    }
  def compiler_options(self) -> jax.stages.CompilerOptions:
    return self.XLA_TPU_FLAGS | self.MBLO_FLAGS
  @classmethod
  def serving_mesh_shape(cls) -> list[int]:
    return [1, 1, 1, 1, 8]
    # replica, data_mdl2, mdl, fprop_data, fprop_mdl
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS16DMTXLA(LLaMA33BQuant):
  BATCH_SIZE = 16
  DECODE_MESH_TRANSPOSE = {
      'fprop_mdl': 'mdl',
      'mdl': 'fprop_mdl',
  }
  BUCKET_KEYS = [4096]
  # Compiler options.
  XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
    }
  MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    }
  def compiler_options(self) -> jax.stages.CompilerOptions:
    return self.XLA_TPU_FLAGS | self.MBLO_FLAGS
  @classmethod
  def serving_mesh_shape(cls) -> list[int]:
    return [1, 1, 1, 1, 8]
    # replica, data_mdl2, mdl, fprop_data, fprop_mdl
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS16XLAWAIT(LLaMA33BQuant):
  BATCH_SIZE = 16
  BATCH_WAIT_SECS = 4.0
  BUCKET_KEYS = [4096]
  # Compiler options.
  XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
    }
  MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    }
  def compiler_options(self) -> jax.stages.CompilerOptions:
    return self.XLA_TPU_FLAGS | self.MBLO_FLAGS
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS32XLAWAIT(LLaMA33BQuant):
  BATCH_SIZE = 32
  BATCH_WAIT_SECS = 4.0
  BUCKET_KEYS = [4096]
  # Compiler options.
  XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
    }
  MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    }
  def compiler_options(self) -> jax.stages.CompilerOptions:
    return self.XLA_TPU_FLAGS | self.MBLO_FLAGS
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS8DMTXLABKT(LLaMA33BQuant):
  BATCH_SIZE = 8
  BUCKET_KEYS = [128, 1024, 4096]
  DECODE_MESH_TRANSPOSE = {
      'fprop_mdl': 'mdl',
      'mdl': 'fprop_mdl',
  }
  # Compiler options.
  XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
    }
  MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    }
  def compiler_options(self) -> jax.stages.CompilerOptions:
    return self.XLA_TPU_FLAGS | self.MBLO_FLAGS
  @classmethod
  def serving_mesh_shape(cls) -> list[int]:
    return [1, 1, 1, 1, 8]
    # replica, data_mdl2, mdl, fprop_data, fprop_mdl
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS16DMTXLABKTWAIT(LLaMA33BQuant):
  BATCH_SIZE = 16
  BATCH_WAIT_SECS = 4.0
  BUCKET_KEYS = [128, 1024, 4096]
  DECODE_MESH_TRANSPOSE = {
      'fprop_mdl': 'mdl',
      'mdl': 'fprop_mdl',
  }
  # Compiler options.
  XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
    }
  MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    }
  def compiler_options(self) -> jax.stages.CompilerOptions:
    return self.XLA_TPU_FLAGS | self.MBLO_FLAGS
  @classmethod
  def serving_mesh_shape(cls) -> list[int]:
    return [1, 1, 1, 1, 8]
    # replica, data_mdl2, mdl, fprop_data, fprop_mdl
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS32DMTXLABKTWAIT(LLaMA33BQuant):
  BATCH_SIZE = 32
  BATCH_WAIT_SECS = 4.0
  BUCKET_KEYS = [128, 1024, 4096]
  DECODE_MESH_TRANSPOSE = {
      'fprop_mdl': 'mdl',
      'mdl': 'fprop_mdl',
  }
  # Compiler options.
  XLA_TPU_FLAGS = {
    'xla_jf_auto_cross_replica_sharding': 'False',
    'xla_tpu_nd_short_transfer_max_chunks': '2048',
    'xla_tpu_perform_spmd_cse_prevention': 'True',
    'xla_tpu_rwb_fusion': 'False',
    }
  MBLO_FLAGS = {
    'xla_tpu_enforce_prefetch_fifo_order': 'true',
    'xla_tpu_memory_bound_loop_optimizer_options': 'enabled:true',
    }
  def compiler_options(self) -> jax.stages.CompilerOptions:
    return self.XLA_TPU_FLAGS | self.MBLO_FLAGS
  @classmethod
  def serving_mesh_shape(cls) -> list[int]:
    return [1, 1, 1, 1, 8]
    # replica, data_mdl2, mdl, fprop_data, fprop_mdl
  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS16(LLaMA33BQuant):
  BATCH_SIZE = 16
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS32(LLaMA33BQuant):
  BATCH_SIZE = 32
  @property
  def test_mode(self) -> bool:
    return True


@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS16ICI(LLaMA33BQuant):
  BATCH_SIZE = 16
  ICI_MESH_SHAPE = [1, 8, 1]
  @property
  def test_mode(self) -> bool:
    return True

@servable_model_registry.register
@quantization.for_transformer()
class LLaMA33BQuantBS32ICI(LLaMA33BQuant):
  BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 8, 1]
  @property
  def test_mode(self) -> bool:
    return True
