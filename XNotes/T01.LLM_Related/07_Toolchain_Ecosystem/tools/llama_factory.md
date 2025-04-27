# LLaMA-Factory

## 1. 简介

LLaMA-Factory 是一个用于大模型微调的统一框架，支持多种模型架构和训练策略。它提供了丰富的功能和灵活的配置选项，适用于各种应用场景。其项目目标是整合主流的各种高效训练微调技术，适配市场主流开源模型，形成一个功能丰富、适配性好的训练框架。项目提供了多个高层次抽象的调用接口，包含多阶段训练，推理测试，benchmark评测，API Server 等，使开发者开箱即用。同时借鉴 Stable Diffsion WebUI相关，本项目提供了基于gradio的网页版工作台，方便初学者可以迅速上手操作，开发出自己的第一个模型。

### 框架特性

- **模型种类**：支持LLaMA、LLaVA、Mistral、Mixtral - MoE、Qwen、Yi、Gemma、Baichuan、ChatGLM、Phi等多种大型语言模型。
- **训练算法**：涵盖（增量）预训练、（多模态）指令监督微调、奖励模型训练、PPO训练、DPO训练、KTO训练、ORPO训练等。
- **运算精度**：包括16比特全参数微调、冻结微调、LoRA微调和基于AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ的2/3/4/5/6/8比特QLoRA微调。
- **优化算法**：提供GaLore、BAdam、DoRA、LongLoRA、LLaMA Pro、Mixture - of - Depths、LoRA+、LoftQ和PiSSA等。
- **加速算子**：支持FlashAttention-2和Unsloth。
- **推理引擎**：支持Transformers和vLLM。
- **实验监控**：支持LlamaBoard、TensorBoard、Wandb、MLflow、SwanLab等。

## 2. 主要功能

- **多模型支持**：可对多种预训练模型进行微调，如LLaMA、Qwen、Baichuan等。
- **多阶段训练**：支持预训练、指令微调、基于人工反馈的对齐等全链路训练。
- **可视化操作**：提供基于gradio的网页版工作台，方便初学者操作。
- **多精度微调**：支持不同精度的微调方式，如全参数微调和LoRA微调。
- **高效优化算法**：集成多种优化算法，提高训练效率。
- **实验监控**：可通过多种工具监控训练过程和结果。
- **推理服务**：支持基于vLLM的OpenAI风格API、浏览器界面和命令行接口进行推理。

## 3. 训练数据

> 此部分参考自链接：`https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html`

### 3A. 预训练数据

> 大语言模型通过学习未被标记的文本进行预训练，从而学习语言的表征。通常，预训练数据集从互联网上获得，因为互联网上提供了大量的不同领域的文本信息，有助于提升模型的泛化能力。 预训练数据集文本描述格式如下：

```json
[
  {"text": "document"},
  {"text": "document"}
]
```

`dataset_info.json`中的注册方式：

```json
{
    "数据集名称": {
        "file_name": "数据集文件名",
        "columns": {
            "prompt": "text"
        }
    }
}
```

### 3B. 指令监督微调数据

#### 默认格式

```json
[
  {
    "instruction": "今天的天气怎么样？",
    "input": "",
    "output": "今天的天气不错，是晴天。",
    "images": [
        "image_url_1",
        "image_url_2"
    ],
    "audio": [
        "audio_url_1",
        "audio_url_2"
    ],
    "video": [
        "video_url_1",
        "video_url_2"
    ],
    "history": [
      [
        "今天会下雨吗？",
        "今天不会下雨，是个好天气。"
      ],
      [
        "今天适合出去玩吗？",
        "非常适合，空气质量很好。"
      ]
    ]
  }
]
```

`dataset_info.json`中的注册方式：

```json
"数据集名称": {
    "file_name": "数据集文件名",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "images": "images",
        "audio": "audio",
        "video": "video",
        "system": "system",
        "history": "history"
    }
}
```

#### ShareGPT 格式

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "function_call",
        "value": "工具参数"
      },
      {
        "from": "observation",
        "value": "工具结果"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "system": "系统提示词（选填）",
    "tools": "工具描述（选填）"
  }
]
```

`dataset_info.json`中的注册方式：

```json
"数据集名称": {
    "file_name": "数据集文件名",
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "system": "system",
        "tools": "tools"
    }
}
```

### 3C. 偏好数据

#### 默认格式

```json
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "chosen": "优质回答（必填）",
    "rejected": "劣质回答（必填）"
  }
]
```

`dataset_info.json`中的注册方式：

```json
"数据集名称": {
    "file_name": "数据集文件名",
    "ranking": true,
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "chosen": "chosen",
        "rejected": "rejected"
    }
}
```

#### ShareGPT 格式

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      },
      {
        "from": "human",
        "value": "人类指令"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "优质回答"
    },
    "rejected": {
      "from": "gpt",
      "value": "劣质回答"
    }
  }
]
```

`dataset_info.json`中的注册方式：

```json
"数据集名称": {
    "file_name": "数据集文件名",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
        "messages": "conversations",
        "chosen": "chosen",
        "rejected": "rejected"
    }
}
```

## 4. 训练参数

### 常用参数

> 这个列表中的参数是从官方文档`https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html#id4`获取的，但是内容根据最新代码做了补充完善！

| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| model_name_or_path | str | 模型名称或路径 | 无 |
| stage | Literal[“pt”, “sft”, “rm”, “ppo”, “dpo”, “kto”, “orpo”] | 指定训练阶段，pt表示预训练，sft表示指令监督微调，rm表示奖励模型训练，ppo表示近端策略优化训练，dpo表示直接偏好优化训练，kto表示KTO训练，orpo表示ORPO训练 | sft |
| do_train | bool | 是否进行训练 | False |
| finetuning_type | Literal[“lora”, “freeze”, “full”] | 微调方法，lora是低秩自适应法，冻结原模型参数，只训练新增网络层参数；freeze是参数冻结法，只对原始模型部分参数冻结，仅训练部分参数；full是全参数微调，训练模型的所有参数 | lora |
| lora_target | str | 应用LoRA方法的模块名称，使用逗号分隔多个模块，使用 `all` 指定所有模块 | all |
| dataset | str | 使用的数据集的名称，使用”,”分隔多个数据集 | 无 |
| template | str | 数据集模板，请保证数据集模板与模型相对应 | 无 |
| output_dir | str | 输出路径 | 类似“saves/llama3-8b/lora/pretrain”这种，help显示的是“trainer_output” |
| logging_steps | int | 日志步数，每训练多少步记录一次日志 | 大部分配置文件中都是设置的10，WebUI默认值为5，help显示默认值为500 |
| save_steps | int | 保存步数，每训练多少步保存一次模型 | 大部分配置文件中都是设置的500，WebUI默认值为100, help显示默认值为500 |
| overwrite_output_dir | bool | 是否覆盖输出目录 | 大部分配置文件中都是设置的True，help显示默认值为False |
| per_device_train_batch_size | int | 每个设备上训练的批次大小（批处理大小），指在每次迭代中输入到模型中的样本数量，批处理太大会占用更多的内存（显存） | 需根据硬件情况设置，大部分配置文件中都是设置的`1`, help显示默认值为`8` |
| gradient_accumulation_steps | int | 梯度积累步数，用于在受限的GPU内存情况下，模拟更大的批处理大小 | 需根据硬件情况设置，大部分配置文件中都是设置的`8` |
| max_grad_norm | float | 最大梯度范数，也称为梯度裁剪阈值，用于防止梯度爆炸，通常在0.1到10之间，太小会限制模型学习，太大无法有效防止梯度爆炸；该参数是深度学习训练中非常重要的稳定性保障参数 | 需根据具体情况设置，WebUI中默认值为1.0, help显示默认值为`1` |
| learning_rate | float | 控制模型学习速度的参数，学习率高时，模型学习速度快，但可能导致学习过程不稳定；学习率低时，模型学习速度慢，训练时间长，效率低。常见取值如1e - 1（0.1）用于初期快速探索，1e - 2（0.01）常用于许多标准模型的初始学习率，1e - 3（0.001）适用于接近优化目标时的细致调整，1e - 4（0.0001）用于当模型接近收敛时的微调，5e - 5（0.00005）常见于预训练模型的微调阶段 | 需根据具体情况设置，代码配置中使用的有：`1e-4`、`1e-5`、`5e-5`、`5e-6`, help显示默认值为`5e-05` |
| lr_scheduler_type | Literal[“linear”，“cosine”， “polynomial”， “constant”] | 学习率调度器类型，linear表示线性调度，cosine表示余弦调度，polynomial表示多项式调度，constant表示常数调度 | 需根据具体情况设置，大部分配置文件中都是设置的`cosine`，help显示默认值为`linear` |
| num_train_epochs | float | 训练轮数，对于大语言模型的微调，通常在2到10个epoch之间，轮数过多可能导致过拟合，特别是在小数据集上 | 需根据具体情况设置，大部分配置文件中都是设置的`3.0`, help显示默认值为`3.0` |
| bf16 | bool | 是否使用bf16精度 | 无，大部分配置文件中都是设置的`true`, help显示默认值为`False` |
| compute_type | Literal[“bf16”, “fp16”, “fp32”, “purebf16”] | 计算类型，若硬件支持bf16，且希望最大化内存效率和计算速度，可选择bf16或purebf16；若硬件支持fp16，希望加速训练过程且能接受较低的数值精度，可以选择fp16；如果不确定硬件支持哪些类型，或需要高精度计算，可以选择fp32 | WebUI中默认值为`bf16`, help显示没有这个参数了！！！ |
| warmup_ratio | float | 学习率预热比例 | 大部分配置文件中都是设置的`0.1`，help显示默认值为`0.0` |
| warmup_steps | int | 学习率预热步数，模型训练初期用于逐渐增加学习率的步骤数 | 很少有配置文件中设置，WebUI中默认值为`0`，help显示默认值为`0` |
| push_to_hub | bool | 是否将模型推送到Hub | 这个参数貌似没怎么用了，有一个`export_hub_model_id`貌似是差不多的作用, help显示默认值为`False` |

### 其他参数
#### 模型和适配器相关
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| adapter_name_or_path | str | 适配器的名称或路径 | 无 |
| adapter_folder | str | 适配器文件夹路径 | 无 |
| cache_dir | str | 缓存目录 | 无 |
| use_fast_tokenizer | bool | 是否使用快速分词器 | 无 |
| resize_vocab | bool | 是否调整词汇表大小 | 无 |
| split_special_tokens | bool | 是否拆分特殊标记 | 无 |
| new_special_tokens | str | 新的特殊标记 | 无 |
| model_revision | str | 模型修订版本 | 无 |
| low_cpu_mem_usage | bool | 是否使用低CPU内存 | 无 |

#### 量化和推理相关
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| quantization_method | Literal[“bitsandbytes”, “hqq”, “eetq”] | 量化方法 | 无 |
| quantization_bit | int | 量化位数 | 无 |
| quantization_type | Literal[“fp4”, “nf4”] | 量化类型 | 无 |
| double_quantization | bool | 是否进行双重量化 | 无 |
| quantization_device_map | Literal[“auto”] | 量化设备映射 | 无 |
| rope_scaling | Literal[“linear”, “dynamic”] | ROPE缩放 | 无 |
| flash_attn | Literal[“auto”, “disabled”, “sdpa”, “fa2”] | 闪存注意力设置 | 无 |
| shift_attn | bool | 是否移位注意力 | 无 |
| mixture_of_depths | Literal[“convert”, “load”] | 深度混合策略 | 无 |
| use_unsloth | bool | 是否使用Unsloth | 无 |
| visual_inputs | bool | 是否包含视觉输入 | 无 |

#### 训练相关
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| cutoff_len | int | 截断长度，指在处理输入时，模型能接受的最大标记（token）数量，若输入序列超过这个长度，多余的部分将被截断，确保输入长度不会超出模型的处理能力。对于文本分类任务，通常截断到128或256个标记就足够了，而对于更复杂的任务，如文本生成或翻译，可能需要更长的长度 | 需根据任务设置 |
| lora_rank | int | [LoRA微调](../algorithms/lora.md)的本征维数 `r`，`r`越大可训练的参数越多 | 8 |
| lora_alpha | Optional[int] | LoRA缩放系数，一般情况下为lora_rank * 2 | None |
| lora_dropout | float | LoRA微调中的dropout率 | 0 |
| moe_aux_loss_coef | float | MOE辅助损失系数 | 无 |
| disable_gradient_checkpointing | bool | 是否禁用梯度检查点 | 无 |
| upcast_layernorm | bool | 是否上转换LayerNorm | 无 |
| upcast_lmhead_output | bool | 是否上转换LM头输出 | 无 |
| train_from_scratch | bool | 是否从头开始训练 | 无 |
| infer_backend | Literal[“huggingface”, “vllm”] | 推理后端，huggingface基于Hugging Face平台提供推理API；vllm利用向量化计算加速大模型推理过程 | 无 |
| vllm_maxlen | int | vLLM最大长度 | 无 |
| vllm_gpu_util | float | vLLM GPU利用率 | 无 |
| vllm_enforce_eager | bool | 是否强制启用eager模式 | 无 |
| vllm_max_lora_rank | int | vLLM最大LoRA排名 | 无 |
| offload_folder | str | 离线文件夹路径 | 无 |
| use_cache | bool | 是否使用缓存 | 无 |
| infer_dtype | Literal[“auto”, “float16”, “bfloat16”, “float32”] | 推理数据类型 | 无 |
| hf_hub_token | str | Hugging Face Hub令牌 | 无 |
| ms_hub_token | str | ModelScope Hub令牌 | 无 |

#### 导出相关
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| export_dir | str | 导出目录 | 无 |
| export_size | int | 导出大小 | 无 |
| export_device | Literal[“cpu”, “auto”] | 导出设备 | 无 |
| export_quantization_bit | int | 导出量化位数 | 无 |
| export_quantization_dataset | str | 导出量化数据集 | 无 |
| export_quantization_nsamples | int | 导出量化样本数 | 无 |
| export_quantization_maxlen | int | 导出量化最大长度 | 无 |
| export_legacy_format | bool | 是否导出为遗留格式 | 无 |
| export_hub_model_id | str | 导出到Hub的模型ID | 无 |
| print_param_status | bool | 是否打印参数状态 | 无 |

#### 数据和训练配置
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| dataset_dir | str | 数据集目录 | 无 |
| split | str | 数据集拆分 | 无 |
| train_on_prompt | bool | 是否在提示上训练 | 无 |
| streaming | bool | 是否启用流式传输 | 无 |
| buffer_size | int | 缓冲区大小 | 无 |
| mix_strategy | Literal[“concat”, “interleave_under”, “interleave_over”] | 数据混合策略 | 无 |
| interleave_probs | float | 混合概率 | 无 |
| overwrite_cache | bool | 是否覆盖缓存 | 无 |
| preprocessing_num_workers | int | 预处理工作线程数 | 无 |
| max_samples | int | 最大样本数，它决定了每个数据集中使用多少样本进行训练，如果原始数据集很大，设置一个合理的最大样本数可以减少训练时间，如果计算资源有限，较小的样本数可以加快训练速度 | 无 |
| eval_num_beams | int | 评估时使用的beam数量 | 无 |
| ignore_pad_token_for_loss | bool | 是否在计算损失时忽略填充标记 | 无 |
| val_size | float | 验证集大小 | 无 |
| packing | bool | 是否启用数据打包 | 无 |
| neat_packing | bool | 是否启用整洁打包 | 无 |
| tool_format | str | 工具格式 | 无 |
| tokenized_path | str | 分词后的数据路径 | 无 |

#### 训练过程控制
| 参数名称 | 类型 | 介绍 | 默认值 |
| --- | --- | --- | --- |
| do_eval | bool | 是否进行评估 | 无 |
| do_predict | bool | 是否进行预测 | 无 |
| eval_steps | int | 评估步数，每训练多少步进行一次评估 | 无 |
| weight_decay | float | 权重衰减，用于防止过拟合 | 无 |
| adam_beta1 | float | Adam优化器的beta1参数 | 无 |
| adam_beta2 | float | Adam优化器的beta2参数 | 无 |
| adam_epsilon | float | Adam优化器的epsilon参数 | 无 |
| max_steps | int | 最大训练步数 | 无 |
| seed | int | 随机种子，用于保证训练的可重复性 | 无 |
| local_rank | int | 本地排名，用于分布式训练 | 无 |
| ddp_find_unused_parameters | bool | 是否查找未使用的参数，用于分布式训练 | 无 |
| deepspeed | str | DeepSpeed配置文件路径，用于分布式训练 | 无 |
| fsdp | str | FSDP配置文件路径，用于分布式训练 | 无 |
| neftune_noise_alpha | float | NEFTune（Noise Embedding Finetuning）噪声参数，在微调过程的词向量中引入一些均匀分布的噪声，可明显地提升模型的表现 | 无 |

## 5. 训练方法

### 5A. 常规训练

> 更多训练示例见：`https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/`

- 首先，准备训练数据集，参考`3A`、`3B`、`3C`节；

- 准备训练配置文件，参考`https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_lora/llama3_lora_sft.yaml`

- 执行训练命令，比如：

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

### 5B. 分布式训练

- 多节点分布式训练：

```bash
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.100.2 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.100.2 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

## 6. XXX











## 参考资源

- [LLaMA-Factory 官方文档](https://llamafactory.readthedocs.io/)
- [LLaMA-Factory 源码](https://github.com/hiyouga/LLaMA-Factory)
- [LLaMA-Factory 训练参数列表](./llama_factory_trainer_params.md)
- [LoRA微调](../algorithms/lora.md)
