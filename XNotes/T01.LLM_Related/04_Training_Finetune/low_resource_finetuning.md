# 💸 低资源微调技术

## 📋 低资源微调概述

### 🎯 为什么需要低资源微调

随着大语言模型规模不断扩大，传统全参数微调面临巨大挑战：
- 💾 **内存消耗巨大**：一个70B参数模型全精度微调需要超过140GB显存
- 💰 **硬件成本高昂**：需要多张高端GPU或TPU
- ⏱️ **训练时间长**：全参数微调大模型需要数天到数周
- 🔋 **能耗巨大**：环境影响与电力成本问题

低资源微调技术旨在解决这些问题，使个人开发者、学术研究者和小型团队也能定制化大模型。

### 🌟 低资源微调的主要优势

- 💻 **降低硬件要求**：在消费级GPU甚至CPU上可行
- ⚡ **加速训练过程**：训练速度提升数倍至数十倍
- 💾 **减少存储需求**：模型权重存储空间显著减少
- 🔄 **便于迭代与部署**：快速尝试不同微调策略和模型切换

## 🧩 主要低资源微调技术

### 1. 🔑 LoRA (Low-Rank Adaptation)

**核心原理**：使用低秩分解来表示权重更新，仅训练一小部分参数。

**技术细节**：
- 对原始权重矩阵W的更新被参数化为两个低秩矩阵的乘积：ΔW = A×B
- 矩阵A尺寸为(d×r)，矩阵B尺寸为(r×k)，其中r远小于d和k
- 仅训练这些低秩适配器，保持原始预训练权重冻结

**参数配置重点**：
- **秩(r)**：控制表达能力与参数量，通常8-64
- **alpha**：缩放因子，通常设为r的2倍
- **应用层**：可选择性应用到特定层或特定模块

**代码实现**：
```python
from peft import get_peft_model, LoraConfig, TaskType

# LoRA配置
lora_config = LoraConfig(
    r=16,                        # 低秩矩阵的秩
    lora_alpha=32,               # 缩放参数
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 目标模块
    lora_dropout=0.05,           # LoRA层的Dropout率
    bias="none",                 # 是否包含偏置项
    task_type=TaskType.CAUSAL_LM # 任务类型
)

# 创建PEFT模型
model = get_peft_model(base_model, lora_config)

# 检查可训练参数比例
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params} ({100 * trainable_params / all_params:.2f}%)")
```

**性能指标**：
- 可训练参数比例：通常为原模型的0.1%-1%
- 显存需求：与原模型相比减少80%-95%
- 训练速度：提升2-5倍
- 效果：接近全参数微调，某些任务可达到95%-99%性能

### 2. 🔍 QLoRA (Quantized LoRA)

**核心原理**：将基础模型量化到4位精度，同时使用LoRA进行微调。

**技术亮点**：
- 4位NormalFloat (NF4)量化：针对正态分布权重优化的4位量化格式
- 双重量化技术：通过二次量化进一步压缩内存占用
- 分页优化器：有效管理有限GPU内存

**代码实现**：
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# 4位量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 启用4位量化
    bnb_4bit_quant_type="nf4",        # 使用NF4量化格式
    bnb_4bit_compute_dtype=torch.float16,  # 计算使用half精度
    bnb_4bit_use_double_quant=True    # 启用双重量化
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "llama-65b",
    quantization_config=bnb_config,
    device_map="auto"                 # 自动管理模型在设备间的分布
)

# 应用LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
```

**性能指标**：
- 内存需求：可将65B模型在单张24GB GPU上微调（全参数需要130GB+）
- 训练速度：比全参数慢10%-30%，但比其他低资源方法快
- 效果：经证明达到全参数微调的96%-99%性能

### 3. 📝 参数高效微调其他技术

#### a. 🎯 Prefix/Prompt Tuning

**核心原理**：为模型添加一组可训练的连续向量作为输入前缀。

**优势**：
- 极少量参数（通常<0.1%）
- 任务切换只需替换前缀，不需更改模型
- 训练速度极快

**局限性**：
- 对较小模型效果有限
- 表达能力较LoRA弱

**代码示例**：
```python
from peft import PrefixTuningConfig, get_peft_model

prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,       # 虚拟前缀标记数量
    prefix_projection=True,      # 是否使用前缀投影
    encoder_hidden_size=128      # 投影层隐藏大小
)

prefix_model = get_peft_model(model, prefix_config)
```

#### b. 🧩 Adapter Tuning

**核心原理**：在Transformer层之间插入小型可训练模块（瓶颈层）。

**优势**：
- 模块化设计，灵活性强
- 对任务迁移性能良好
- 可组合不同任务的适配器

**代码示例**：
```python
from peft import AdapterConfig, get_peft_model

adapter_config = AdapterConfig(
    r=64,                         # 适配器维度大小
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

adapter_model = get_peft_model(model, adapter_config)
```

### 4. 📊 模型量化技术

模型量化是低资源微调的重要辅助技术，通过降低权重精度减少内存需求。

#### a. 🔢 8位量化 (INT8)

**特点**：
- 将FP32/FP16权重转换为INT8
- 显存需求减少50%-60%
- 推理速度通常提升20%-40%
- 精度损失极小（<1%）

**实现方式**：
```python
from transformers import AutoModelForCausalLM

# 加载8位量化模型
model = AutoModelForCausalLM.from_pretrained("llama-7b", load_in_8bit=True)
```

#### b. 🧮 4位量化 (INT4/NF4)

**特点**：
- 显存需求减少70%-80%
- 支持GPTQ和bitsandbytes两种量化方法
- NF4比INT4更适合语言模型权重分布
- 微调场景下结合QLoRA使用效果最佳

**实现方式**：
```python
from transformers import BitsAndBytesConfig

# 4位量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",   # 或使用"fp4"
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "llama-13b", 
    quantization_config=quant_config
)
```

## 🧪 低资源微调实践案例

### 1. 💻 单GPU上微调70B模型

**场景**：在RTX 4090（24GB显存）上微调Llama-2-70B

**解决方案**：
1. 使用QLoRA + NF4量化
2. 设置适当的批量大小和梯度累积步数
3. 使用Flash Attention优化注意力计算

**关键配置**：
```python
# QLoRA配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    output_dir="outputs",
    save_strategy="steps",
    save_steps=100
)
```

**效果**：
- 显存峰值：约21GB
- 训练速度：约180秒/100步（batch_size=4）
- 性能：约为全参数微调的98%

### 2. 📱 手机/边缘设备微调

**场景**：在低端设备（8GB内存）上微调小型模型（1-2B参数）

**解决方案**：
1. INT4量化 + 极小LoRA配置（r=4, alpha=8）
2. 使用梯度检查点和内存高效优化器
3. 优先选择较小的基础模型（Phi-1.5, Gemma-2B等）

**关键技术**：
- 使用CPU微调，避免显存限制
- 8位量化优化器减少内存占用
- 每次只加载和处理少量数据

**代码示例**：
```python
# 加载量化模型（CPU模式）
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-1_5", 
    load_in_4bit=True,
    device_map="cpu"
)

# 极小LoRA配置
config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 8位优化器配置
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
```

## 🔧 低资源微调最佳实践

### 1. 📏 参数配置优化

**LoRA关键参数选择**：
- 推荐秩(r)取值：
  - 对话微调：r=8~16通常足够
  - 特定领域适应：r=32~64效果更好
  - 复杂任务学习：考虑r=128以上

- Alpha值与稳定性：
  - 经验法则：alpha=2×r是不错的起点
  - 较小alpha（如alpha=r）训练更稳定但学习能力弱
  - 较大alpha（如alpha=4×r）学习能力强但可能不稳定

- 目标模块选择：
  - 全模块应用：`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - 核心模块应用：`["q_proj", "v_proj"]`
  - 仅注意力：`["q_proj", "k_proj", "v_proj", "o_proj"]`

### 2. 🔄 训练策略优化

**批量大小与学习率**：
- 较大批量通常需要较高学习率
- 推荐学习率区间：1e-5至2e-4
- 批量大小受限时增加梯度累积步数

**学习率调度**：
- 对低资源微调，余弦学习率衰减通常效果好
- 包含适当预热步数（总步数的1%-3%）
- 可考虑学习率在训练后期重启以跳出局部最优

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=2e-4)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)
```

### 3. 💾 数据效率优化

**数据质量优先策略**：
- 低资源环境下，高质量数据价值远超大量低质量数据
- 通常3000-5000高质量样本已足够获得良好效果
- 考虑使用模型自己生成更多样本（基于少量高质量种子数据）

**数据增强技术**：
- 回译增强：通过多语言翻译增加表达多样性
- 同义替换：替换关键词为同义词创建变体
- 提示多样化：同一内容用不同提问方式表达

### 4. 🧩 模型融合与组合

**多个LoRA适配器融合**：
- 可将多个专业领域LoRA组合创建多功能模型
- 支持加权融合以平衡不同能力

```python
from peft import PeftModel

# 加载多个LoRA适配器
model1 = PeftModel.from_pretrained(base_model, "medical-lora")
model2 = PeftModel.from_pretrained(base_model, "coding-lora")

# 加权融合
merged_model = model1.merge_and_unload(model2, alpha=0.7)  # 70% model1, 30% model2
```

## 📱 低资源微调部署优化

### 1. 📊 推理优化技术

**ONNX转换**：
- 将PyTorch模型转换为ONNX格式
- 可显著提升CPU推理性能（30%-200%）
- 支持更广泛设备部署

```python
from transformers.onnx import export
from pathlib import Path

# 导出ONNX模型
export(
    preprocessor=tokenizer,
    model=model,
    opset=13,
    output=Path("./model.onnx")
)
```

**量化推理**：
- 将微调后模型进一步量化为INT8/INT4
- 使用GPTQ或AWQ等算法保持精度
- 集成KV缓存量化进一步提速

### 2. ⚡ 边缘设备部署

**模型压缩策略**：
- 知识蒸馏：将大模型知识迁移到小模型
- 结构剪枝：移除不重要连接/神经元
- 低位量化：INT4/INT2甚至二进制量化

**轻量级推理框架**：
- GGML/GGUF：高效C++推理库
- llama.cpp：优化的CPU推理引擎
- MLC-LLM：面向移动设备的优化框架

### 3. 🔍 内存优化技术

**激活检查点**：
- 在前向传播时丢弃中间激活值，反向传播时重新计算
- 大幅减少内存占用，代价是计算量增加

**选择性计算**：
- KV缓存复用：避免重复计算历史token的键值
- 仅保留和计算必要层的激活值

## 📚 扩展资源与工具

### 1. 🛠️ 常用工具库

- [PEFT](https://github.com/huggingface/peft)：参数高效微调工具库
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)：量化和优化工具
- [transformers](https://github.com/huggingface/transformers)：基础模型加载与处理
- [accelerate](https://github.com/huggingface/accelerate)：分布式训练与混合精度工具
- [llama.cpp](https://github.com/ggerganov/llama.cpp)：高效CPU推理

### 2. 📝 学习资源

- [QLoRA论文](https://arxiv.org/abs/2305.14314)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Transfer Learning综述](https://arxiv.org/abs/2303.15647)
- [低资源NLP技术指南](https://huggingface.co/blog/low-resource-nlp) 