# 📊 大模型专业术语表

## 📝 基础术语

- 🧠 **LLM (Large Language Model)**: 大型语言模型，指参数规模庞大的语言模型
- 🔄 **Transformer**: 一种基于自注意力机制的神经网络架构，是现代大模型的基础
- 🧩 **Token**: 文本的基本单位，可以是单词、子词或字符
- 📊 **Embedding**: 将文本转换为稠密向量表示的过程或结果
- 🔍 **Attention**: 注意力机制，允许模型关注输入序列中的特定部分
- 🧮 **Parameters**: 参数，模型中可学习的变量
- 📈 **Perplexity**: 困惑度，评估语言模型预测能力的指标，越低越好

## 🔬 模型架构相关

- 🔄 **Encoder**: 编码器，将输入序列转换为向量表示
- 🔄 **Decoder**: 解码器，从向量表示生成输出序列
- 🔄 **Encoder-Decoder**: 编码器-解码器架构，用于序列到序列任务
- 🔄 **Decoder-only**: 仅解码器架构，如GPT系列
- 🧪 **MLP (Multi-Layer Perceptron)**: 多层感知机，神经网络的基本组件
- 🧮 **FFN (Feed-Forward Network)**: 前馈网络，Transformer架构的组件之一
- 🔄 **Self-Attention**: 自注意力，计算序列内元素之间的关联
- 🔀 **Multi-Head Attention**: 多头注意力，并行计算多组注意力

## 💻 训练与优化术语

- 🎯 **Pre-training**: 预训练，在大规模无标注数据上训练基础模型
- 🔧 **Fine-tuning**: 微调，在特定任务数据上调整预训练模型
- 📝 **Instruction Tuning**: 指令微调，使模型更好地遵循指令
- 👥 **RLHF (Reinforcement Learning from Human Feedback)**: 基于人类反馈的强化学习
- 🔄 **PPO (Proximal Policy Optimization)**: 近端策略优化，一种常用于RLHF的强化学习算法
- 🧪 **SFT (Supervised Fine-Tuning)**: 监督微调，RLHF的第一阶段
- 🏆 **Preference Optimization**: 偏好优化，根据人类偏好调整模型输出
- 📉 **Loss Function**: 损失函数，衡量模型预测与目标之间差距的函数
- 🔄 **Gradient Descent**: 梯度下降，模型参数优化算法
- 🦙 **LoRA (Low-Rank Adaptation)**: 低秩适应，一种参数高效的微调方法
- 📌 **QLoRA**: 量化版LoRA，进一步减少内存需求
- 🧪 **P-Tuning**: 提示调整，一种参数高效的微调方法
- 🔧 **Adapter**: 适配器，在原模型中插入小型可训练模块的微调方法

## 📊 评估与能力相关

- 📋 **Few-shot Learning**: 少样本学习，模型通过少量示例学习新任务的能力
- 📋 **Zero-shot Learning**: 零样本学习，无需示例即可完成任务的能力
- 💫 **Emergent Abilities**: 涌现能力，随着模型规模增长突然出现的新能力
- 🧿 **Hallucination**: 幻觉，模型生成看似合理但实际不正确的内容
- 🎯 **Alignment**: 对齐，确保模型行为符合人类意图和价值观
- 🛡️ **Guardrails**: 护栏，限制模型输出不当内容的技术措施
- 📏 **MMLU**: 大规模多任务语言理解，评估模型知识广度的基准测试
- 📊 **HELM**: 全面语言模型评估，评估模型多方面能力的框架

## 🚀 应用与部署相关

- 📝 **Prompt Engineering**: 提示工程，设计有效提示以引导模型生成所需输出
- 🔗 **CoT (Chain-of-Thought)**: 思维链，引导模型逐步推理解决问题
- 🌐 **Context Window**: 上下文窗口，模型能处理的最大输入长度
- 🔍 **RAG (Retrieval-Augmented Generation)**: 检索增强生成，结合外部知识源增强模型输出
- 🤖 **Agent**: 智能体，具有规划、记忆和工具使用能力的LLM系统
- 🧙 **ReAct**: 推理和行动相结合的框架，增强LLM解决复杂任务能力
- 🔄 **In-context Learning**: 上下文学习，模型从提示中的示例学习任务的能力
- 📉 **Quantization**: 量化，降低模型精度以减少内存和计算需求
- 🚀 **Inference**: 推理，模型生成输出的过程
- 🔒 **KV Cache**: 键值缓存，加速自回归生成的优化技术
- 🧩 **MoE (Mixture of Experts)**: 专家混合，将模型拆分为专门子网络的架构
- 🔄 **Streaming**: 流式输出，逐步返回模型生成结果 