# 📋 大模型数据准备指南

## 📊 数据准备概述

### 🎯 数据质量的关键性

在大语言模型的训练和微调过程中，**数据质量**是决定模型性能的最关键因素之一。即使是最先进的模型架构，在低质量数据上训练的结果也无法令人满意。数据准备阶段的投入将直接影响模型的：

- 🧠 **知识覆盖范围**
- 🔍 **推理能力**
- 🗣️ **语言流畅度**
- 🛡️ **安全性与偏见控制**
- 🎯 **领域适应性**

### 💼 数据准备流程概览

完整的数据准备流程通常包括以下环节：

1. **数据收集** - 从多种来源获取原始数据
2. **数据清洗** - 去除噪声、重复和低质量内容
3. **数据过滤** - 根据质量指标筛选数据
4. **数据转换** - 将数据转换为模型可用格式
5. **数据增强** - 扩充数据量和多样性
6. **数据分割** - 划分训练、验证和测试集
7. **数据标注** - 为特定任务添加标签（如需）
8. **数据格式化** - 按特定格式组织数据

## 🧩 不同训练阶段的数据需求

### 1. 📚 预训练阶段数据

预训练阶段需要**大规模、多样化**的文本语料库，以使模型学习语言的基本结构和广泛知识。

**典型数据来源**：
- 📰 **网页文本**：CommonCrawl等大规模网络爬虫数据
- 📕 **书籍**：如BookCorpus、Gutenberg项目等
- 📝 **学术论文**：arXiv、PubMed等学术文献
- 📜 **百科全书**：维基百科等结构化知识来源
- 💻 **代码**：GitHub、Stack Overflow等代码仓库
- 🌐 **多语言资源**：各语言的语料库

**预训练数据量级**：
- 小型模型（1-3B参数）：约100GB-500GB文本
- 中型模型（7-13B参数）：约1TB-4TB文本
- 大型模型（70B+参数）：5TB以上文本

### 2. 🔄 指令微调阶段数据

指令微调阶段需要**高质量的指令-响应对**，以教会模型按照人类指令生成回答。

**典型数据结构**：
```json
{
  "instruction": "解释量子纠缠的基本原理",
  "input": "",
  "output": "量子纠缠是量子力学中的一种现象，指的是两个或多个粒子的量子状态不能独立描述..."
}
```

**常用指令微调数据集**：
- **Alpaca** (52K条指令数据)
- **Stanford Human Preferences (SHP)**
- **Natural Instructions**
- **Anthropic Helpful and Harmless (HH)** 
- **FLAN** 集合

**指令微调数据量级**：
- 基础指令跟随能力：10K-100K条高质量样本
- 专业领域适应：1K-10K条领域相关样本

### 3. 🏆 RLHF阶段数据

RLHF（基于人类反馈的强化学习）阶段需要**人类偏好标注数据**，通常是同一提示下多个回答的排序或比较。

**典型数据结构**：
```json
{
  "prompt": "讨论人工智能发展的伦理问题",
  "chosen": "人工智能的发展带来了许多伦理考量，包括隐私保护、算法偏见...",
  "rejected": "人工智能挺好的，没什么问题，应该随便发展就行。"
}
```

**RLHF数据特点**：
- 需要人类评价者参与数据生成
- 通常数据量较小但质量要求极高
- 需涵盖多种场景和回答类型

**RLHF数据量级**：
- 人类偏好数据：5K-50K条人工排序的回答对
- 奖励模型训练：10K-100K条比较样本

## 🛠️ 数据处理关键技术

### 1. 📊 数据清洗技术

**文本去重**：
```python
# 使用MinHashLSH进行近似去重
from datasketch import MinHash, MinHashLSH

def deduplicate(texts, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_texts = []
    
    for idx, text in enumerate(texts):
        minhash = MinHash(num_perm=128)
        for word in text.split():
            minhash.update(word.encode('utf-8'))
        
        if not lsh.query(minhash):
            lsh.insert(idx, minhash)
            unique_texts.append(text)
    
    return unique_texts
```

**质量过滤**：
```python
def filter_by_quality(texts, min_length=50, max_length=10000):
    filtered = []
    for text in texts:
        # 基本长度过滤
        if len(text) < min_length or len(text) > max_length:
            continue
            
        # 语言检测
        if detect_language(text) != 'desired_language':
            continue
            
        # 内容质量得分
        quality_score = compute_quality(text)
        if quality_score < threshold:
            continue
            
        filtered.append(text)
    return filtered
```

**常见清洗操作**：
- 🔍 HTML标记、JavaScript代码移除
- 🧹 特殊字符、控制字符处理
- 🚫 广告、导航栏等无意义内容过滤
- 🌐 URL、邮箱地址规范化
- ♻️ 重复段落、句子去重
- 📏 过长/过短文本过滤

### 2. 💯 数据质量评估

**内容质量得分**：
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载语言质量评估模型
tokenizer = AutoTokenizer.from_pretrained("quality-model")
model = AutoModelForSequenceClassification.from_pretrained("quality-model")

def score_text_quality(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze()
    return scores.item()  # 返回质量得分
```

**主要质量指标**：
- 🔤 **困惑度**：衡量文本自然度
- 📊 **词汇多样性**：独特词汇占比
- 📚 **信息密度**：有价值信息的密度
- 🔍 **语法正确性**：语法错误的频率
- 🧠 **逻辑连贯性**：句子间关系是否合理

### 3. 🧬 数据增强方法

**文本变换增强**：
```python
# 回译增强
from transformers import MarianMTModel, MarianTokenizer

def backtranslation(text, src_lang="en", mid_lang="fr"):
    # 首先翻译到中间语言
    mid_model = f"Helsinki-NLP/opus-mt-{src_lang}-{mid_lang}"
    mid_tokenizer = MarianTokenizer.from_pretrained(mid_model)
    mid_model = MarianMTModel.from_pretrained(mid_model)
    
    inputs = mid_tokenizer(text, return_tensors="pt", padding=True)
    translated = mid_model.generate(**inputs)
    mid_text = mid_tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # 再翻译回原语言
    back_model = f"Helsinki-NLP/opus-mt-{mid_lang}-{src_lang}"
    back_tokenizer = MarianTokenizer.from_pretrained(back_model)
    back_model = MarianMTModel.from_pretrained(back_model)
    
    inputs = back_tokenizer(mid_text, return_tensors="pt", padding=True)
    translated = back_model.generate(**inputs)
    back_text = back_tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return back_text
```

**GPT增强**：使用现有的强大模型来扩展和变换数据。

```python
import openai

def gpt_augment(instruction, n=3):
    """使用GPT生成多个变体"""
    augmented = []
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"请为以下指令创建{n}个不同的变体，保持意思相同但表达方式不同:"},
            {"role": "user", "content": instruction}
        ],
        n=n,
        temperature=0.7
    )
    
    for choice in response.choices:
        augmented.append(choice.message.content)
        
    return augmented
```

**常用增强策略**：
- 🔄 **回译增强**：通过中间语言翻译创建变体
- 🔀 **同义词替换**：替换关键词为同义词
- 🧩 **句法重组**：调整句子结构但保持语义
- 📝 **提示变换**：同一问题多种问法
- 🤖 **模型自生成**：大模型生成原始数据变体

### 4. 🏷️ 数据标注技术

**人工标注平台**：
- 内部标注团队
- 众包平台（如Amazon Mechanical Turk）
- 专业标注服务提供商

**半自动标注**：
```python
# 预测+人工修正流程
def semi_auto_labeling(texts, model, confidence_threshold=0.9):
    labeled_data = []
    for text in texts:
        # 模型预测
        prediction, confidence = model.predict(text)
        
        if confidence > confidence_threshold:
            # 高置信度样本直接采用
            labeled_data.append((text, prediction))
        else:
            # 低置信度样本标记为需人工审核
            labeled_data.append((text, "需人工审核", prediction))
    
    return labeled_data
```

**标注质量控制**：
- 🔄 多人标注同一样本取一致性
- 📏 设置黄金标准样本验证标注者
- 📊 定期评估标注者内部一致性
- 🧠 使用强模型辅助检查标注错误

## 📦 数据格式与组织

### 1. 📄 常用数据格式

**JSONL格式**（每行一个JSON对象）：
```jsonl
{"text": "这是第一个样本", "label": "正面"}
{"text": "这是第二个样本", "label": "负面"}
```

**指令格式**：
```jsonl
{"instruction": "用简单的语言解释相对论", "input": "", "output": "相对论是爱因斯坦提出的物理学理论..."}
{"instruction": "总结以下段落", "input": "机器学习是人工智能的一个分支...", "output": "机器学习是AI的子领域，专注于让计算机从数据中学习。"}
```

**对话格式**：
```jsonl
{"messages": [{"role": "system", "content": "你是一个有用的助手"}, {"role": "user", "content": "什么是机器学习？"}, {"role": "assistant", "content": "机器学习是..."}]}
```

### 2. 📂 数据分割策略

**传统分割**：
```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
```

**时间序列分割**：确保测试数据时间上晚于训练数据。

**内容感知分割**：确保不同集合主题分布相似。

**推荐分割比例**：
- 预训练：98% 训练, 1% 验证, 1% 测试
- 监督微调：80% 训练, 10% 验证, 10% 测试
- RLHF阶段：70% 训练, 15% 验证, 15% 测试

### 3. 🧰 数据存储与管理

**高效文件格式**：
- 小型数据集：JSONL或CSV格式
- 中型数据集：SQLite数据库
- 大型数据集：Apache Parquet或Arrow格式
- 超大数据集：分布式存储（HDFS、S3）

**数据版本控制**：
- 使用DVC (Data Version Control)
- 与Git集成管理数据版本变更

```bash
# 使用DVC管理数据
dvc init
dvc add data/
git add data.dvc .gitignore
git commit -m "Add training data"
```

## 🔍 专业领域数据准备

### 1. 🏥 医疗领域

**数据来源**：
- 医学教科书、指南和论文
- 医疗问答平台（经过脱敏）
- 医学百科全书和词典
- 脱敏病例报告

**特殊处理**：
- 敏感信息脱敏
- 术语规范化
- 专业准确性验证（医学专家审核）
- 增加疾病、症状和治疗的多样化表达

### 2. 💼 法律领域

**数据来源**：
- 法律条文和法规
- 判例和案例分析
- 法律解释和评论
- 法律问答和咨询记录

**特殊处理**：
- 法律术语标准化
- 司法管辖区的明确标注
- 时效性信息的明确标记
- 多种法律情境的覆盖

### 3. 💻 代码与编程

**数据来源**：
- GitHub公共代码库
- Stack Overflow问答
- 编程教程和文档
- 代码注释和技术博客

**特殊处理**：
- 代码片段与解释配对
- 多编程语言平衡
- 代码质量筛选
- 执行验证（确保代码可运行）

## 🚀 数据准备实战案例

### 1. 📈 通用指令微调数据准备

**目标**：创建高质量的通用指令数据集

**步骤**：
1. **收集初始种子数据**：从现有的开源指令数据集选取高质量样本
2. **质量过滤**：应用质量指标筛选初始数据
3. **分类扩展**：确保数据涵盖多种指令类型（创意写作、信息查询、逻辑推理等）
4. **自举扩展**：使用强大模型基于种子生成更多样本
5. **人工审核**：对生成的样本进行人工审核和修正
6. **多轮测试**：使用小批量数据进行微调测试，迭代改进数据集

**示例流程**：
```python
# 1. 加载和合并多个种子数据集
seed_data = load_datasets(['alpaca', 'dolly', 'oasst'])

# 2. 质量过滤
filtered_data = filter_by_quality(seed_data, min_score=0.7)

# 3. 类别平衡与抽样
balanced_data = balance_by_category(filtered_data)

# 4. 使用强大模型扩展数据
expanded_data = []
for seed in balanced_data:
    variations = generate_variations(seed, model="gpt-4", n=3)
    expanded_data.extend(variations)

# 5. 人工审核（通过UI或平台实现）
reviewed_data = human_review(expanded_data)

# 6. 导出为训练格式
export_to_jsonl(reviewed_data, "high_quality_instructions.jsonl")
```

### 2. 🏥 医疗领域微调数据准备

**目标**：创建医疗问答微调数据集

**步骤**：
1. **收集医学知识**：从医学教科书、指南和可靠来源提取信息
2. **构建问答对**：将医学知识转化为问答格式
3. **多样化表述**：为同一问题创建多种问法
4. **专业审核**：邀请医学专业人员审核内容准确性
5. **分级分类**：按专业水平和医学领域分类数据

**示例医疗数据条目**：
```json
{
  "instruction": "解释高血压的诊断标准和常见治疗方法",
  "input": "",
  "output": "高血压的诊断标准通常为血压持续高于130/80 mmHg。诊断需要多次测量确认。\n\n治疗方法包括：\n1. 生活方式干预：减少钠盐摄入、控制体重、规律运动、限制酒精\n2. 药物治疗：可能包括利尿剂、ACE抑制剂、钙通道阻滞剂、β受体阻滞剂等\n3. 定期监测：定期测量血压并调整治疗方案\n\n具体治疗方案应在医生指导下制定，因个体差异可能需要个性化调整。"
}
```

## 💡 数据准备最佳实践

### 1. 🔄 数据迭代策略

**渐进式数据改进**：
1. 从小规模高质量种子数据开始
2. 训练初始模型并评估
3. 分析错误和不足表现的领域
4. 有针对性地收集解决具体弱点的数据
5. 整合新数据并重新训练
6. 重复上述过程直至达到目标性能

**数据质量与数量平衡**：
- 微调初期：优先考虑数据质量，即使数量有限
- 能力扩展阶段：在保证基本质量的前提下扩大数据量和多样性
- 精细优化阶段：专注于高质量、难度适中的关键数据

### 2. 📏 数据质量体系

**建立全面的数据质量评估体系**：
- 自动化指标：困惑度、语法正确性、多样性等
- 人工评估：专家审核、内容准确性、有用性等
- 模型验证：使用强模型评估内容质量、相关性等

**质量审核流程**：
1. 自动化筛选明显低质量内容
2. 随机抽样进行人工质量审核
3. 对难以判断的边界情况进行特别审核
4. 持续更新质量标准和筛选规则

### 3. 🛡️ 安全与伦理

**偏见与有害内容控制**：
- 使用安全过滤器识别和移除有害内容
- 平衡不同视角和观点
- 避免隐含偏见的表达

**数据隐私保护**：
- 确保所有数据合规收集和使用
- 移除个人身份信息
- 遵循数据使用的法律和伦理准则

**透明度和可追溯性**：
- 记录数据来源和处理过程
- 明确标记合成或增强数据
- 建立数据谱系记录系统

## 📚 扩展资源与工具

### 1. 🛠️ 常用数据处理工具

- [🔧 HuggingFace Datasets](https://github.com/huggingface/datasets) - 大型数据集库和工具
- [🔧 NLTK](https://www.nltk.org/) - 自然语言处理工具包
- [🔧 SpaCy](https://spacy.io/) - 高效NLP库
- [🔧 OpenAI Dataset Preparation](https://platform.openai.com/docs/guides/fine-tuning) - 指令微调数据准备指南
- [🔧 LangChain](https://github.com/langchain-ai/langchain) - 大语言模型应用框架
- [🔧 DVC](https://dvc.org/) - 数据版本控制系统

### 2. 📊 开源数据集

**预训练数据集**：
- [📁 The Pile](https://pile.eleuther.ai/) - 800GB多样文本语料库
- [📁 RedPajama](https://github.com/togethercomputer/RedPajama-Data) - 开源预训练数据集
- [📁 SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) - 高质量精简预训练数据

**指令微调数据集**：
- [📁 Alpaca](https://github.com/tatsu-lab/stanford_alpaca) - Stanford的指令微调数据
- [📁 Open Assistant](https://huggingface.co/datasets/OpenAssistant/oasst1) - 多语言会话数据集
- [📁 ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) - ChatGPT对话数据集
- [📁 LIMA](https://huggingface.co/datasets/GAIR/lima) - 高质量1K指令数据集

### 3. 📄 延伸阅读

- [📝 数据质量与模型性能关系研究](https://arxiv.org/abs/2212.04356)
- [📝 LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [📝 Self-Instruct: Aligning LMs with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- [📝 超参数和数据集大小的权衡研究](https://arxiv.org/abs/2203.15556) 