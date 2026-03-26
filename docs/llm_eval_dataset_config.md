# LLM 评估数据集生成配置说明

## API Key 和 URL 配置方式

### 方式1: 环境变量（推荐）

创建 `.env` 文件在项目根目录：

```bash
# OpenAI API 配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
```

或者在命令行中设置：

```bash
# Windows
set OPENAI_API_KEY=your_api_key_here
set OPENAI_BASE_URL=https://api.openai.com/v1

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=https://api.openai.com/v1
```

### 方式2: 命令行参数

```bash
python app/evaluation/generate_eval_dataset_llm.py \
  --input-csv match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv \
  --output-json data/eval_dataset_llm.json \
  --num-samples 100 \
  --api-key "your_api_key_here" \
  --base-url "https://api.openai.com/v1" \
  --model "gpt-3.5-turbo"
```

---

## 支持的 API 提供商

### OpenAI
```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
```

### Azure OpenAI
```bash
OPENAI_API_KEY=your_azure_key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
OPENAI_MODEL=gpt-35-turbo
```

### 其他兼容 OpenAI API 的服务
```bash
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://your-service-url/v1
OPENAI_MODEL=your-model-name
```

---

## 使用示例

### 基础使用（使用环境变量）
```bash
python app/evaluation/generate_eval_dataset_llm.py \
  --input-csv match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv \
  --output-json data/eval_dataset_llm.json \
  --num-samples 100
```

### 完整参数
```bash
python app/evaluation/generate_eval_dataset_llm.py \
  --input-csv match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv \
  --output-json data/eval_dataset_llm.json \
  --num-samples 200 \
  --min-diseases 1 \
  --min-symptoms 2 \
  --confidence-min 0.5 \
  --confidence-max 0.75 \
  --api-key "sk-..." \
  --base-url "https://api.openai.com/v1" \
  --model "gpt-3.5-turbo"
```

---

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input-csv` | 输入药物表CSV文件路径 | 必填 |
| `--output-json` | 输出评估数据集JSON文件路径 | 必填 |
| `--num-samples` | 生成样本数量 | 100 |
| `--min-diseases` | 药物最少疾病标签数 | 1 |
| `--min-symptoms` | 药物最少症状标签数 | 1 |
| `--confidence-min` | 标签置信度最小值 | 0.5 |
| `--confidence-max` | 标签置信度最大值 | 0.75 |
| `--api-key` | LLM API密钥 | 从环境变量读取 |
| `--base-url` | LLM API基础URL | 从环境变量读取 |
| `--model` | LLM模型名称 | gpt-3.5-turbo |

---

## 生成的数据集格式

```json
{
  "query_id": "eval_0001",
  "symptom_text": "I've been experiencing severe chest pain and shortness of breath, especially when I try to exercise or climb stairs.",
  "diseases": [
    {"name": "angina", "confidence": 0.68}
  ],
  "symptoms": [
    {"name": "chest_pain", "confidence": 0.72},
    {"name": "shortness_of_breath", "confidence": 0.65}
  ],
  "relevant_drugs": ["nitroglycerin"],
  "relevance_scores": {"nitroglycerin": 3}
}
```

**关键特点**：
- `symptom_text`: 由LLM生成的自然语言描述
- `diseases/symptoms`: 使用药物表中的真实标签
- `confidence`: 设置为较低值（0.5-0.75）用于测试链路鲁棒性

---

## 注意事项

1. **API 费用**: 生成100个样本约消耗 0.01-0.05 USD（使用 gpt-3.5-turbo）
2. **速率限制**: 脚本会自动重试，但请注意 API 速率限制
3. **质量检查**: 建议人工抽查 10-20% 的生成结果
4. **置信度设置**: 当前设置为 0.5-0.75，模拟 BERT 提取不够准确的场景
