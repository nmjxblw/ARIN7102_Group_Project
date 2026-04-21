# 端到端评估指南 (macOS)

本文档说明如何在 macOS 上运行药物推荐系统的端到端评估。

## 前置条件

### 1. Python 环境
- Python 3.10+（代码使用了 `str | None` 语法）
- 推荐使用 conda 或 venv

```bash
# conda
conda create -n arin7102 python=3.11 -y
conda activate arin7102

# 或 venv
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 依赖安装

```bash
cd ARIN7102_Group_Project

# 方式一：使用 Mac 适配后的 requirements
pip install -r requirements_mac.txt

# 方式二：只装核心依赖（更快）
pip install torch torchvision torchaudio
pip install transformers sentence-transformers faiss-cpu
pip install fastapi uvicorn openai pydantic pandas numpy tqdm python-dotenv
```

> macOS 没有 NVIDIA GPU，torch 会自动使用 CPU。Apple Silicon Mac 上 cross-encoder 会自动使用 MPS 加速。

### 3. 数据文件准备

确保以下文件存在：
- `match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv` - 增强药物表
- `drug_comprehensive_embeddings.npy` - 药物向量（在 repo 根目录）
- `data/eval_dataset_llm.json` - 评估数据集（如不存在需先生成，见步骤一）

---

## 步骤一：生成评估数据集

如果 `data/eval_dataset_llm.json` 不存在，需要先使用 LLM 生成：

```bash
cd ARIN7102_Group_Project/app

python -m evaluation.generate_eval_dataset_llm \
    --input-csv "../match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv" \
    --output-json "../data/eval_dataset_llm.json" \
    --num-samples 100 \
    --min-diseases 1 \
    --min-symptoms 2 \
    --model "deepseek-v3-2-251201" \
    --base-url "https://ark.cn-beijing.volces.com/api/coding/v3"
```

**重要参数说明**:
| 参数 | 说明 |
|------|------|
| `--input-csv` | 增强药物表路径 |
| `--output-json` | 输出评估数据集路径 |
| `--num-samples` | 生成样本数量 |
| `--model` | **必须指定**，LLM 模型名称 |
| `--base-url` | **必须指定**，API 端点 |
| `--api-key` | 可选，也可通过环境变量 `OPENAI_API_KEY` 设置 |

---

## 步骤二：运行端到端评估

```bash
cd ARIN7102_Group_Project/app

python -m evaluation.run_evaluation \
    --eval-dataset "../data/eval_dataset_llm.json" \
    --k-values 5 10 20 \
    --output "../data/eval_results.json"
```

**参数说明**:
| 参数 | 说明 |
|------|------|
| `--eval-dataset` | 评估数据集路径 |
| `--k-values` | 计算 Precision@K, Recall@K 的 K 值列表 |
| `--output` | 评估结果输出路径（可选） |

---

## 评估指标

评估脚本会计算以下指标：

| 指标 | 说明 |
|------|------|
| `precision@k` | 前 K 个推荐中相关药物的比例 |
| `recall@k` | 前 K 个推荐覆盖了多少相关药物 |
| `hit@k` | 前 K 个推荐中是否包含至少一个相关药物 |
| `mrr` | Mean Reciprocal Rank，第一个相关药物排名的倒数均值 |
| `ndcg@k` | Normalized Discounted Cumulative Gain |

---

## 环境变量配置

在 shell 中设置（或写入 `app/.env`）：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://ark.cn-beijing.volces.com/api/coding/v3"
export OPENAI_MODEL="deepseek-v3-2-251201"
```

也可以通过命令行参数 `--api-key`、`--base-url`、`--model` 覆盖。

---

## 潜在问题及解决方案

### 问题 1：LLM API 调用失败 (UnsupportedModel)

**错误信息**:
```
Error code: 404 - {'error': {'code': 'UnsupportedModel', 'message': 'The gpt-3.5-turbo model does not support...'}}
```

**原因**: 脚本默认使用 `gpt-3.5-turbo`，但 API 端点不支持该模型

**解决方案**: 必须显式指定 `--model` 和 `--base-url` 参数：
```bash
--model "deepseek-v3-2-251201" --base-url "https://ark.cn-beijing.volces.com/api/coding/v3"
```

---

### 问题 2：模型下载缓慢或失败

**原因**: 首次运行时需要从 Hugging Face 下载模型（PubMedBERT, Cross-Encoder 等）

**解决方案**:
1. 确保网络通畅（可能需要代理）
2. 设置 Hugging Face 镜像（可选）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. 预先下载模型到本地

---

### 问题 3：Apple Silicon (M1/M2/M3) 上 torch 相关警告

Apple Silicon Mac 上 cross-encoder 会自动使用 MPS 后端。如果遇到 MPS 不兼容的 op，可以强制回退 CPU：

```bash
export PYTORCH_MPS_FALLBACK=1
```

---

## 完整运行示例

```bash
# 1. 激活环境
conda activate arin7102

# 2. 进入项目目录
cd ARIN7102_Group_Project

# 3. 设置 API key
export OPENAI_API_KEY="your_api_key"

# 4. 生成评估数据集（如需要）
cd app
python -m evaluation.generate_eval_dataset_llm \
    --input-csv "../match_data_preprocessing/data/enhanced_drug_table_v1_structured.csv" \
    --output-json "../data/eval_dataset_llm.json" \
    --num-samples 100 \
    --model "deepseek-v3-2-251201" \
    --base-url "https://ark.cn-beijing.volces.com/api/coding/v3"

# 5. 运行评估
python -m evaluation.run_evaluation \
    --eval-dataset "../data/eval_dataset_llm.json" \
    --k-values 5 10 20 \
    --output "../data/eval_results.json"
```

---

## 输出结果

评估完成后，结果保存在 `data/eval_results.json`：

```json
{
  "metrics": {
    "precision@5": 0.xxxx,
    "precision@10": 0.xxxx,
    "precision@20": 0.xxxx,
    "recall@5": 0.xxxx,
    "recall@10": 0.xxxx,
    "recall@20": 0.xxxx,
    "hit@5": 0.xxxx,
    "hit@10": 0.xxxx,
    "hit@20": 0.xxxx,
    "mrr": 0.xxxx,
    "ndcg@5": 0.xxxx,
    "ndcg@10": 0.xxxx,
    "ndcg@20": 0.xxxx
  },
  "num_queries": 100,
  "k_values": [5, 10, 20]
}
```
