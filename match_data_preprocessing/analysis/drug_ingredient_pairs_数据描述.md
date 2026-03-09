# drug_ingredient_pairs 数据描述文档

## 一、数据来源

基于 `enhanced_drug_table.json`（5,595 条药物记录）中的 `drug_name`（商品名）和 `generic_name`（通用名/有效成分）字段，经数据清洗后构建。

---

## 二、原始数据统计

| 指标 | 数值 | 占比 |
|------|------|------|
| 原始数据总记录数 | 5,595 | 100% |
| `generic_name` 非空 | 2,869 | 51.3% |
| `generic_name` 为空 | 2,726 | 48.7% |

### generic_name 非空记录中的成分类型

| 类型 | 数值 | 占非空记录比例 |
|------|------|----------------|
| 单成分药物 | 1,926 | 67.1% |
| 复方药物（多成分） | 943 | 32.9% |

### 单成分统计

| 指标 | 数值 |
|------|------|
| 清洗后唯一单成分种类 | 805 |
| 其中 ≥2 个品牌名的成分 | 503 |
| 其中仅 1 个品牌名的成分 | 302 |

### 品牌数分布（≥2 品牌的成分）

| 品牌数范围 | 成分数量 |
|------------|----------|
| 2-3 个品牌 | 382 种 |
| 4-5 个品牌 | 67 种 |
| 6-10 个品牌 | 42 种 |
| 11-20 个品牌 | 12 种 |
| 20+ 个品牌 | 0 种 |

### Top 20 有效成分（按品牌数排序）

| 有效成分 | 品牌数 |
|----------|--------|
| acetaminophen | 18 |
| diphenhydramine | 18 |
| benzoyl peroxide topical | 16 |
| aspirin | 16 |
| ibuprofen | 16 |
| guaifenesin | 15 |
| benzoyl peroxide | 14 |
| chlorpheniramine | 12 |
| menthol topical | 12 |
| salicylic acid | 11 |
| coal tar topical | 11 |
| diltiazem | 11 |
| oxycodone | 10 |
| lidocaine topical | 10 |
| methylphenidate | 10 |
| erythromycin | 10 |
| hydrocortisone topical | 10 |
| methyl salicylate topical | 10 |
| triamcinolone | 9 |
| bismuth subsalicylate | 9 |

---

## 三、数据清洗规则

构建脚本：`scripts/analyze_and_build_dataset.py`

### 规则 1：排除 generic_name 为空的记录
- 排除 2,726 条 `generic_name` 为空字符串的记录

### 规则 2：仅保留单成分药物
- **判定复方药物的规则**：`generic_name` 中包含以下任一标志则视为复方：
  - 包含 ` and `（不区分大小写）
  - 包含 `,`（逗号）
  - 包含 ` / `（空格-斜杠-空格）
- 排除 943 条复方记录

### 规则 3：清洗 generic_name 格式
对保留的单成分 `generic_name` 执行以下清洗：

| 步骤 | 操作 | 示例 |
|------|------|------|
| 3.1 | 去除首尾空格 | `" ibuprofen "` → `"ibuprofen"` |
| 3.2 | 去除括号中的给药途径注释 | `"ibuprofen (oral route)"` → `"ibuprofen"` |
| 3.3 | 统一转为小写 | `"Acetaminophen"` → `"acetaminophen"` |
| 3.4 | 合并多余空格 | `"benzoyl  peroxide"` → `"benzoyl peroxide"` |

**被清洗的括号注释分布（Top 15）：**

| 注释内容 | 出现次数 |
|----------|----------|
| (oral route) | 136 |
| (oral) | 122 |
| (oral/injection) | 51 |
| (injection) | 47 |
| (topical route) | 43 |
| (topical application route) | 26 |
| (intravenous route) | 12 |
| (nasal) | 12 |
| (rectal) | 10 |
| (inhalation) | 10 |
| (subcutaneous route) | 9 |
| (injection route) | 9 |
| (topical) | 9 |
| (transdermal) | 7 |
| (sublingual) | 3 |

### 规则 4：仅保留 ≥2 个品牌名的成分
- 排除 302 种仅有 1 个品牌名的成分（无法用于聚类验证）

### 规则 5：drug_name 清洗
- 去除首尾空格
- 统一转为小写

### 规则 6：去重
- 以 `(drug_name, active_ingredient)` 为键去重

---

## 四、最终数据集概况

| 指标 | 数值 |
|------|------|
| **总记录数** | **1,624** |
| **唯一有效成分** | **503 种** |
| **唯一商品名** | **1,624 个** |
| 平均每种成分对应品牌数 | 3.2 个 |

### 输出文件

| 文件 | 路径 | 格式 |
|------|------|------|
| CSV | `data/drug_ingredient_pairs.csv` | 含表头，`original_conditions` 以 `\|` 分隔 |
| JSON | `data/drug_ingredient_pairs.json` | 数组格式，`original_conditions` 为列表 |

### 字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `drug_name` | string | 药品商品名（小写） |
| `active_ingredient` | string | 有效成分（清洗后，小写） |
| `drug_classes` | string | 药物分类（原始值，可为空） |
| `original_conditions` | list/string | 适应症列表 |

---

## 五、数据用途

本数据集用于：
1. 对 `drug_name` 做 embedding 后聚类，用 `active_ingredient` 验证聚类准确性
2. 分析 embedding 向量中哪些维度与有效成分最相关

详见方案文档：`embedding_方案说明.md`
