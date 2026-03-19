我这部分主要是在做药物推荐 pipeline 的实验和分析。
一开始我先用 PubMedBERT 做 pure KNN 语义检索，但我发现对于像 common cold 这种很短的 query，在全药库上直接做相似度搜索会失效，因为分数都挤在很小的区间里，区分不出来。
所以我后面改成了一个 hybrid pipeline：先用疾病和症状做 baseline 粗筛，再在筛出来的候选里用 KNN 做精排。然后我把 pure KNN 和 hybrid 的 score distribution 画出来做对比分析。再往后我还尝试了用 Random Forest 做 re-ranking，把语义分数、症状命中数、疾病命中数和 rating 这些特征结合起来继续排序。