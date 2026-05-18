import os
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
#import metrics.sbert

@contextmanager
def sentence_transformer_context(model_name):
    """上下文管理器确保资源正确释放"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("需要安装sentence-transformers: pip install sentence-transformers")
    model = None
    try:
        print(f"model start loading {os.environ['TFHUB_CACHE_DIR']}")
        model = SentenceTransformer(model_name, cache_folder=os.environ['TFHUB_CACHE_DIR'])
        print("stsb-roberta-large model loaded")
        yield model
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()
            import gc
            gc.collect()

def calculate_sbert_similarity(predictions, references, model_name='stsb-roberta-large'):
    """
    计算Sentence-BERT语义相似度指标

    Args:
        predictions: 预测文本列表，每个元素是字符串
        references: 参考文本列表，每个元素是字符串
        model_name: Sentence-BERT模型名称，默认使用在STS-B上训练的RoBERTa-large

    Returns:
        dict: 包含余弦相似度和欧氏距离的字典
    """
    # 使用
    with sentence_transformer_context(model_name) as model:
        model.eval()
        # 编码文本为向量
        ref_embeddings = model.encode(references)
        pred_embeddings = model.encode(predictions)

    # 计算余弦相似度
    cosine_scores = []
    for ref_emb, pred_emb in zip(ref_embeddings, pred_embeddings):
        score = cosine_similarity([ref_emb], [pred_emb])[0][0]
        cosine_scores.append(score)

    # 计算欧氏距离
    euclidean_scores = []
    for ref_emb, pred_emb in zip(ref_embeddings, pred_embeddings):
        distance = euclidean_distances([ref_emb], [pred_emb])[0][0]
        euclidean_scores.append(distance)

    # print(cosine_scores,"\n", euclidean_scores)
    # 返回平均指标
    return {
        'sbert_cosine': np.mean(cosine_scores),
        'sbert_euclidean': np.mean(euclidean_scores)
    }
