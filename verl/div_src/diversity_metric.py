import numpy as np
import heapq
from nltk.translate.bleu_score import sentence_bleu
import re


def extract_formulas(response):
    # Define regular patterns for formulas
    patterns = [
        r'\\\[([^\]]*?)\\\]',     # \[ \]
        r'\\\(([^\)]*?)\\\)',     # \( \)
        r'\$([^\$]*?)\$'          # $ $
    ]
    
    formulas = set()
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        formulas.update(matches)
    return list(formulas) 


def calculate_unique_diversity(formulas, current_index):
    if not formulas[current_index]:  # 如果当前response为空
        return 0
    
    # 获取所有其他response中的公式
    other_formulas = set()
    for i in range(len(formulas)):
        if i != current_index:
            other_formulas.update(formulas[i])
    
    # 获取当前response中的公式
    current_formulas = set(formulas[current_index])
    
    # 计算只在当前response中出现的公式（独特公式）
    unique_formulas = current_formulas - other_formulas
    
    # 计算多样性指标 D_eq
    D_eq = len(unique_formulas) / len(formulas[current_index])

    # print(f'{len(unique_formulas)} / {len(formulas[current_index])}')
    
    return D_eq

def calculate_equation_matrix(group_rollouts):
    formulas = []
    for i in range(len(group_rollouts)):
        formulas.append(extract_formulas(group_rollouts[i]))
    
    diversity = []

    for i in range(len(formulas)):
        diversity.append(calculate_unique_diversity(formulas, i))

    return sum(diversity)/len(diversity)

def calculate_belu_matrix(group_rollouts):
    n = len(group_rollouts)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # calculate BLEU score
            reference_i = [group_rollouts[i].split()]
            candidate_j = group_rollouts[j].split()
            bleu_i_j = sentence_bleu(reference_i, candidate_j)
            
            reference_j = [group_rollouts[j].split()]
            candidate_i = group_rollouts[i].split()
            bleu_j_i = sentence_bleu(reference_j, candidate_i)
            
            # Similarity is bidirectional
            similarity = (bleu_i_j + bleu_j_i) / 2
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    avg_similarities = np.sum(similarity_matrix, axis=1) / (n-1)

    return avg_similarities.mean()