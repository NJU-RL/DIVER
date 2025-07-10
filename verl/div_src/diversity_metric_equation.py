import re
import heapq
import numpy as np

def extract_formulas(response):
    # 定义不同格式公式的正则模式
    patterns = [
        r'\\\[([^\]]*?)\\\]',     # \[ \]
        r'\\\(([^\)]*?)\\\)',     # \( \)
        r'\$([^\$]*?)\$'          # $ $
    ]
    
    # 存储所有找到的公式
    formulas = set()
    
    # 对每个模式进行匹配
    for pattern in patterns:
        matches = re.findall(pattern, response)
        # 将找到的公式添加到集合中（自动去重）
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

def calculate_similarity_matrix(group_rollouts, select_n, filter_high_div):
    n = len(group_rollouts)
    formulas = []
    for i in range(len(group_rollouts)):
        formulas.append(extract_formulas(group_rollouts[i]))
    
    diversity = []

    for i in range(len(formulas)):
        diversity.append(calculate_unique_diversity(formulas, i))
    
    # print(f'diversity: {diversity}')
    
    if filter_high_div:
        indices = heapq.nlargest(select_n, range(len(diversity)), key=lambda i: diversity[i])
    else:
        indices = heapq.nsmallest(select_n, range(len(diversity)), key=lambda i: diversity[i])
    
    select_seqs = [group_rollouts[i] for i in indices]
    # for i in range(len(select_seqs)):
        # print(f'diversity of response {i}: {diversity[indices[i]]}')

    return np.array(indices), select_seqs

