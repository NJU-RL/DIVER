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

def calculate_similarity_matrix(group_rollouts, select_n, filter_high_div, reward_flag):
    # reward_flag = reward_flag.numpy()
    # print(f'reward flag: {reward_flag}')
    n = len(group_rollouts)
    formulas = []
    for i in range(len(group_rollouts)):
        formulas.append(extract_formulas(group_rollouts[i]))
    
    diversity = []

    for i in range(len(formulas)):
        diversity.append(calculate_unique_diversity(formulas, i))
    
    # divide into 2 groups
    group1_indices = [i for i in range(n) if reward_flag[i] == 1]
    group0_indices = [i for i in range(n) if reward_flag[i] == 0]

    select_n1 = (len(group1_indices) + 1) // 2
    select_n0 = len(group0_indices) // 2

    if filter_high_div:
        group1_selected = heapq.nlargest(select_n1, 
                                   group1_indices, 
                                   key=lambda i: diversity[i])
        group0_selected = heapq.nlargest(select_n0, 
                                    group0_indices, 
                                    key=lambda i: diversity[i])
    
    else:
        group1_selected = heapq.nsmallest(select_n1, 
                                   group1_indices, 
                                   key=lambda i: diversity[i])
        group0_selected = heapq.nsmallest(select_n0, 
                                    group0_indices, 
                                    key=lambda i: diversity[i])
        
    selected_indices = group1_selected + group0_selected

    # print('group1:')
    # for item in group1_selected:
    #     print(reward_flag[item])
    
    # print('group0:')
    # for item in group0_selected:
    #     print(reward_flag[item])

    
    select_seqs = [group_rollouts[i] for i in selected_indices]
    # for i in range(len(select_seqs)):
        # print(f'diversity of response {i}: {diversity[indices[i]]}')

    return np.array(selected_indices), select_seqs

