import numpy as np
import heapq
from nltk.translate.bleu_score import sentence_bleu
import re
from difflib import SequenceMatcher
from collections import Counter

think_phrases = [
    "alternatively",
    "as a result",
    "assume",
    "because",
    "calculate",
    "check",
    "compute",
    "confirm",
    "consequently",
    "consider",
    "define",
    "determine",
    "does not work",
    "doesn't work",
    "error",
    "evaluate",
    "finally",
    "find",
    "first",
    "firstly",
    "for example",
    "for instance",
    "get",
    "given there",
    "hence",
    "however",
    "if",
    "makes sence",
    "next",
    "not correct",
    "not working",
    "now",
    "re-calculate",
    "re-evaluate",
    "recalculate",
    "reevaluate",
    "since",
    "so",
    "solved",
    "step",
    "still not",
    "summarize",
    "then",
    "thereby",
    "therefore",
    "thus",
    "try",
    "verify",
    "wait"
]

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

def calculate_equation_matrix(group_rollouts, div_len=200):
    formulas = []
    for i in range(len(group_rollouts)):
        formulas.append(extract_formulas(group_rollouts[i][:]))
    
    diversity = []

    for i in range(len(formulas)):
        diversity.append(calculate_unique_diversity(formulas, i))

    return diversity

def calculate_belu_matrix(group_rollouts):
    n = len(group_rollouts)
    weights = (0.02, 0.1, 0.15, 0.25, 0.38)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # calculate BLEU score
            reference_i = [group_rollouts[i].split()]
            candidate_j = group_rollouts[j].split()
            bleu_i_j = sentence_bleu(reference_i, candidate_j, weights=weights)
            
            reference_j = [group_rollouts[j].split()]
            candidate_i = group_rollouts[i].split()
            bleu_j_i = sentence_bleu(reference_j, candidate_i, weights=weights)
            
            # Similarity is bidirectional
            similarity = (bleu_i_j + bleu_j_i) / 2
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    avg_similarities = np.sum(similarity_matrix, axis=1) / (n-1)

    return avg_similarities

def calculate_phrase_repetition_penalty(phrases, n=1, penalty_value=0.01):
    """
    计算短语序列的重复惩罚值
    
    参数:
    phrases - 短语序列列表
    n - n-gram大小，默认为1（单个短语）
    penalty_value - 惩罚值，默认为0.1
    
    返回:
    total_penalty - 总惩罚值
    """
    if not phrases:
        return 0.0
    
    # 序列长度
    l = len(phrases)
    
    # 如果序列长度小于n-gram大小，则返回零惩罚
    if l < n:
        return 0.0
    
    # 初始化惩罚向量和已观察到的n-grams集合
    penalties = [0.0] * l
    observed_ngrams = set()
    
    # 对每个可能的n-gram位置进行迭代
    for j in range(l - n + 1):
        # 提取当前的n-gram
        current_ngram = tuple(phrases[j:j+n])
        
        # 如果当前n-gram已经在观察集合中，应用惩罚
        if current_ngram in observed_ngrams:
            for t in range(j, j + n):
                penalties[t] += penalty_value
        
        # 将当前n-gram添加到观察集合
        observed_ngrams.add(current_ngram)
    
    # 计算总惩罚值
    total_penalty = sum(penalties)
    
    return total_penalty

def extract_phrases_regex(text, phrases_to_extract):
    occurrences = []
    for phrase in phrases_to_extract:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        for match in pattern.finditer(text):
            occurrences.append((match.start(), text[match.start():match.end()]))
    occurrences.sort(key=lambda x: x[0])
    extracted_phrases = [item[1] for item in occurrences]
    return extracted_phrases 

def calculate_struct_similarity(group_rollouts, phrases_to_extract):
    n = len(group_rollouts)
    similarity_matrix = np.zeros((n, n))
    repetition_penalty = []
    for i in range(n):
        phrases_i = extract_phrases_regex(group_rollouts[i], phrases_to_extract)
        repetition_penalty.append(0.05 * calculate_phrase_repetition_penalty(phrases=phrases_i))
        for j in range(i+1, n):
            # phrases_i = extract_phrases_regex(group_rollouts[i], phrases_to_extract)
            phrases_j = extract_phrases_regex(group_rollouts[j], phrases_to_extract)
            struct_sim_i_j = SequenceMatcher(None, phrases_i, phrases_j).ratio()
            struct_sim_j_i = SequenceMatcher(None, phrases_j, phrases_i).ratio()

            struct_sim = (struct_sim_i_j + struct_sim_j_i) / 2
            similarity_matrix[i][j] = struct_sim
            similarity_matrix[j][i] = struct_sim

    avg_similarities = np.sum(similarity_matrix, axis=1) / (n-1)
    # print(f'avg sim: {avg_similarities}')
    # print(f'repetition: {repetition_penalty}')
    avg_similarities += repetition_penalty
    # print(avg_similarities)
    return avg_similarities

