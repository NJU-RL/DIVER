import numpy as np
import heapq
from nltk.translate.bleu_score import sentence_bleu

def calculate_similarity_matrix(group_rollouts, select_n):
    n = len(group_rollouts)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # calculate BLEU score
            reference_i = [group_rollouts[i][0].split()]
            candidate_j = group_rollouts[j][0].split()
            bleu_i_j = sentence_bleu(reference_i, candidate_j)
            
            reference_j = [group_rollouts[j][0].split()]
            candidate_i = group_rollouts[i][0].split()
            bleu_j_i = sentence_bleu(reference_j, candidate_i)
            
            # Similarity is bidirectional
            similarity = (bleu_i_j + bleu_j_i) / 2
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    avg_similarities = np.sum(similarity_matrix, axis=1) / (n-1)
    if select_n >= n:
        return list(range(n))
    # Use the min-heap to find the smallest n elements and their indexes
    indices = heapq.nsmallest(select_n, range(len(avg_similarities)), key=lambda i: avg_similarities[i])
    select_seqs = [group_rollouts[i][0] for i in indices]

    return indices, select_seqs