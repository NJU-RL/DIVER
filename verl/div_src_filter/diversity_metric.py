import numpy as np
import heapq
from nltk.translate.bleu_score import sentence_bleu


################ BELU metric#################

def calculate_similarity_matrix(group_rollouts, select_n, div_type='high'):
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
    if select_n >= n:
        return list(range(n))
    if div_type == 'high':
        # Use the min-heap to find the smallest n elements and their indexes
        indices = heapq.nsmallest(select_n, range(len(avg_similarities)), key=lambda i: avg_similarities[i])
    else:
        indices = heapq.nlargest(select_n, range(len(avg_similarities)), key=lambda i: avg_similarities[i])

    select_seqs = [group_rollouts[i] for i in indices]

    return np.array(indices), select_seqs

import torch
import torch.nn.functional as F

def select_diverse_embeddings(embeddings, select_n, div_type='high'):
    """    
    args:
        embeddings: (n_rollout, embedding_dim)
        select_n: The number of samples to be selected
        div_type: 'high' is the selection of the most diverse, 'low' is the selection of the most similar
        
    returns:
        tensor, The index containing the selected samples
    """
    n_rollout = embeddings.shape[0]
    
    if n_rollout <= 1 or select_n >= n_rollout:
        return torch.arange(n_rollout, device=embeddings.device)
    

    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
    

    mask = torch.ones_like(similarity_matrix) - torch.eye(n_rollout, device=similarity_matrix.device)
    masked_sim = similarity_matrix * mask
    
    # Calculate the average similarity
    avg_similarities = masked_sim.sum(dim=1) / (n_rollout - 1)
    
    if div_type == 'high':
        _, indices = torch.topk(avg_similarities, k=select_n, largest=False)
    else:
        _, indices = torch.topk(avg_similarities, k=select_n, largest=True)
    
    return indices


def select_diverse_embeddings_batch(embeddings, select_n, div_type='high'):
    """    
    args:
        embeddings: (bsz, n_rollout, embedding_dim)
        select_n: The number of samples to be selected
        div_type: 'high' is the selection of the most diverse, 'low' is the selection of the most similar
        
    returns:
        tensor, The indices containing the selected samples with shape (bsz, select_n)
    """
    bsz, n_rollout, embedding_dim = embeddings.shape
    
    if n_rollout <= 1 or select_n >= n_rollout:
        return torch.arange(n_rollout, device=embeddings.device).unsqueeze(0).expand(bsz, n_rollout)
    
    # L2
    normalized_embeddings = F.normalize(embeddings, p=2, dim=2)  # (bsz, n_rollout, embedding_dim)
    
    # bmm: batch matrix multiplication
    similarity_matrix = torch.bmm(normalized_embeddings, normalized_embeddings.transpose(1, 2))  # (bsz, n_rollout, n_rollout)
    
    similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
    
    # exclude diagonal elements
    eye_mask = torch.eye(n_rollout, device=embeddings.device).unsqueeze(0).expand(bsz, -1, -1)
    mask = torch.ones_like(similarity_matrix) - eye_mask
    
    masked_sim = similarity_matrix * mask
    
    # Calculate the average similarity
    avg_similarities = masked_sim.sum(dim=2) / (n_rollout - 1)  # (bsz, n_rollout)
    
    if div_type == 'high':
        _, indices = torch.topk(avg_similarities, k=select_n, largest=False, dim=1)
    else:
        _, indices = torch.topk(avg_similarities, k=select_n, largest=True, dim=1)
    
    return indices  # (bsz, select_n)