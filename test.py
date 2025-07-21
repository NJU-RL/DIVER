import torch

test = torch.tensor([
    [1,2,3,4],
    [1,7,8,8],
    [1,2,4,5],
    [3,4,5,6],
    [3,4,1,4],
    [3,4,0,1]
])
print(test.reshape(2,3,4)[0])

test = test.reshape(2,3,4).repeat_interleave(3, dim=0).reshape(6,3,4)
print(test.shape)
print("*************")
print(test[3])
print(torch.Tensor(range(test.size(0)))%test.size(1))



import torch

def repeat_by_groups(x, n, mini_batch):
    """
    将形状为 (n*mini_batch, dim) 的向量按组重复，形成 (n*mini_batch*n, dim) 的向量
    每个组重复n次，保持原始组顺序
    
    参数:
    - x: 输入张量，形状为 (n*mini_batch, dim)
    - n: 每个组重复的次数
    - mini_batch: 每个批次的大小
    
    返回:
    - 形状为 (n*mini_batch*n, dim) 的张量，按组重复
    """
    total_size, dim = x.shape
    
    # 首先将输入重塑为 (n, mini_batch, dim)，以便分组
    x_grouped = x.reshape(n, mini_batch, dim)
    
    # 在第二个维度(mini_batch)上重复每个组n次
    # repeat_interleave会按顺序重复每个元素
    x_repeated = torch.repeat_interleave(x_grouped, n, dim=1)
    
    # 现在形状是 (n, mini_batch*n, dim)
    # 重塑回原始维度顺序
    x_final = x_repeated.reshape(n*mini_batch*n, dim)
    
    return x_final