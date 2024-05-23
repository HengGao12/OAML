import numpy as np
from scipy.optimize import linprog

def wasserstein_dual(mu, nu, cost_matrix):
    """
    计算Wasserstein-1距离的Kantorovich-Rubinstein对偶形式
    
    参数：
    mu : 一维numpy数组
        表示源分布的权重
    nu : 一维numpy数组
        表示目标分布的权重
    cost_matrix : 二维numpy数组
        表示从源分布到目标分布的转移成本矩阵
        
    返回：
    Wasserstein-1距离的对偶形式的值
    """
    c = cost_matrix.flatten()
    A_eq = np.kron(np.eye(len(mu)), np.ones(len(nu))) + np.kron(np.ones(len(mu)), np.eye(len(nu)))
    b_eq = np.concatenate([mu, nu])
    
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    
    return result.fun

# 示例用法
mu = np.array([0.4, 0.6])  # 源分布的权重
nu = np.array([0.7, 0.3])  # 目标分布的权重
cost_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])  # 转移成本矩阵

distance = wasserstein_dual(mu, nu, cost_matrix)
print("Wasserstein-1距离的对偶形式:", distance)
