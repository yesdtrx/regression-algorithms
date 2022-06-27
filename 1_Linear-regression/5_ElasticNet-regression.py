def elasticNet(X, y, lambdas=0.1, rhos=0.5, max_iter=1000, tol=1e-4):
    """
    弹性网络回归，使用坐标下降法（coordinate descent）
    args:
        X - 训练数据集
        y - 目标标签值
        lambdas - 惩罚项系数
        rhos - 混合参数，取值范围[0,1]
        max_iter - 最大迭代次数
        tol - 变化量容忍值
    return:
        w - 权重系数
    """
    # 初始化 w 为零向量
    w = np.zeros(X.shape[1])
    for it in range(max_iter):
        done = True
        # 遍历所有自变量
        for i in range(0, len(w)):
            # 记录上一轮系数
            weight = W[i]
            # 求出当前条件下的最佳系数
            w[i] = down(X, y, w, i, lambdas, rhos)
            # 当其中一个系数变化量未到达其容忍值，继续循环
            if (np.abs(weight - w[i]) > tol):
                done = False
        # 所有系数都变化不大时，结束循环
        if (done):
            break
    return w

def down(X, y, w, index, lambdas=0.1, rhos=0.5):
    """
    cost(w) = (x1 * w1 + x2 * w2 + ... - y)^2 / 2n + ... + λ * ρ * (|w1| + |w2| + ...) + [λ * (1 - ρ) / 2] * (w1^2 + w2^2 + ...)
    假设 w1 是变量，这时其他的值均为常数，带入上式后，其代价函数是关于 w1 的一元二次函数，可以写成下式：
    cost(w1) = (a * w1 + b)^2 / 2n + ... + λρ|w1| + [λ(1 - ρ)/2] * w1^2 + c (a,b,c,λ 均为常数)
    => 展开后
    cost(w1) = [aa / 2n + λ(1 - ρ)/2] * w1^2 + (ab / n) * w1 + λρ|w1| + c (aa,ab,c,λ 均为常数)
    """
    # 展开后的二次项的系数之和
    aa = 0
    # 展开后的一次项的系数之和
    ab = 0
    for i in range(X.shape[0]):
        # 括号内一次项的系数
        a = X[i][index]
        # 括号内常数项的系数
        b = X[i][:].dot(w) - a * w[index] - y[i]
        # 可以很容易的得到展开后的二次项的系数为括号内一次项的系数平方的和
        aa = aa + a * a
        # 可以很容易的得到展开后的一次项的系数为括号内一次项的系数乘以括号内常数项的和
        ab = ab + a * b
    # 由于是一元二次函数，当导数为零是，函数值最小值，只需要关注二次项系数、一次项系数和 λ
    return det(aa, ab, X.shape[0], lambdas, rhos)

def det(aa, ab, n, lambdas=0.1, rhos=0.5):
    """
    通过代价函数的导数求 w，当 w = 0 时，不可导
    det(w) = [aa / n + λ(1 - ρ)] * w + ab / n + λρ = 0 (w > 0)
    => w = - (ab / n + λρ) / [aa / n  + λ(1 - ρ)]

    det(w) = [aa / n + λ(1 - ρ)] * w + ab / n  - λρ = 0 (w < 0)
    => w = - (ab / n - λρ) / [aa / n  + λ(1 - ρ)]

    det(w) = NaN (w = 0)
    => w = 0
    """
    w = - (ab / n + lambdas * rhos) / (aa / n + lambdas * (1 - rhos))
    if w < 0:
        w = - (ab / n - lambdas * rhos) / (aa / n + lambdas * (1 - rhos))
        if w > 0:
            w = 0
    return w


