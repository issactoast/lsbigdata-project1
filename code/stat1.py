# 균일확률변수 만들기
import numpy as np

np.random.rand(1)

def X(num):
    return np.random.rand(num)

X(1)

# 베르누이 확률변수 모수: p 만들어보세요!
# num = 3
# p = 0.5
def Y(num, p):
    x=np.random.rand(num)
    return np.where(x < p, 1, 0)

# sum(Y(num=100, p=0.5))/100
Y(num=1, p=0.5)
Y(num=100, p=0.5).mean()
Y(num=100000, p=0.5).mean()

# 새로운 확률변수
# 가질수있는 값: 0, 1, 2
# 20%, 50%, 30%
def Z():
    x=np.random.rand(1)
    return np.where(x < 0.2, 0, np.where(x < 0.7, 1, 2))

Z()

# p = np.array([0.2, 0.5, 0.3)
def Z(p):
    x=np.random.rand(1)
    p_cumsum=p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where(x < p_cumsum[1], 1, 2))

p = np.array([0.1, 0.6, 0.3])
Z(p)


