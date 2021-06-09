import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from diffraq.quadrature import lgwt

ncycles = 4
num_quad = 100

func = lambda x: np.sin(np.pi*x)


pr, wr = lgwt(num_quad, 0, 1)

# pr = np.linspace(0,1,num_quad,endpoint=False)[::-1]
# wr = np.ones(num_quad)/num_quad

pr = np.concatenate((pr[::-1], 1+pr[::-1]))
wr = np.concatenate((wr[::-1],   wr[::-1]))

for i in range(1,ncycles-1):
    pr = np.concatenate((pr, 2*i+pr))
    wr = np.concatenate((wr,     wr))

yr = func(pr)


ans = (yr * wr).sum()
abans = (abs(yr)*wr).sum()

tru = 1.2732395447351683*ncycles

print(ans, abans-tru)

plt.axhline(0,color='k')
plt.plot(pr, yr, 'x-')

breakpoint()
