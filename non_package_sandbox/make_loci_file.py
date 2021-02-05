import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

num_pts = 10000

is_circle =  True

if is_circle:
    name = 'circle'
    xfunc = lambda t: 12*np.cos(t)
    yfunc = lambda t: 12*np.sin(t)
else:
    name = 'test'
    xfunc = lambda t: 0.5*np.cos(t) + 0.5*np.cos(2*t)
    yfunc = lambda t: np.sin(t)

the = np.linspace(0, 2*np.pi, num_pts)

xx = xfunc(the)
yy = yfunc(the)

plt.plot(xx, yy, 'x-')

with open(f'{diffraq.int_data_dir}/Test_Data/{name}_loci_file.txt', 'w') as f:
    for i in range(len(xx)):
        f.write(f'{xx[i]},{yy[i]}\n')

breakpoint()
