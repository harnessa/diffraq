import numpy as np
import matplotlib.pyplot as plt;plt.ion()

inn_file = 'full_M12P6'
out_file = 'M12P6_1x'

xtra_dir = './xtras'

#Load from file
with open(f'{xtra_dir}/{inn_file}.txt', 'r') as f:
    lines = f.readlines()

#Read data
data,  tmp = [], []
for ln in lines:
    #Ignore build process metadata
    if ln.startswith('@@'):
        continue
    #Read metadata for petal with data if '$$' line
    elif ln.startswith('$$'):
        continue
    #Read metadata for petal without data if '$X' line
    elif ln.startswith('$X'):
        continue
    #Start new shape if '#' line
    elif ln.startswith('#'):
        data.append(np.array(tmp))
        tmp = []
        continue

    #Get data
    tmp.append(np.array(ln.split(',')).astype(float))

#Get last shape
if len(tmp) > 0:
    data.append(np.array(tmp))

#Plot
for i in range(len(data)):
    plt.plot(*data[i].T)

#Write out
with open(f'{xtra_dir}/{out_file}.dat', 'w') as f:
    for i in range(len(data)):
        for j in range(len(data[i])):
            f.write(f'{data[i][j][0]}, {data[i][j][1]}\n')
        f.write('**,**\n')


# breakpoint()
