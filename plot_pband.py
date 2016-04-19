import numpy as np
import matplotlib.pyplot as plt
import easygui


color_map = ['r', 'b', 'y', 'g']
scale = 500
energy_window = [3,6]

# read pband-0
file = open('pband-0')
tmp = file.readline().split()
nkpt = int(tmp[1])
nband = int(tmp[3])
data = np.loadtxt(file)
a,b = np.shape(data)
norbitals = b-3
file.close()

print('nkpt', nkpt, 'nband', nband)
print('data shape', np.shape(data))
data = data[:,1:]
data = np.reshape(data, (nband, nkpt, b-1))

alpha = [1-1/norbitals * i for i in range(norbitals)]
print("alpha",alpha)
x = data[0,:,0]
for i in range(nband):
    # plot band structure
    plt.plot(x, data[i, :, 1], 'k-')

    for j in range(norbitals) : # note here is  number of orbitals , need to add 2 when extracting data
        for k in range(nkpt):
            c=color_map[j]
            if energy_window[0] < data[i, k, 1] < energy_window[1]:
                plt.scatter(x[k], data[i, k, 1], s=data[i, k, j+2]*scale, color=c, alpha=alpha[j])

plt.axis([x[0], x[-1], energy_window[0], energy_window[1]])
plt.show()