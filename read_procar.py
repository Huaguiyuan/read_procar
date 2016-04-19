##
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import easygui


parameters = {'data_set': [True, False, False, False], 'orbital_set':[i for i in range(9)], 'energy_window':[3, 6]}
parameters['scale'] = 500

parameters['ions']= [[0,1,2], [3,4]]
# [[28, 29, 42, 43], [56,57,70,71]]
# [[40,41,54,55],[68,69,82,83]]
# [[i for i in range(20, 40)],[i for i in range(40, 60)]]
parameters['orbital'] = [[1,2,3], [1,2,3]]
parameters['color_map'] = ['r', 'b', 'y', 'g']
##

def read_CONTCAR(file=open('CONTCAR', 'r')):
    file.readline()
    scale = float(file.readline())
    a1 = file.readline().split()
    a2 = file.readline().split()
    a3 = file.readline().split()
    a = [a1, a2, a3]
    a = np.array(a, dtype='float') * scale
    next(file)
    '''
    num = np.array(file.readline().split(), dtype='int')
    species = np.zeros(len(num))
    cnt = 0
    for i in num:
        cnt += num[i]
        species[i] = cnt
    '''
    file.close()
    return a


def get_b(a):  # this funciton is to get reciprocal lattice from primitive lattice

    v = np.dot(a[0], np.cross(a[1], a[2]))
    b = []
    b.append(2 * np.pi * np.cross(a[1], a[2]) / v)
    b.append(2 * np.pi * np.cross(a[2], a[0]) / v)
    b.append(2 * np.pi * np.cross(a[0], a[1]) / v)
    b = np.array(b, dtype='float')
    return b


def point_scale(pt, a):
    # point_scale, including rpt and kpt, if it is rpt, put a as lattice, if it is kpt, put a as inversed lattice
    pt_scaled = a[0, :] * pt[0] + a[1, :] * pt[1] + a[2, :] * pt[2]
    pt_scaled = np.array(pt_scaled, dtype='float')
    return pt_scaled


def get_k_distance(klist, b):
    kd = [0]
    tmp = 0
    for i in range(len(klist) - 1):
        tmp += LA.norm(point_scale(klist[i + 1][0:-1] - klist[i][0:-1], b))
        kd.append(tmp)
    return kd


def read_procar(parameters, file=open('PROCAR','r')):

    next(file)
    buffer = file.readline().split()
    nkpt = int(buffer[3])
    nband = int(buffer[7])
    nions = int(buffer[11])

    kpt = np.zeros((nkpt, 4), dtype='float')
    occ = np.zeros((nkpt, nband), dtype='float')
    eng = np.zeros((nkpt, nband), dtype='float')
    data = np.zeros((nkpt, nband, parameters['data_set'].count(True), nions + 1, len(parameters['orbital_set'])+1), dtype='float')


    for i in range(nkpt):
        next(file)
        buffer = file.readline()
        k1 = float(buffer[18:29])
        k2 = float(buffer[29:40])
        k3 = float(buffer[40:51])
        k4 = float(buffer[-11:])
        kpt[i,:]=np.array([k1, k2, k3, k4], dtype='float')
        next(file)
        for j in range(nband):
            buffer = file.readline().split()
            occ[i,j] = buffer[-1]
            eng[i,j] = float(buffer[4])
            next(file)
            next(file)
            cnt = 0
            for k in parameters['data_set']:
                if k == False:
                    for l in range(nions+1):
                        next(file)
                else:
                    for l in range(nions+1):
                        buffer = file.readline().split()
                        data[i, j, cnt, l, :] = np.array(buffer[1:], dtype='float')
                    cnt += 1
            next(file)
    dict = {'nkpt': nkpt, 'nband': nband, 'nions': nions, 'kpt':kpt, 'occ':occ, 'eng':eng, 'data': data}

    file.close()

    return dict

#file_name = easygui.fileopenbox(default='/home/liuxy/Documents/workspace/KHgSb/')
#file = open(file_name, 'r')
dict = read_procar(parameters)
#print(dict['data'][0:20, 729:736, :])
print("reading process is done")

a = read_CONTCAR()
##


def band_analyze(dict, parameters):
    nkpt, nband, ndataset, nions, norbital = np.shape(dict['data'])
    print("data shape", nkpt, nband, ndataset, nions, norbital)

    # sum ions and orbitals
    ions_set = len(parameters['ions'])
    # orbitals_set = len(parameters['orbital']) orbital set should equal to ions set
    data = np.zeros((nkpt, nband, ndataset, ions_set), dtype='float')
    for i in range(ions_set):
        for k in parameters['ions'][i]:
            for l in parameters['orbital'][i]:
                data[:,:,:,i] += dict['data'][:,:,:, k, l]

    print("data shape after sum ion", np.shape(data))
    dict['data'] = data
    return


def write_pband(dict, parameters):
    nkpt, nband, ndataset, norbital = np.shape(dict['data'])
    indices = [i for i, x in enumerate(parameters['data_set']) if x == True]
    for i in range(ndataset):
        filename = 'pband-'+ str(indices[i])
        file = open(filename, 'w')
        print('nkpt', nkpt, 'nband', nband, 'data set', i, file=file)
        for j in range(nband):
            for k in range(nkpt):
                line = '{0: 4d}    {1: 8f}    {2: 8f}    '.format(j, kd[k], dict['eng'][k, j])
                for l in range(norbital):
                    line += '{: 4f}    '.format(dict['data'][k, j, i, l])
                print(line.strip(), file=file)
            print("\n", file=file)


##
def plot_band(kd, parameters, dict, data_set):
    # plot
    # x = kd
    # y = eng[:,i]
    lwidths = dict['data'][:, :, data_set, :]
    nkpt, nband, norbitals = np.shape(lwidths)
    print("lwidth shape", nkpt, nband, norbitals)
    scale = parameters['scale']
    alpha = [1-1/norbitals * i for i in range(norbitals)]
    print("alpha",alpha)
    for i in range(nband):
        # plot band structure
        plt.plot(kd, dict['eng'][:, i], 'k-')

        for j in range(norbitals) :
            for k in range(nkpt):
                c=parameters['color_map'][j]
                if parameters['energy_window'][0] < dict['eng'][k, i] < parameters['energy_window'][1]:
                    plt.scatter(kd[k], dict['eng'][k, i], s=lwidths[k, i, j]*scale, color=c, alpha=alpha[j])

    plt.axis([kd[0], kd[-1], parameters['energy_window'][0], parameters['energy_window'][1]])
    plt.show()
    return

b = get_b(a)
kd = get_k_distance(dict['kpt'], b)
band_analyze(dict, parameters)
write_pband(dict, parameters)
#plot_band(kd, parameters, dict, 0)

print("All done!")