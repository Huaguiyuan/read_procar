##
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#from matplotlib.collections import LineCollection
import easygui


parameters = {'data_set': [True, False, False, False], 'orbital_set':[i for i in range(9)], 'energy_window':[0, 2.5]}
parameters['scale'] = 500

parameters['ions']= [[20.21, 30, 31], [40, 41, 50, 51]]
# [[28, 29, 38, 39], [48, 49, 58, 59]]
# [[20.21, 30, 31], [40, 41, 50, 51]]
# [[i for i in range(20, 40)],[i for i in range(40, 60)]]
parameters['orbital_sum'] = [[0],[1,2,3]]
parameters['color_map'] = ['r', 'b', 'g']
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

file_name = easygui.fileopenbox(default='/home/liuxy/Documents/workspace/KHgSb/')
file = open(file_name, 'r')
dict = read_procar(parameters, file=file)
print("reading process is done")
print("nband", dict['nband'])

a = read_CONTCAR()
##


def band_analyze(dict, parameters):
    nkpt, nband, ndataset, nions, norbital = np.shape(dict['data'])
    print("data shape", nkpt, nband, ndataset, nions, norbital)

    # sum ions
    nion_sets = len(parameters['ions'])
    data = np.zeros((nkpt, nband, ndataset, nion_sets, norbital), dtype='float')
    for i in range(nion_sets):
        for k in parameters['ions'][i]:
            data[:, :, :, i, :] += dict['data'][:, :, :, k, :]
    print("data shape after sum ion", np.shape(data))

    # sum orbitals
    norbital_set = len(parameters['orbital_sum'])
    data1 = np.zeros((nkpt, nband, ndataset, nion_sets, norbital_set))
    for i in range(norbital_set):
        for j in parameters['orbital_sum'][i]:
            data1[:, :, :, :, i] += data[:,:,:,:,j]
    print("data shape after sum orbitals", np.shape(data1))
    del data

    # construct line_width, color
    # we think nion_sets should be same with norbital_set
    linewidth = np.zeros((nkpt, nband, ndataset, nion_sets), dtype='float')
    color = np.zeros((nkpt, nband, ndataset, nion_sets), dtype='int')
    for i in range(nkpt):
        for j in range(nband):
            for k in range(ndataset):
                tmp  = np.zeros(nion_sets, dtype='float')
                for l in range(nion_sets):
                    tmp[l] = data1[i, j, k, l, l]
                linewidth[i, j, k, :] = np.sort(tmp)
                color[i, j, k, :] = np.argsort(tmp) + 1
    if 0 in color:
        print("ERROR: there is a 0 in color matrix")

    dict['linewidth'] = linewidth
    dict['color'] = color
    return

##
def plot_band(kd, parameters, dict, data_set):
    # plot
    # x = kd
    # y = eng[:,i]

    for i in range(dict['nband']):
        # only present the largest contribution
        lwidths = dict['linewidth'][:,i,data_set, 0]
        scale = parameters['scale']
        #print(lwidths)
        color = dict['color'][:,i, data_set,0]
        #points = np.array([kd, dict['eng'][:,i]]).T.reshape(-1, 1, 2)
        #segments = np.concatenate([points[:-1], points[1:]], axis=1)
        plt.plot(kd, dict['eng'][:, i], 'k-')

        for j in range(len(lwidths)-1) :
            #if not np.allclose(lwidths[j],0):
            c=parameters['color_map'][color[j]-1]
            #print(c)
            #print(kd[j:j+2], dict['eng'][j:j+2, i])
            if parameters['energy_window'][0] < dict['eng'][j, i] < parameters['energy_window'][1] and \
                parameters['energy_window'][0] < dict['eng'][j+1, i] < parameters['energy_window'][1] \
                    and not np.allclose(lwidths[j], 0, atol= 1e-3):
                plt.plot(kd[j:j+2], dict['eng'][j:j+2, i], linewidth=lwidths[j]*scale, color=c)

    plt.axis([kd[0], kd[-1], parameters['energy_window'][0], parameters['energy_window'][1]])
    plt.show()
    return




b = get_b(a)
kd = get_k_distance(dict['kpt'], b)
band_analyze(dict, parameters)
plot_band(kd, parameters, dict, 0)
print("All done!")