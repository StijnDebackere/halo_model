import click
import numpy as np
import re
import os

@click.command()
@click.option('--filename', default=None, help='file to tabelize')
@click.option('--l',
              default=None,
              help='box size in Mpc, or taken from filename')
@click.option('--n',
              default=None,
              help='number of particles, or taken from filename')
def tabelize(filename, l, n):
    if l == None:
        l = re.search('_L([0-9]*)N([0-9]*)_', filename.split('/')[-1]).group(1)
        l = float(l)
        V = l**3
    else:
        V = float(l)**3
    if n == None:
        n = re.search('_L([0-9]*)N([0-9]*)_', filename.split('/')[-1]).group(2)

    n = float(n)
    k, P_rough, W = np.loadtxt(filename, skiprows=1, usecols=(0,3,4), unpack=True)
    k *= 2*np.pi / l
    if 'DMONLY' in filename:
        P_true = V * (P_rough - W/n**3)
        print 'N : %i^3'%n

    else:
        # double the number of particles
        P_true = V * (P_rough - W/(2*n**3))
        print 'N : 2x%i^3'%n

    print 'V : %.1f^3'%l
    file_split = filename.split('/')
    file_path = '/'.join(file_split[:-1]) + '/tables/'
    file_name = file_split[-1]
    new_filename = file_path + file_name[:file_name.rfind('.')] + '_table.dat'

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    data = np.column_stack([k, P_true, 1./(2*np.pi**2) * k**3 * P_true])
    print 'Saving to %s'%new_filename
    np.savetxt(new_filename, data,
               header = 'k [h/Mpc] P [(Mpc/h)^3] Delta')

if __name__ == '__main__':
    tabelize()
