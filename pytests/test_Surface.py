import pytest
import os
import numpy as np
import matplotlib.pyplot as plt

from pysubsurface.objects import Surface, SurfacePair

surfacefile = 'testdata/Surface/dsg5_long.txt'

par1 = {'finegrid':0, 'gridints':[-1, -1], 'nil':84, 'nxl':2343, 'label':1}
par2 = {'finegrid':1, 'gridints':[-1, -1], 'nil':84, 'nxl':2343, 'label':2}
par3 = {'finegrid':2, 'gridints':[0.5, 0.5], 'nil':167, 'nxl':4685, 'label':3}
par4 = {'finegrid':2, 'gridints':[1, 0.5], 'nil':84, 'nxl':4685, 'label':4}
par5 = {'finegrid':2, 'gridints':[0.5, 1], 'nil':167, 'nxl':2343, 'label':5}


def test_surface_create():
    # Create Surface from scratch, save it to file and check data is the same
    ny, nx = 101, 50
    data = np.random.normal(0, 1, (nx, ny))

    hor = Surface(None)
    hor.create_surface(il=np.arange(nx), xl=np.arange(ny),
                       y=np.arange(ny) * 10 + 100.,
                       x=np.arange(nx) * 10 + 100.,
                       data=data)
    hor.save('testdata/Surface/surfacefromscratch.txt',
             'fromscratch', format='plain_long', fmt='%d,%d,%.4f,%.4f,%.8f',
             regulargrid=True, verb=False)
    hor1 = Surface('testdata/Surface/surfacefromscratch.txt',
                   'plain_long', loadsurface=True)
    assert hor == hor1
    os.remove('testdata/Surface/surfacefromscratch.txt')

"""
NEEDS CHANGED WITH NEW SURFACE
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5)])
def test_read_surface(par):
    # Test various ways of loading a surface
    d = Surface(filename=surfacefile, format='dsg5_long',
                finegrid=par['finegrid'], gridints=par['gridints'],
                loadsurface=True)

    # check first element
    assert d._ils_orig[0] == 857.00
    assert d._xls_orig[0] == 1040.00
    assert d._xs_orig[0] == 479603.00
    assert d._ys_orig[0] == 6681599.90
    assert d._zs_orig[0] == 2809.4470215
    assert d.shape == (par['nil'], par['nxl'])


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5)])
def test_save_copy_surface(par):
    # Test various ways of loading and saving a surface
    d = Surface(filename=surfacefile, format='dsg5_long',
                finegrid=par['finegrid'], gridints=par['gridints'],
                loadsurface=True)
    d.save('testdata/Surface/surfacecopy{}.txt'.format(par['label']), 'NEW',
           fmt='%.2f %.2f %.8f %.8f %.8f')

    dcopy = d.copy()
    dsave = Surface(filename='testdata/Surface/surfacecopy{}.txt'.format(par['label']),
                    format='dsg5_long', loadsurface=True)
    assert d == dcopy
    assert d == dsave
    os.remove('testdata/Surface/surfacecopy{}.txt'.format(par['label']))

"""

@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5)])
def test_magic_surfaces(par):
    # Test magic commands overloaded in Surface
    d = Surface(filename=surfacefile, format='dsg5_long',
                finegrid=par['finegrid'], gridints=par['gridints'],
                loadsurface=True)
    dcopy = d.copy()
    dsum = d + d
    ddiff = d - d

    assert dcopy == d
    assert np.allclose(d.data*2, dsum.data, equal_nan=True)
    assert np.allclose(d.data*0, ddiff.data, equal_nan=True)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5)])
def test_same_grid(par):
    # Test one surface compared with itself returns the same grid
    d = Surface(filename=surfacefile, format='dsg5_long',
                finegrid=par['finegrid'], gridints=par['gridints'],
                loadsurface=True)
    assert d.same_grid(d)


@pytest.mark.parametrize("par", [(par1), (par3)])
def test_view_surfaces(par):
    # Test view command for Surface
    d = Surface(filename=surfacefile, format='dsg5_long',
                finegrid=par['finegrid'], gridints=par['gridints'],
                loadsurface=True)
    _, _ = d.view()
    _, _ = d.view(which='yx')
    _, _ = d.view(which='yx', figstyle='black', interp='sinc', cmap='jet',
                  ncountour=5, cbar=True, cbartitle='test',
                  chist=True, nhist=21, flipaxis=True, originlower=True,
                  flipy=True, flipx=True, axiskm=True, scalebar=True)

    fig, ax = plt.subplots(1, 1)
    _, _ = d.view(ax)
    _, _ = d.view(savefig = 'testfigs/surface_test{}.png'.format(par['label']))
    plt.close('all')

    # clean up
    plt.close('all')
    os.remove('testfigs/surface_test{}.png'.format(par['label']))


"""
@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4), (par5)])
def test_read_surfacepair(par):
    # Test various ways of loading a SurfacePair
    d = SurfacePair(filename1=surfacefile,
                    filename2=surfacefile, format='dsg5_long',
                    finegrids=par['finegrid'], gridints=par['gridints'],
                    loadsurfaces=True)

    # check first element
    assert d.surface1._ils_orig[0] == 857.00
    assert d.surface1._xls_orig[0] == 1040.00
    assert d.surface1._xs_orig[0] == 479603.00
    assert d.surface1._ys_orig[0] == 6681599.90
    assert d.surface1._zs_orig[0] == 2809.4470215
    assert d.surface1.shape == (par['nil'], par['nxl'])
    assert d.surface1 == d.surface2
"""