from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import timeit
import numpy as np
from cr_elimination import Cosmic_Ray_Elimination
import os
cwd = os.getcwd()

def test_init():
    the_object = Cosmic_Ray_Elimination()
    start_time = timeit.default_timer()
    the_object2 = Cosmic_Ray_Elimination()
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    if elapsed > 5:
        assert False
    elif the_object.model == the_object2.model:
        assert False
    else:
        assert True

def test_fill_space():
    the_object = Cosmic_Ray_Elimination()
    keepers_binary = np.zeros((64,64))
    scale_binary = np.zeros((64,64))
    scale_binary[0:30, 0:30] = scale_binary[0:30, 0:30] + 1
    keepers_binary[3,3] = 1
    keepers_binary[-3,-3] = 1
    keepers_binary = the_object.fill_space((3,3), keepers_binary, scale_binary, max_dim=None)
    keepers_binary = the_object.fill_space((-3,-3), keepers_binary, scale_binary, max_dim=1)
    print(np.where(keepers_binary - scale_binary == -1))
    assert np.sum(keepers_binary - scale_binary) == 0

def test_remove_cosmic_rays():
    print(cwd)
    filename = 'test_resources/kb200221_00042'
    image_file = get_pkg_data_filename(filename+'.fits')
    image_data = fits.getdata(image_file, ext=0)

    the_object = Cosmic_Ray_Elimination()
    pred_answer = the_object.remove_cosmic_rays(image_data)

    assert isinstance(pred_answer, np.ndarray) and len(pred_answer.shape) == 2
