from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import timeit
import numpy as np
import sys
sys.path.append('../cr_elimination')
from cr_elimination import Cosmic_Ray_Elimination

def test_init():
    the_object = Cosmic_Ray_Elimination()
    start_time = timeit.default_timer()
    the_object2 = Cosmic_Ray_Elimination()
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    assert the_object.model == the_object2.model and elapsed < 5

def test_fill_space():
    the_object = Cosmic_Ray_Elimination()
    keepers_binary = np.zeros(64,64)
    scale_binary = np.zeros(64,64)
    scale_binary[0:32, 0:32] = scale_binary[0:32, 0:32] + 1
    keepers_binary[3,3] = 1
    keepers_binary[-3,-3] = 1
    keepers_binary = the_object.fill_space((3,3), keepers_binary, scale_binary)
    keepers_binary = the_object.fill_space((-3,-3), keepers_binary, scale_binary)

    assert keepers_binary == scale_binary

def test_remove_cosmic_rays():
    filename = 'kb200221_00042'
    image_file = get_pkg_data_filename(filename+'.fits')
    image_data = fits.getdata(image_file, ext=0)

    the_object = Cosmic_Ray_Elimination()
    pred_answer = the_object.remove_cosmic_rays(image_data)

    true_answer = np.load("no_cr_kb200221_00042.npy")

    assert np.sum(np.abs(pred_answer - true_answer)) < 5
