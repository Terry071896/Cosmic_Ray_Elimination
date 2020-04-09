import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import ZScaleInterval, AsinhStretch,SinhStretch, BaseStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.io import fits
import timeit
from cr_elimination import Cosmic_Ray_Elimination
import pyximport
pyximport.install()

import _lacosmicx

files = ['LR.20170118.45640', 'LR.20170118.36862', 'LR.20170118.39847', 'LR.20170118.44466',	'LR.20170118.46767', 'LR.20170118.38419', 'LR.20170118.44290', 'LR.20170118.45640', 'kb200221_00042']
#files = ['kb200221_00042', 'LR.20170118.45640']
files = ['kb200221_00042', 'LR.20170118.36862']
for i, file in enumerate(files):
    for model_name in ['model_new_round_org2']:#['model', 'model2', 'model_new', 'model_new_round', 'model_new_round_org']:
        filename = 'test_resources/%s'%file
        print(filename)
        image_file = get_pkg_data_filename(filename+'.fits')
        print(image_file)
        image_data = fits.getdata(image_file, ext=0)
        interval = ZScaleInterval()
        vmin,vmax = interval.get_limits(image_data)
        norm = ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
        scale = norm.__call__(image_data).data
        #model_name = 'model_new'
        print(model_name)
        cr = Cosmic_Ray_Elimination(model_name = model_name)
        #i = 1
        zscores = [2, 0.5]
        max_dims = [None, 0]
        box_widths = [5, 7]
        box_heights = [5, 5]
        new_image_data = cr.remove_cosmic_rays(scale, zscore=zscores[i], pixels_shift=256, max_dim=max_dims[i], box_width=box_widths[i], box_height=box_heights[i])

        #new_image_save = cr.remove_cosmic_rays(scale)
        #np.save('test_resources/%s_%s_medmin'%(model_name, file), new_image_data)

        cnn_subtract = scale - new_image_data

        interval = ZScaleInterval()
        vmin,vmax = interval.get_limits(new_image_data)
        norm = ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
        new_scale = norm.__call__(new_image_data)
        print(cr.times)

        CRR_MINEXPTIME = 60.0
        CRR_PSSL = 0.0
        CRR_GAIN = 1.0
        CRR_READNOISE = 3.2
        CRR_SIGCLIP = 4.5
        CRR_SIGFRAC = 0.3
        CRR_OBJLIM = 4.0
        CRR_PSFFWHM = 2.5
        CRR_FSMODE = "median"
        CRR_PSFMODEL = "gauss"
        CRR_SATLEVEL = 60000.0
        CRR_VERBOSE = False
        CRR_SEPMED = False
        CRR_CLEANTYPE = "meanmask"
        CRR_NITER = 4
        CRR_SIGCLIP = 0.00001
        read_noise = 3
        start_time = timeit.default_timer()
        #lacos = _lacosmicx.lacosmicx(image_data.astype(np.float32), sigclip = .1)
        # mask, la_scale = _lacosmicx.lacosmicx(
        #     scale.astype(np.float32), gain=1.0, readnoise=CRR_READNOISE,
        #     psffwhm=CRR_PSFFWHM,
        #     sigclip=CRR_SIGCLIP,
        #     sigfrac=CRR_SIGFRAC,
        #     objlim=CRR_OBJLIM,
        #     fsmode=CRR_FSMODE,
        #     psfmodel=CRR_PSFMODEL,
        #     verbose=CRR_VERBOSE,
        #     sepmed=CRR_SEPMED,
        #     cleantype=CRR_CLEANTYPE)
        # elapsed = timeit.default_timer() - start_time
        # print('LA Cosmic: %s'%elapsed)
        #
        # la_subtract = scale - la_scale
        # interval = ZScaleInterval()
        # vmin,vmax = interval.get_limits(la_scale)
        # norm = ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
        # la_scale = norm.__call__(la_scale)

        cr = Cosmic_Ray_Elimination(model_name = model_name)
        #i = 1
        zscores = [2, 0.5]
        max_dims = [0, 0]
        box_widths = [1, 1]
        box_heights = [1, 1]
        la_scale = cr.remove_cosmic_rays(scale, zscore=zscores[i], pixels_shift=256, max_dim=max_dims[i], box_width=box_widths[i], box_height=box_heights[i])
        la_subtract = scale - la_scale
        interval = ZScaleInterval()
        vmin,vmax = interval.get_limits(la_scale)
        norm = ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
        la_scale = norm.__call__(la_scale)

        matplotlib.rcParams['figure.figsize'] = (int(new_image_data.shape[0]*0.01), int(new_image_data.shape[1]*0.01))
        i = 1
        if i == 0:
            plt.imshow(scale, cmap='gray')
            plt.show()

            fig = plt.figure()
            # ax1 = fig.add_subplot(1,3,1)
            # ax1.imshow(scale, cmap='gray')
            # ax1.set_title('Original')
            ax2 = fig.add_subplot(1,2,1)
            ax2.imshow(new_scale, cmap='gray')
            ax2.set_title('CNN Method')
            ax3 = fig.add_subplot(1,2,2)
            ax3.imshow(la_scale, cmap='gray')
            ax3.set_title('Without \'Filling the Gaps\'')
            plt.show()

            fig = plt.figure()
            ax2 = fig.add_subplot(1,2,1)
            ax2.imshow(cnn_subtract, cmap='gray')
            ax2.set_title('CNN Method')
            ax3 = fig.add_subplot(1,2,2)
            ax3.imshow(la_subtract, cmap='gray')
            ax3.set_title('Without \'Filling the Gaps\'')
            plt.show()
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,3,1)
            ax1.imshow(scale, cmap='gray')
            ax1.set_title('Original')
            ax2 = fig.add_subplot(1,3,2)
            ax2.imshow(new_scale, cmap='gray')
            ax2.set_title('CNN Method')
            ax3 = fig.add_subplot(1,3,3)
            ax3.imshow(la_scale, cmap='gray')
            ax3.set_title('Without \'Filling the Gaps\'')
            plt.show()

            fig = plt.figure()
            ax1 = fig.add_subplot(1,3,1)
            ax1.imshow(scale, cmap='gray')
            ax1.set_title('Original')
            ax2 = fig.add_subplot(1,3,2)
            ax2.imshow(cnn_subtract, cmap='gray')
            ax2.set_title('CNN Method')
            ax3 = fig.add_subplot(1,3,3)
            ax3.imshow(la_subtract, cmap='gray')
            ax3.set_title('Without \'Filling the Gaps\'')
            plt.show()
