import pvinspect
from pathlib import Path
import skimage.draw
import skimage.color
from skimage import img_as_ubyte, exposure, transform, io
import pandas as pd
import numpy as np
from matplotlib import cm, colors
from matplotlib import pyplot as plt


def cell_corners(corners, row, col):
    x0 = corners[col*7+row]
    x1 = corners[(col+1)*7+row]
    x2 = corners[(col+1)*7+row+1]
    x3 = corners[col*7+row+1]
    return np.array([x0, x1, x2, x3])


def visualize_fmap(img, fmap, path, prefix):
    # detection
    pvimg = pvinspect.data.ModuleImage(data=img, path=Path(), modality=pvinspect.data.EL_IMAGE, rows=6, cols=10)
    pvimg = pvinspect.preproc.detection.locate_module_and_cells(pvimg)
    corners = pvimg.get_meta('transform')(pvimg.grid())
    
    # resize fmap
    fmap_resized = transform.resize(fmap.T, (img.shape[0], img.shape[1]), order=1)
    fmap_resized_cubic = transform.resize(fmap.T, (img.shape[0], img.shape[1]), order=2)
    fmap_resized *= np.prod(fmap.shape)/np.prod(fmap_resized.shape)
    
    # color-code img with fmap
    norm = colors.Normalize(-np.max(fmap_resized_cubic), -np.min(fmap_resized_cubic))
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    fmap_color = mapper.to_rgba(-fmap_resized_cubic.squeeze())
    fmap_hsv = skimage.color.rgb2hsv(fmap_color[:,:,:3])
    fmap_hsv[:,:,2] = exposure.rescale_intensity(img, in_range=(img.min(), np.percentile(img, 95)), out_range=(0, 1.0))
    img_cc = skimage.color.hsv2rgb(fmap_hsv)

    # plot color-coded image
    plt.imshow(img_cc)
    
    # create fake mapper for colorbar
    mapper_100 = cm.ScalarMappable(norm=colors.Normalize(0, 100), cmap=cm.viridis)
    plt.colorbar(mapper_100)
    
    # sum per cell
    for row in range(6):
        for col in range(10):
            coords = cell_corners(corners, row, col)
            cx = coords[:,0].mean()
            cy = coords[:,1].mean()
            mask = skimage.draw.polygon2mask(pvimg.shape, np.flip(coords, axis=1)) # could be improved, since this is without interpolation
            v = np.sum(fmap_resized.squeeze()*mask)
            plt.text(cx, cy, '{:.1f}%'.format(v*100), horizontalalignment='center', verticalalignment='center', fontsize=5, bbox=dict(facecolor='white', alpha=0.8, lw=0))
    plt.savefig(path / '{}_cc_fig.png'.format(prefix), dpi=200)
    plt.clf()
   
    # save
    io.imsave(path / '{}.png'.format(prefix), img_as_ubyte(exposure.rescale_intensity(img, out_range=(0, 1.0))))
    io.imsave(path / '{}_cc.png'.format(prefix), img_as_ubyte(img_cc))