import math
import os

import cv2
import numpy as np
from tqdm import trange
import joblib

class LocalLaplaceFilter:
    def __init__(self, input_image: np.ndarray, kAlphaIn: float, kBetaIn: float, kSigmaRIn: float, skala: int, num_workers: int = 1, verbose: int = 1):
        self.kAlpha = kAlphaIn
        self.kBeta = kBetaIn
        self.kSigmaR = kSigmaRIn
        self.rows = 0
        self.cols = 0
        self.result = 0
        self.color = ''
        self.num_workers = num_workers
        self.verbose = verbose
        self.gaussian_pyramid = []
        self.laplacian_pyramid = []
        self.output_write = []
        self.reconstructed_image = None
        self.width = 0
        self.height = 0
        self.dim = (self.width, self.height)

        # self.input = cv2.imread(self.inputFileName, cv2.IMREAD_UNCHANGED)
        self.input = input_image
        if len(self.input.shape) == 2:
            self.color = 'lum'
        else:
            self.color = 'rgb'
        self.scale_percent = skala  # percent of original size
        self.width = int(math.ceil(self.input.shape[1] * self.scale_percent / 100))
        self.height = int(math.ceil(self.input.shape[0] * self.scale_percent / 100))
        self.dim = (self.width, self.height)
        if self.input is not None:
            self.img_norm2 = self.input.astype(float) / 255.0
            self.img_resized = cv2.resize(self.img_norm2, self.dim, interpolation=cv2.INTER_AREA)
            self.result = 1
        else:
            self.result = 0
        self.num_levels = self.get_num_levels(self.img_resized)

    # number of pyramid levels, as many as possible up 1x1
    def get_num_levels(self, image: np.ndarray) -> int:
        self.rows, self.cols = image.shape[:2]
        min_d = min(self.rows, self.cols)
        nlev = 1
        while min_d > 1:
            nlev = nlev + 1
            min_d = (min_d + 1) // 2
        return nlev

    # 2D for building Gaussian and Laplacian pyramids
    def filter(self):
        a = [.05, .25, .4, .25, .05]
        kernel_1 = np.array(a, np.float64)
        kernel_2 = np.array(kernel_1)[np.newaxis]
        kernel = kernel_2.T
        f = np.multiply(kernel, kernel_1, None)
        return f

    # smooth step edge
    def smooth_step(self, xmin, xmax, x):
        y = (x - xmin) / (xmax - xmin)
        y = np.minimum(y, 1)
        y = np.maximum(y, 0)
        y_1 = np.multiply(y, y - 2)
        y_2 = np.square(y_1)
        return y_2

    # detail remapping function
    def fd(self, d):
        noise_level = float(0.01)
        out = d ** self.kAlpha
        if self.kAlpha < 1.0:
            tau = self.smooth_step(noise_level, 2*noise_level, d*self.kSigmaR)
            out = tau * out + (1-tau) * d
        return out

    # edge remapping function
    def fe(self, a):
        out = self.kBeta * a
        return out

    # color remapping function
    def r_color(self, i, g0, sigma_r):
        g0 = np.tile(g0, [i.shape[0], i.shape[1], 1])
        dnrm = np.sqrt(np.sum((i - g0) ** 2, axis=2))
        eps_dnrm = np.spacing(1) + dnrm
        unit = (i - g0) / np.tile(eps_dnrm[..., None], [1, 1, 3])
        rd = g0 + unit * np.tile(sigma_r * self.fd(dnrm / sigma_r)[..., None], [1, 1, 3])
        re = g0 + unit * np.tile(sigma_r + self.fe(dnrm - sigma_r)[..., None], [1, 1, 3])
        isedge = np.tile((dnrm > sigma_r)[..., None], [1, 1, 3])
        return np.logical_not(isedge) * rd + isedge * re

    # grayscale remapping function
    def r_gray(self, i, g0, sigma_r):
        dnrm = abs(i-g0)
        dsgn = np.sign(i-g0)

        rd = g0 + dsgn*sigma_r*self.fd(dnrm/sigma_r)
        re = g0 + dsgn*(self.fe(dnrm - sigma_r) + sigma_r)

        isedge = dnrm > sigma_r
        return np.logical_not(isedge) * rd + isedge * re

    def remapping(self, image: np.ndarray, gauss: np.ndarray):
        if self.color == 'rgb':
            return self.r_color(image, gauss, self.kSigmaR)
        if self.color == 'lum':
            return self.r_gray(image, gauss, self.kSigmaR)
        return None

    def child_window(self, parent):
        child = parent.copy()

        child[0] = math.ceil((float(child[0]) + 1.0) / 2.0)
        child[2] = math.ceil((float(child[2]) + 1.0) / 2.0)
        child[1] = math.floor((float(child[1]) + 1.0) / 2.0)
        child[3] = math.floor((float(child[3]) + 1.0) / 2.0)
        return child

    def upsample(self, image, subwindow):
        r = int(subwindow[1] - subwindow[0] + 1)
        c = int(subwindow[3] - subwindow[2] + 1)
        if self.color == 'rgb':
            k = image.shape[2]

        reven = int(subwindow[0] % 2 == 0)
        ceven = int(subwindow[2] % 2 == 0)

        if self.color == 'lum':
            R = np.zeros((r, c))
            Z = np.zeros((r, c))
        if self.color == 'rgb':
            R = np.zeros((r, c, k))
            Z = np.zeros((r, c, k))
        kernel = self.filter()

        if self.color == 'lum':
            if R[reven: r: 2, ceven: c: 2].shape != image.shape:
                rslice = slice(reven, r, 2).indices(r)
                cslise = slice(ceven, c, 2).indices(c)
                np.put(R, [rslice, cslise], image.copy())
            else:
                R[reven: r: 2, ceven: c: 2] = image.copy()
            Z[reven : r : 2, ceven : c : 2] = 1.0

        if self.color == 'rgb':
            if R[reven: r: 2, ceven: c: 2, :].shape != image.shape:
                rslice = slice(reven, r, 2).indices(r)
                cslise = slice(ceven, c, 2).indices(c)
                np.put(R, [rslice, cslise], image.copy())
            else:
                R[reven: r: 2, ceven: c: 2, :] = image.copy()
            Z[reven : r : 2, ceven : c : 2, :] = 1.0

        R = cv2.filter2D(R, -1, kernel)
        Z = cv2.filter2D(Z, -1, kernel)
        R /= Z

        return R

    def downsample(self, image: np.ndarray, subwindow: tuple):
        r = image.shape[0]
        c = image.shape[1]

        if not subwindow:
            subwindow = np.arange(r * c).reshape(r, c)
        subwindow_child = self.child_window(subwindow)
        R, Z = None, None
        kernel = self.filter()

        R = cv2.filter2D(image.astype(np.float64), -1, kernel)
        
        if self.color == 'rgb':
            Z = np.ones([r, c, 3], dtype=np.float64)
        if self.color == 'lum':
            Z = np.ones([r, c], dtype=np.float64)

        Z = cv2.filter2D(Z, -1, kernel)
        R /= Z

        reven = int(subwindow[0] % 2 == 0)
        ceven = int(subwindow[2] % 2 == 0)

        if self.color == 'rgb':
            R = R[reven: r: 2, ceven: c: 2, :]
        if self.color == 'lum':
            R = R[reven: r: 2, ceven: c: 2]
            
        return R, subwindow_child

    def reconstruct_laplacian_pyramid(self, subwindow=None):
        nlev = self.num_levels
        subwindow_all = np.zeros((nlev, 4))
        if not subwindow:
            subwindow_all[1, :] = [1, self.height, 1, self.cols]
        else:
            subwindow_all[1, :] = subwindow
        for lev in range(2, nlev):
            subwindow_all[lev, :] = self.child_window(subwindow_all[lev - 1, : ])
        R = self.laplacian_pyramid[nlev - 1].copy()
        for lev in range(nlev - 1, 0, -1):
            upsampled = self.upsample(R, subwindow_all[lev, :])
            R = np.add(self.laplacian_pyramid[lev - 1], upsampled)
        return R

    def gauss_pyramid(self, image: np.ndarray, nlev: int, subwindow):
        r = image.shape[0]
        c = image.shape[1]
        if not subwindow:
            subwindow = [1, r, 1, c]
        if not nlev:
            nlev = self.get_num_levels(image)
        pyr = [None] * nlev
        pyr[0] = image.copy()
        for level in range(1, nlev):
            image, _ = self.downsample(image, subwindow)
            pyr[level] = image.copy()
        return pyr

    def laplace_pyramid(self, image: np.ndarray, nlev: int, subwindow: tuple):
        r = image.shape[0]
        c = image.shape[1]
        j_image = []
        if not subwindow:
            subwindow = [1, r, 1, c]
        if not nlev:
            nlev = self.get_num_levels(image)
        pyr = [None] * nlev
        for level in range(0, nlev - 1):
            j_image = image.copy()
            image, subwindow_child = self.downsample(j_image, subwindow)
            upsampled = self.upsample(image, subwindow)
            pyr[level] = j_image - upsampled
            subwindow = subwindow_child.copy()

        pyr[nlev - 1] = j_image
        return pyr

    def compute_laplace_layer(self, level: int):
        hw = 3 * 2 ** level - 2

        guassian_layer = self.gaussian_pyramid[level - 1]
        layer_h, layer_w = guassian_layer.shape[:2]

        Ys, Xs = np.meshgrid(np.arange(layer_h) + 1, np.arange(layer_w) + 1)
        Ys = Ys.flatten()
        Xs = Xs.flatten()

        laplace_layer = np.zeros_like(guassian_layer)

        if level == 1:
            iters = trange(len(Ys), colour='green', leave=False, ncols=80)
        else:
            iters = range(len(Ys))
            
        for i in iters:
            y = Ys[i]
            x = Xs[i]
            yf = (y - 1) * 2 ** (level - 1) + 1
            xf = (x - 1) * 2 ** (level - 1) + 1

            yrng = [max(1, yf - hw), min(self.dim[1], yf + hw)]
            xrng = [max(1, xf - hw), min(self.dim[0], xf + hw)]

            isub = self.img_resized[yrng[0] - 1:yrng[1], xrng[0] - 1: xrng[1]]

            if self.color == 'lum':
                gauss = guassian_layer[y - 1, x - 1]
            if self.color == 'rgb':
                gauss = guassian_layer[y - 1, x - 1, :]

            img_remapped = self.remapping(isub, gauss)

            l_remap = self.laplace_pyramid(img_remapped, level + 1, [yrng[0], yrng[1], xrng[0], xrng[1]])

            yfc = yf - yrng[0] + 1
            xfc = xf - xrng[0] + 1

            yfclev0 = math.floor((yfc - 1) / 2 ** (level - 1)) + 1
            xfclev0 = math.floor((xfc - 1) / 2 ** (level - 1)) + 1

            if self.color == 'rgb':
                laplace_layer[y - 1, x - 1, :] = l_remap[level - 1][yfclev0 - 1, xfclev0 - 1, :]
            if self.color == 'lum':
                laplace_layer[y - 1, x - 1] = l_remap[level - 1][yfclev0 - 1, xfclev0 - 1]


        return laplace_layer

    def print_message(self, *args, **kwargs):
        if self.verbose >= 1:
            print(*args, **kwargs)

    def local_laplace_filter(self):
        self.gaussian_pyramid = self.gauss_pyramid(self.img_resized, None, None)
        self.print_message('[√] Build Gaussian Pyramid')
        self.laplacian_pyramid = self.gaussian_pyramid.copy()

        if self.num_workers == 1:
            self.print_message('[√] Dispatch One CPU')
            for level in trange(1, self.num_levels, colour='green', ncols=80):
                laplace_layer = self.compute_laplace_layer(level)
                self.laplacian_pyramid[level - 1] = laplace_layer
        else:
            self.print_message(f'[√] Dispatch {os.cpu_count()} CPUs')
            parallel = joblib.Parallel(n_jobs=self.num_workers)
            laplace_layers = parallel(
                joblib.delayed(self.compute_laplace_layer)(level)
                for level in range(1, self.num_levels)
            )
            for i, layer in enumerate(laplace_layers):
                self.laplacian_pyramid[i] = layer

        self.print_message('[√]', 'Build Laplacian Pyramid')
        out = self.reconstruct_laplacian_pyramid()
        self.print_message('[√]', 'Rebuild Output Image')
        return out

def local_laplace_filter(image: np.ndarray, kAlphaIn: float = 0.2, kBetaIn: float = 0.5, kSigmaRIn: float = 0.8, 
                         skala: int = 100, num_workers: int = 1, verbose: int = 0, return_layers: bool = False) -> np.ndarray:
    """Implement of Local Laplace Filter
    From paper [Local Laplacian filters: Edge-aware image processing with a Laplacian pyramid.](http://people.csail.mit.edu/hasinoff/pubs/ParisEtAl11-lapfilters-lowres.pdf)

    Parameters
    ----------
    image : numpy array
        input image, support both `color` and `gray` type of image.
    kAlphaIn: float
        default is `0.2`
    kBetaIn: float
        default is `0.5`
    kSigmaRIn: float
        default is `0.8`
    skala: int
        scale factor, default is `10`, represent size of input
        image will be resize to 10% of original
    num_workers : int
        number of cpu to do the transform, default is `1`. 
        You can set -1 to dispatch all the cpus.
    verbose: int
        level to print the message, set it to `1` to print message 
        after each stage of transform, default is `0`.
    return_layers: bool
        If true, will return the each laplace layers during the first step.
        Default is false
    Returns
    -------
    result : numpy array 
        result after transform
    """
    if len(image.shape) >= 3:
        image = image[..., : 3]
    lp_filter = LocalLaplaceFilter(image, kAlphaIn, kBetaIn, kSigmaRIn, skala, num_workers, verbose)
    output = lp_filter.local_laplace_filter()
    result = np.clip(output * 255, 0, 255)

    if return_layers:
        layers = lp_filter.laplacian_pyramid
        layers = [np.clip(l * 255, 0, 255) for l in layers]
        return result, layers
    else:
        return result