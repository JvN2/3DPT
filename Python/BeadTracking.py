from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from lmfit import Parameters, minimize
from tqdm import tqdm


def fit_sinc(image, show=False, center=[0, 0], k=None):
    def sinc(x, y, x0, y0, A, k, phi):
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        res = np.cos(phi) * np.sinc(2 * r / k) + 1j * np.sin(phi) * np.sinc(2 * r / k)
        return np.abs(A) * res

    def residuals(pars, x, data=None):
        p = [pars[i] for i in pars.keys()]
        res = sinc(x[0], x[1], *p).flatten()
        if data is not None:
            res -= data
        res = res.imag ** 2 + res.real ** 2
        return res

    N = len(image)
    Y, X = np.meshgrid(np.linspace(0, N - 1, N) - N / 2,
                       np.linspace(0, N - 1, N) - N / 2)

    fit_params = Parameters()
    fit_params.add('x0', value=center[0], max=N / 2, min=-N / 2)
    fit_params.add('y0', value=center[1], max=N / 2, min=-N / 2)
    fit_params.add('amplitude', value=np.max(np.abs(image)), min=0)
    if k is None:
        fit_params.add('period', value=N / 10, max=N / 2., min=-1.)
    else:
        fit_params.add('period', value=k, vary=False)
    fit_params.add('phase', value=0, max=np.pi, min=-np.pi)

    fit_results = minimize(residuals, fit_params, args=([X, Y],), kws={'data': image.flatten()})

    p = [fit_results.params[key].value for key in ['x0', 'y0', 'amplitude', 'period', 'phase']]
    fit = sinc(X, Y, *p)

    if show:
        # Plot the 3D figure of the fitted function and the residuals.
        # report_fit(out, show_correl=True, modelpars=fit_params)
        fit_results.params.pretty_print()
        for im in [fit, image]:
            plt.figure()
            plt.imshow(np.imag(im), vmin=-p[2], vmax=p[2])
            plt.colorbar()
        plt.figure()
        plt.imshow(np.imag(image) - np.imag(fit), vmin=-p[2], vmax=p[2])
        plt.colorbar()
        plt.show()
    return fit, p


def to_corners(center, width):
    corners = [[int(c - w // 2), int(c - w // 2) + int(w)] for c, w in zip(center, width)]
    return np.asarray(corners).astype(int)


def get_roi(image, corners):
    x = ll = 0
    y = ur = 1
    return image[corners[x][ll]:corners[x][ur], corners[y][ll]:corners[y][ur]]


def create_ref(roi_size, k=6):
    x_array = np.outer(np.arange(roi_size) - roi_size / 2, np.ones(roi_size))
    r = np.sqrt(x_array ** 2 + x_array.T ** 2)
    ref1 = np.cos(2 * np.pi * r / k) + 1j * np.sin(2 * np.pi * r / k)
    ref1 *= np.cos(np.pi * r / roi_size) ** 2
    ref1[r > roi_size / 2] = 0
    filter = np.fft.fftshift(np.exp(-(r - k / 2 * np.pi) ** 2 / 4))
    return np.conj(np.fft.fft2(ref1)) * filter


def get_position(roi, fft_ref):
    cc = np.fft.fft2(roi) * fft_ref
    cc = np.fft.fftshift(np.fft.ifft2(cc)).T
    pos = np.asarray(np.unravel_index(np.real(cc).argmax(), cc.shape)).astype(int)
    phase = np.angle(cc[pos[0], pos[1]])
    pos = pos - np.asarray(np.shape(roi)) * 0.5
    pos = np.append(np.asarray(pos).astype(float), phase)
    return cc, np.append(pos, np.max(np.abs(cc)))


path = Path(r'C:\Users\noort\Downloads\MTtransmission\3 um Bead - 20 nm')
filenames = [f for f in path.rglob('*.tif')]
roi_size = 50
i = 30
k = 5.8

beads = []
beads.append([395, 530])
beads.append([607, 668])
beads.append([280, 510])
beads = np.asarray(beads)+10

beadnr = 1
filenr = 170

# im = np.sum(Image.open(filenames[170]), axis=2)
# roi = get_roi(im, to_corners(beads[beadnr], [roi_size, roi_size]))
# plt.imshow(roi, 'gray')
# plt.show()

fft_ref = create_ref(roi_size, k=k)
track = []
for i, f in enumerate(tqdm(filenames)):
    im = np.sum(Image.open(f), axis=2)
    roi = get_roi(im, to_corners(beads[beadnr], [roi_size, roi_size]))
    cc, pos = get_position(roi, fft_ref)

    fit, p = fit_sinc(cc, center=pos[0:2], k=k)
    if i == filenr:
        print(p)
        for im in [roi, np.abs(cc), np.abs(fit)]:
            plt.figure()
            plt.imshow(im, 'gray')
        plt.show()

    track.append(p)

z = np.arange(0, 0.02 * len(filenames), 0.02)
z = np.arange(0, 1 * len(filenames), 1)

track = np.asarray(track).T
plt.plot(z, track[-1])
plt.xlabel('image')
plt.ylabel('phase (rad)')
plt.figure()
plt.plot(z, track[2])
plt.xlabel('image')
plt.ylabel('amplitude (a.u.)')
plt.figure()
plt.plot(z, track[0:2].T)
plt.xlabel('image')
plt.ylabel('x, y (pix)')

plt.show()
