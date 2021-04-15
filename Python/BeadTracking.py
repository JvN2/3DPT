import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

x = ll = 0
y = ur = 1


def to_corners(center, width):
    corners = [[int(c - w // 2), int(c - w // 2) + int(w)] for c, w in zip(center, width)]
    return np.asarray(corners).astype(int)


def clip_edges(image, corners, roi):
    if corners[x][0] < 0:
        roi = roi[-corners[x][0]:, :]
        corners[x][0] = 0
    if corners[y][0] < 0:
        roi = roi[:, -corners[y][0]:]
        corners[y][0] = 0
    image_size = np.shape(image)
    if corners[x][1] > image_size[x]:
        roi = roi[0:image_size[x] - corners[x][1]:, :]
        corners[x][1] = image_size[x]
    if corners[y][1] > image_size[y]:
        roi = roi[:, 0:image_size[y] - corners[y][1]]
        corners[y][1] = image_size[y]
    return corners, roi


def get_roi(image, corners):
    return image[corners[x][ll]:corners[x][ur], corners[y][ll]:corners[y][ur]]


def create_ref(roi_size, k = 6.7):
    x_array = np.outer(np.arange(roi_size)-roi_size/2, np.ones(roi_size))
    r = np.sqrt(x_array**2 + x_array.T**2)
    ref1 = np.cos(2*np.pi*r/k)+ 1j* np.sin(2*np.pi*r/k)
    filter = np.fft.fftshift(np.exp(-(r-k/2*np.pi)**2/4))
    return np.conj(np.fft.fft2(ref1))*filter

def get_position(roi, fft_ref):
    cc = np.fft.fft2(roi) * fft_ref
    cc = np.fft.fftshift(np.fft.ifft2(cc))
    pos = np.asarray(np.unravel_index(np.abs(cc).argmax(), cc.shape)).astype(int)
    phase = np.angle(cc[pos[0], pos[1]])
    pos = np.append(np.asarray(pos).astype(float), phase)
    return cc, np.append(pos, np.max(np.abs(cc)))

path = Path(r'C:\Users\noort\Downloads\MTtransmission\3 um Bead - 20 nm')
filenames = [f for f in path.rglob('*.tif')]
roi_size = 64
fft_ref = create_ref(roi_size, k=6.1)

filename = filenames[90]

track = []
for f in filenames:
    im = np.sum(Image.open(f), axis=2)
    roi = get_roi(im, to_corners([394, 527], [roi_size, roi_size]))
    cc, pos = get_position(roi, fft_ref)
    track.append(pos)


track = np.asarray(track).T
plt.plot(track[3])
plt.show()

