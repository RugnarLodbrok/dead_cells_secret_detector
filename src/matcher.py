import os
from typing import Tuple

import cv2 as cv
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(__file__))


class Needle:
    def __init__(self, f_name: str, channel=None, radius=None, scale: int = 2):
        self.radius = radius
        self.channel = channel
        im = np.array(Image.open(f_name))
        im = im[:, :, ::-1]  # convert to bgr
        im = self.convert_mode(im)
        h = im.shape[0]
        w = im.shape[1]
        if scale and scale != 1:
            h //= scale
            w //= scale
            self.im = cv.resize(im, (w, h), interpolation=cv.INTER_LINEAR)
        else:
            self.im = im
        self.w = w
        self.h = h

    def convert(self, im: np.ndarray) -> np.ndarray:
        im = self.convert_mode(im)
        if self.radius:
            r = self.radius
            blurred = cv.blur(im, [r, r])
            im = (im - blurred) + 127
        return im

    def convert_mode(self, im: np.ndarray) -> np.ndarray:
        if self.channel in {'red', 2}:
            im = im[:, :, 2]
        elif self.channel in {'blue', 0}:
            im = im[:, :, 0]
        elif self.channel == 'bw':
            im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        return im

    def rectangle(self, frame, x, y):
        s = 10
        p0 = (x - s, y - s)
        p1 = (x + s + self.w, y + s + self.h)
        cv.rectangle(frame, p0, p1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)


class Matcher:
    def __init__(self, threshold=.6):
        self.threshold = threshold
        self.needles = [
            Needle(f_name=ROOT + '/data/ram_bw.png', channel='red', radius=20),
            Needle(f_name=ROOT + '/data/secret_bw.png', channel='bw', radius=8),
            Needle(f_name=ROOT + '/data/rift.png', channel='blue', radius=15, scale=1),
        ]

    def locate(self, frame: np.ndarray, needle: Needle) -> Tuple[float, Tuple[int, int]]:
        frame = needle.convert(frame)
        # cv.imshow('loc', frame)
        match = cv.matchTemplate(frame, needle.im, cv.TM_CCOEFF_NORMED)
        _, max_val, _, (x, y) = cv.minMaxLoc(match)
        # print(max_val, x, y)
        return max_val, (x, y)

    def find(self, frame: np.ndarray):
        for n in self.needles:
            max_val, (x, y) = self.locate(frame, n)
            r = max_val > self.threshold
            if r:
                n.rectangle(frame, x, y)
                return True
        return False
