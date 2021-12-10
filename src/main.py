import os
from contextlib import contextmanager
from typing import Iterator, Tuple, Union

import win32gui
import win32ui
import win32con
import numpy as np
import cv2 as cv
import py_tools
import time

from PIL import ImageGrab
from PIL import Image


def capture(window_name: str = None, size: Union[float, Tuple[int, int]] = None) -> Iterator[np.ndarray]:
    """
    Based on
    https://stackoverflow.com/a/3586280/3367753
    """
    bmp = dcObj = cDC = wDC = hwnd = None
    try:
        hwnd = window_name and win32gui.FindWindow(None, window_name)
        x0, y0, x1, y1 = win32gui.GetWindowRect(hwnd)
        w = x1 - x0
        h = y1 - y0
        if isinstance(size, float):
            size = round(w * size), round(h * size)
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        while True:
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(dcObj, w, h)
            cDC.SelectObject(bmp)
            cDC.BitBlt(
                (0, 0),
                (w, h),  # width, height
                dcObj,
                (0, 0),  # left; top
                win32con.SRCCOPY,
            )
            # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
            data = bmp.GetBitmapBits(True)
            frame = np.frombuffer(data, dtype=np.uint8)
            frame = frame.reshape([h, w, 4])
            if size:
                frame = cv.resize(frame, (size[0], size[1]), interpolation=cv.INTER_LINEAR)
            frame = frame[..., : 3]
            frame = np.ascontiguousarray(frame)
            yield frame
            if bmp:
                win32gui.DeleteObject(bmp.GetHandle())

    finally:
        # Free Resources
        if dcObj:
            dcObj.DeleteDC()
        if cDC:
            cDC.DeleteDC()
        if hwnd and wDC:
            win32gui.ReleaseDC(hwnd, wDC)
        if bmp:
            win32gui.DeleteObject(bmp.GetHandle())


def main():
    size = 2560, 1440
    try:
        t = time.perf_counter()
        for frame in capture('Dead Cells', size=.5):
            cv.imshow('CV', frame)
            t1 = time.perf_counter()
            print(1 / (t1 - t))
            t = t1
            if cv.waitKey(1) == ord('q'):
                break
            # Image.fromarray(frame).show()
    finally:
        cv.destroyAllWindows()
        print('done')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    main()
