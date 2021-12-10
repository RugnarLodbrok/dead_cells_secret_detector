from contextlib import contextmanager
from typing import Iterator

import win32gui
import win32ui
import win32con
import numpy as np
import cv2 as cv
import py_tools
import time

from PIL import ImageGrab


def capture(windowname: str = None, size=(1920, 1080)) -> Iterator[np.ndarray]:
    w, h = size
    dataBitMap = dcObj = cDC = wDC = hwnd = None
    try:
        hwnd = windowname and win32gui.FindWindow(None, windowname)
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        while True:
            cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)
            # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
            data = dataBitMap.GetBitmapBits(True)
            frame = np.frombuffer(data, dtype=np.uint8)
            frame = frame.reshape([h, w, 4])
            yield frame

    finally:
        # Free Resources
        if dcObj:
            dcObj.DeleteDC()
        if cDC:
            cDC.DeleteDC()
        if hwnd and wDC:
            win32gui.ReleaseDC(hwnd, wDC)
        if dataBitMap:
            win32gui.DeleteObject(dataBitMap.GetHandle())


if __name__ == '__main__':
    try:
        t = time.perf_counter()
        for frame in capture():
            cv.imshow('CV', frame)
            t1 = time.perf_counter()
            print(1 / (t1 - t))
            t = t1
            if cv.waitKey(1) == ord('q'):
                break
    finally:
        cv.destroyAllWindows()
        print('done')
