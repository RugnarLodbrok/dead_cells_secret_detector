import pyautogui
import numpy as np
import cv2 as cv
import py_tools
import time

from PIL import ImageGrab


def grab(windowname=None):
    import win32gui
    import win32ui
    import win32con

    w = 1920  # set this
    h = 1080  # set this
    bmpfilenamename = "out.bmp"  # set this

    hwnd = windowname and win32gui.FindWindow(None, windowname)
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)
    # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
    data = dataBitMap.GetBitmapBits(True)

    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return data


if __name__ == '__main__':
    try:
        t = time.perf_counter()
        while 1:
            # screenshot = pyautogui.screenshot()
            # screenshot = ImageGrab.grab()
            # screenshot = np.array(screenshot)[:, :, ::-1]
            data = grab()
            screenshot = np.frombuffer(data, dtype=np.uint8).reshape([1080, 1920, 4])
            cv.imshow('CV', screenshot)
            t1 = time.perf_counter()
            print(1 / (t1 - t))
            t = t1
            if cv.waitKey(1) == ord('q'):
                break
    finally:
        cv.destroyAllWindows()
        print('done')
