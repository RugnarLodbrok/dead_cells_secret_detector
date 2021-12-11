import os
from time import time, sleep

import cv2 as cv
import pygame

from src.capturer import capture
from src.matcher import Matcher

ROOT = os.path.dirname(os.path.dirname(__file__))


class Alert:
    file = ROOT + '/data/alert.wav'

    def __init__(self):
        pygame.mixer.init()
        assert os.path.exists(self.file)
        self.sound = pygame.mixer.Sound(self.file)
        self.last_played = 0

    def play(self):
        t = time()
        if t - self.last_played > 5:
            self.last_played = t
            self.sound.play()


alert = Alert()


def main():
    found = False
    m = Matcher()
    try:
        for frame in capture('Dead Cells', size=.5):
            if m.find(frame):
                if not found:
                    alert.play()
                found = True
            else:
                found = False
            cv.imshow('CV', frame)
            if found:
                if cv.waitKey(3000) == ord('q'):
                    break
            if cv.waitKey(100) == ord('q'):
                break

    finally:
        cv.destroyAllWindows()
        print('done')


if __name__ == '__main__':
    global sound
    main()
