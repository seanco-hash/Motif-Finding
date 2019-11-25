import random
import numpy as np


def simulateSingleRun(p1, p2, minTimeInState):
    curTime = minTimeInState
    x = random.uniform(0, 1)
    if x < p1:
        curTime += 1
    else:
        curTime += 2

    y = random.uniform(0, 1)
    while y < p2:
        curTime += 1
        y = random.uniform(0, 1)

    print(curTime)
    return curTime


def simulateHistogram(p1, p2, minTimeInState):
    histogram = np.zeros((300))
    iterates = 0
    while iterates < 3000:
        histogram[simulateSingleRun(p1, p2, minTimeInState)-1] += 1
        iterates += 1

    print(histogram)

simulateHistogram(0.15, 0.8, 2)