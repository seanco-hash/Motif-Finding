import numpy as np
import sys
import TCM_EM

EPSILON = 0.000000000000001
MINUS_INF = np.log(EPSILON)
nucDec = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def doViterbi(seq, emissions, transitions):
    """
    Viterbi algorithm that finds the most probable assignment for the hidden states that emitted the
    observation
    :param seq: A given sequence
    :param transitions: The transition matrix - in log scale
    :param emissions: The emissions matrix - in log scale
    """
    # Initialization: v - calculation matrix, ptr - trace matrix
    k = emissions.shape[0]  # Amount of states: K
    n = len(seq)  # n

    v = np.zeros([k, n], dtype=float)
    ptr = np.zeros([k, n], dtype=int)
    for i in range(1, k):
        v[i, 0] = float('-inf')

    for i in range(1, n):
        for j in range(0, k):
            prevCol = v[:, i-1]
            maxI, maxVal = max(enumerate(prevCol + transitions[:, j]), key=lambda item: item[1])
            v[j, i] = maxVal + emissions[j, nucDec[seq[i]]]
            ptr[j, i] = maxI

    path = ['B']
    bestStateIndex = 0
    for j in range(n-1, 0, -1):
        if ptr[bestStateIndex, j] == 0:
            path.append('B')
        else:
            path.append('M')
        bestStateIndex = ptr[bestStateIndex, j]

    path.reverse()
    return path


def calculateProbTable(p, numOfStates):
    """
    Creates the transition matrix from a give p.
    :param p: transition value from s_0 to s_1
    :param numOfStates:
    :return: transition matrix, regular scale.
    """
    t = np.full([numOfStates, numOfStates], fill_value=EPSILON)
    t[0, 0] = 1-p
    t[0, 1] = p
    for i in range(1, numOfStates - 1):
        t[i, i+1] = 1
    t[numOfStates-1, 0] = 1
    return t


def printMotif(path, seq):
    """
    Prints sequance and the Viterbi path result beneath it.
    :param path: String of 'B' for background state, and 'M' for Motif states
    :param seq:
    :return:
    """
    for i in range(0, len(path), 50):
        print(''.join(seq[i:i + 50]))
        print(''.join(path[i:i + 50]))
        print()


def printViterbiVsDecoding(path, seq, decoding):
    for i in range(0, len(path), 50):
        print(''.join(seq[i:i + 50]))
        print(''.join(path[i:i + 50]))
        print(''.join(decoding[i:i + 50]))
        print()


def posteriorDecoding(f, b, seq, states):
    decoding = np.zeros([states, len(seq)])
    likelihood = f[0, -1]
    for state in range(states):
        for i in range(len(seq)):
            decoding[state, i] = f[state, i] + b[state, i] - likelihood

    maxValInicies = [np.argmax(decoding[:, j]) for j in range(len(seq))]
    path = [str(0) if maxValInicies[x] == 0 else str(1) for x in range(len(seq))]
    return path


def main():
    """
    get sequence, emissions matrix and p and runs the viterbi algorithm
    """
    emissions = np.genfromtxt(fname=sys.argv[2], delimiter="\t", skip_header=1, filling_values=1)
    emissions = np.vstack([[0.25, 0.25, 0.25, 0.25], emissions])
    statesNum = emissions.shape[0]
    s = sys.argv[1]
    p = float(sys.argv[3])
    transitions = calculateProbTable(p, statesNum)

    path = doViterbi(s, np.log(emissions), np.log(transitions))
    printMotif(path, s)
    f = TCM_EM.doForeward(s, np.log(emissions), np.log(transitions))
    b = TCM_EM.doBackWard(s, np.log(emissions), np.log(transitions))
    decoding = posteriorDecoding(f, b, s, transitions.shape[0])
    printViterbiVsDecoding(path, s, decoding)


if __name__ == '__main__':
    main()

