from itertools import groupby
from collections import Counter
import numpy as np
import sys
import Motif_viterbi
from functools import reduce
import matplotlib.pyplot as plt


EPSILON = 0.000000000000001
MINUS_INF = np.log(EPSILON)
nucDec = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def fastaread(fasta_name):
    """
    Read all sequences from a given fasta file path
    :param fasta_name: The path to the fasta file
    :return: An iterator to the (header, seq) of the sequences present in the file
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def count_words(fasta_name, k, L1):
    """
    Recognizes motifs and count their appearances at multiple given sequences
    :param fasta_name: file path
    :param k: wanted motif length
    :param L1: the wanted number of top motifs
    :return: list of tuples: ('motif', number of appearances)
    """
    cntr = Counter()
    for _, seq in fastaread(fasta_name):
        cntr.update([seq[i:i+k] for i in range(len(seq) - k + 1)])
    sortedMotifs = sorted(cntr.items(), key=lambda item: item[1], reverse=True)
    return sortedMotifs[:L1]


def calculateProbTable(motif, motifAppearance, possibleApp):
    """
    Calculates p and creates the transitions matrix and return it in log scale
    :param motif: Given single motif
    :param motifAppearance: number of the motif's appearances at multiple given sequences.
    :param possibleApp: all the possible locations that the motif can start at.
    :return: transitions matrix at log scale.
    """
    motifLen = len(motif) + 1
    p = motifAppearance / float(possibleApp)
    t = np.full([motifLen, motifLen], fill_value=EPSILON)
    t[0, 0] = 1-p
    t[0, 1] = p
    for i in range(1, motifLen - 1):
        t[i, i+1] = 1
    t[motifLen-1, 0] = 1
    return np.log(t)


def calculateEmissions(alpha, motif):
    """
    Calculates and creates the emissions matrix from a given alpha parameter: The emission for observations
     inside the motif is 1-3alpha and alpha else. The emission at the background state is 0.25 uniformly.
    :param alpha: 0 < alpha < 1
    :param motif: single motif
    :return: log scale emission matrix
    """
    n = len(motif)
    emissions = np.full([n, 4], fill_value=alpha, dtype=float)
    emissions = np.vstack([[0.25, 0.25, 0.25, 0.25], emissions])
    for i in range(n):
        emissions[i+1, nucDec[motif[i]]] = 1 - (3 * alpha)
    return np.log(emissions)


def readFasta(fileName, k):
    """
    Use the 'fastaread' function in order to execute comfortable lists of the sequences and their names and
    calculates the number of the possible locations in which motif in length of k can start.
    :param fileName: fasta file name
    :param k: length of motif
    :return: number of possible_appearances, list of names of genes, list of sequences
    """
    totPossibleApp = 0
    geneNames = []
    seqList = []
    for gene, seq in fastaread(fileName):
        totPossibleApp += len(seq) - k + 1
        geneNames.append(gene)
        seqList.append(seq)
    return totPossibleApp, geneNames, seqList


def doForeward(seq, emissions, transitions):
    """
    Implements the forward algorithm on a single sequence with given log scale emissions and transitions
    matrices
    :param seq:
    :param emissions: log scale numpy matrix STATES x 4 (nucleotides)
    :param transitions: log scale numpy matrix STATES x STATES
    :return: Log scale Dynamic programing forward table. Log-liklihood at f[0, -1]
    """
    k = emissions.shape[0]  # Amount of states: K
    n = len(seq)  # n
    f = np.full([k, n + 1], fill_value=MINUS_INF)  # represents log(epsilon) (log(0) is not defined)
    f[0, 0] = 0
    f[0, 1] = 0

    for i in range(2, n + 1):
        for j in range(k):
            pathByNow = f[:, i-1]
            pathByNow = pathByNow + transitions[:, j]
            logSum = reduce(np.logaddexp, pathByNow)
            f[j, i] = logSum + emissions[j, nucDec[seq[i - 1]]]
    return f


def doBackWard(seq, emissions, transitions):
    """
    Implements the backward algorithm on a single sequence with given log scale emissions and transitions
    matrices
    :param seq:
    :param emissions: log scale numpy matrix STATES x 4 (nucleotides)
    :param transitions: log scale numpy matrix STATES x STATES
    :return: Log scale Dynamic programing backward table.
    """
    k = transitions.shape[0]  # Amount of states
    n = len(seq)  # n
    b = np.full([k, n + 1], fill_value=MINUS_INF, dtype=float)
    b[0, -1] = 0  # Must and with B state.

    for i in range(b.shape[1] - 2, -1, -1):
        for l in range(0, b.shape[0]):
            b[l, i] = reduce(np.logaddexp, b[:, i + 1] + transitions[l, :] + emissions[:, nucDec[seq[i]]])
    return b


def history(motifs, motifLL, maxlen):
    """
    Creates the history file for the project that holds the log-liklihood list for each motif.
    :param motifs: motif list
    :param motifLL: list of the log-liklihoods received during the BW on the motifs
    :param maxlen: max Baum-Welch iterations for single sequence.
    :return: creates output file history
    """
    historyFile = open('History.tab', 'w')

    for i in range(len(motifs)):
        historyFile.write(motifs[i][0] + "\t")  # Writes the motifs names

    historyFile.write("\n")

    for i in range(maxlen):
        for j in range(len(motifLL)):
            if i < len(motifLL[j]):
                historyFile.write(str(motifLL[j][i]) + "\t")
            else:
                historyFile.write("0\t")
        historyFile.write("\n")
    historyFile.close()


def doBaumWelch(emissions, transitions, seqList, th):
    """
    Implements the Baum-Welch algorithm
    :param emissions: log scale emission numpy matrix
    :param transitions: log scale transitions numpy matrix
    :param seqList: sequences list
    :param th: wanted threshold - log-liklihood change between iterations.
    :return: log-liklihoods list (the value at each iteration), distortions list (the LL deltas) final
    emissions matrix, final transitions matrix.
    """
    LLvalues = []
    distortion = []  # LL deltas
    distortion.append(th + 1)
    curLl, prevLl = 0, 0
    iterations = 0
    while distortion[iterations] > th:
        iterations += 1
        # Fill forward and backward results of each sequence.
        forward = [doForeward(seq, emissions, transitions)
                   for seq in seqList]

        backward = [doBackWard(seq, emissions, transitions)
                    for seq in seqList]

        prevLl = curLl
        curLl = reduce(np.logaddexp, [f[0, -1] for f in forward])
        LLvalues.append(curLl)
        distortion.append(np.abs(curLl - prevLl))

        # E STEP - statistics update
        n_k_l = np.zeros([1, transitions.shape[1]])
        for state in range(emissions.shape[0]):
            n_k_l[0, state] = reduce(np.logaddexp, [forward[i][0, n - 1] + transitions[0, state] +
                              emissions[state, nucDec[seq[n-1]]] +
                              backward[i][state, n] - forward[i][0, -1]
                              for i, seq in enumerate(seqList) for n in range(1, len(seq) + 1)])

        n_k_x = np.zeros([emissions.shape[0], emissions.shape[1]])
        for state in range(1, emissions.shape[0]):
            for l in range(emissions.shape[1]):
                n_k_x[state, l] = reduce(np.logaddexp,[forward[index][state, n] + backward[index][state, n] - forward[index][0, -1]
                                                   for index, seq in enumerate(seqList)
                                                   for n in range(1, len(seq) + 1)if nucDec[seq[n - 1]] == l])

        # M STEP:
        # P update
        newPval = n_k_l[0, 1] - reduce(np.logaddexp, n_k_l[0, :])
        transitions[0, 1] = newPval
        transitions[0, 0] = np.log(1 - np.exp(newPval))
        # emissions update
        for state in range(1, emissions.shape[0]):
            for x in range(emissions.shape[1]):
                emissions[state, x] = n_k_x[state, x] - reduce(np.logaddexp, n_k_x[state, :])

    distortion[0] = distortion[1]
    return LLvalues, distortion, emissions, transitions


def drawPlot(LLList, motifToDraw):
    """
    Draws the wanted motif mean likelihood plot.
    :param LLList: motifs log likelihoods values
    :param motifToDraw: The wanted motif
    """
    plt.plot(LLList[motifToDraw])
    plt.ylabel('Mean likelihood')
    plt.xlabel("iteration")
    plt.show()


def main():
    """
    Runs the program with the command python3 TCM_EM . py trainSeq k convergenceThr L1 alpha
    gets 5 args:
    fasta file path, motif length, wanted convergence BW threshold, L1 - number of motifs to run the
    program with, alpha parameter for the emissions matrix.
    """
    fastaFile = sys.argv[1]
    motifLen = int(sys.argv[2])
    thresHold = float(sys.argv[3])
    L1 = int(sys.argv[4])
    alpha = float(sys.argv[5])

    sortedMotifs = count_words(sys.argv[1], motifLen, L1)  # Get list of the L1 most appeared
    # motifs in tuples (motif, num of appearance) sorted by value.
    possibleApp, geneNames, seqList = readFasta(fastaFile, motifLen)
    maxBWIter = 0
    motifsLL, paths = [], []

    # for each motif: (motif = (key, val)
    for i, motif in enumerate(sortedMotifs):
        # e
        emission = calculateEmissions(alpha, motif[0])
        # tau
        transitions = calculateProbTable(motif[0], motif[1],possibleApp)
        # baum welch
        LLlist, distortions, bwEmissions, bwTransitions = doBaumWelch(emission.copy(), transitions.copy(),
                                                                      seqList, thresHold)
        # append motif_iterations for historyFile
        motifsLL.append(LLlist)
        if (len(LLlist) > maxBWIter):
            maxBWIter = len(LLlist)

        # create file with emissions(transpose) and viterbi using last emissions and tau
        for j, seq in enumerate(seqList):
            path = Motif_viterbi.doViterbi(seq, bwEmissions, bwTransitions)
            indices = [str(k + 1) for k in range(len(path) - 1) if
                       (path[k] == "B" and path[k + 1] == "M")]
            if not indices:
                indices = ["0"]
            paths.append('\t'.join([geneNames[j]] + indices))

        with open('Motif{}.txt'.format(i + 1), 'w') as file:
            file.write('\n'.join(' '.join(map("{:.2f}".format, emission))
                                 for emission in np.exp(bwEmissions[1:].transpose())))
            file.write('\n')
            file.write('\n'.join(paths))

    history(sortedMotifs, motifsLL, maxBWIter)
    drawPlot(motifsLL, 0)

if __name__ == '__main__':
    main()
