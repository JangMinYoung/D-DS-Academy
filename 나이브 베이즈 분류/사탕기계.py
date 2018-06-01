import re
import math
import numpy as np

def main():
    M1 = {'r': 0.7, 'g': 0.2, 'b': 0.1} # M1 기계의 사탕 비율
    M2 = {'r': 0.3, 'g': 0.4, 'b': 0.3} # M2 기계의 사탕 비율

    test = {'r': 4, 'g': 3, 'b': 3}

    print(naive_bayes(M1, M2, test, 0.7, 0.3))

def naive_bayes(M1, M2, test, M1_prior, M2_prior):
    M1_test=((M1.get('r')**test.get('r'))*(M1.get('g')**test.get('g'))*(M1.get('b')**test.get('b')))*M1_prior
    M2_test=((M2.get('r')**test.get('r'))*(M2.get('g')**test.get('g'))*(M2.get('b')**test.get('b')))*M2_prior

    return [M1_test/(M1_test+M2_test), M2_test/(M1_test+M2_test)]

if __name__ == "__main__":
    main()
