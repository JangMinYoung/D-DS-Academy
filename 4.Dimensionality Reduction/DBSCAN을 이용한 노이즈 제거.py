from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import elice_utils
import csv

def getNumNoises(filename) :
    '''
    csv 형식의 파일 filename이 주어집니다.
    이 때, Grocery와 Milk만을 고려하였을 때의 노이즈 개수를 반환하세요.
    '''
    X = []
    csvreader = csv.reader(open(filename))
    num=0
    for line in csvreader:
        if num!=0:
            X.append([line[4],line[3]])
        num+=1
    db = DBSCAN(eps=2500, min_samples=4).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    number=0
    for i in labels:
        if i==-1:
            number+=1
    return number

print(getNumNoises("data.csv"))
