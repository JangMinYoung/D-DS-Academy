import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import pylab as plt
import matplotlib.cm as cm
import elice_utils

def sample_data(n_samples):
    X = np.loadtxt("data/mnist2500_X.txt");
    labels = np.loadtxt("data/mnist2500_labels.txt");

    #sampling
    np.random.seed(0)
    sample_idx = np.random.choice(list(range(2500)), n_samples, replace=False)
    sampled_labels = labels[sample_idx]
    sampled_X = X[sample_idx]
    return sampled_X, sampled_labels


def plotting(Y, labels):
    # plot the results
    legend_ = []; colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in sorted(list(set(labels))):
        idxs = (labels==i).nonzero()
        l = plt.scatter(np.squeeze(Y[idxs,0]), Y[idxs,1], 20, color=colors[int(i)])
        legend_.append(l)
    plt.legend(legend_, list(range(10)), loc='center left', ncol=1, fontsize=8, bbox_to_anchor=(1, 0.5))
    plt.savefig("result.png");
    elice_utils.send_image("result.png")
    return


def pca(X, no_dims):
    # implement pca function here.
    # Don't use scikit-learn, please use numpy only.

    # for i in range (len(X.transpose())):
    #     X[:,i]=(X[:,i]-np.mean(X[:,i])) / (np.std(X[:,i]) + (1e-10))
    avg=[]
    std=[]
    X_trans=X.transpose()
    # print(X_trans)
    for i in range (0,len(X_trans)):
        avg.append(np.mean(X_trans[i]))
        std.append(np.std(X_trans[i]))
    std=np.array(std)
    avg=np.array(avg)
    std=std+(1e-10)
    for i in range(len(X_trans)):
        X_trans[i]=(X_trans[i]-avg[i])/std[i]
    X=X_trans.transpose()

    Covariance=np.cov(X.transpose())
    U, D, V=np.linalg.svd(Covariance)
    Z=[]
    for i in range (len(X)):
        X_i=[]
        for j in range(no_dims):
            X_i.append(np.dot(U[:,j].transpose(),X[i].transpose()))
        Z.append(X_i)
    print(Z)
    return Z

def main():
    # load data
    X, labels = sample_data(1000)

    # run pca
    Y = pca(X, 2)

    # plotting
    # plotting(Y, labels)

    return sys.modules


if __name__ == '__main__':
    main()
