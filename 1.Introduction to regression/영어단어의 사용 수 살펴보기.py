from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import elice_utils
import math
import operator

def main():
    # 여기에 내용을 채우세요.
    words=read_data()
    words=sorted(words,key=lambda temp:temp[1], reverse=True)
    X=list(range(1,len(words)+1))
    Y=[x[1] for x in words]
    X,Y=np.array(X), np.array(Y)
    #X, Y = np.log(X), np.log(Y)

    slope, intercept=do_linear_regression(X,Y)
    draw_chart(X,Y,slope,intercept)
    return slope, intercept

def read_data():
    # 여기에 내용을 채우세요.
    words=[]
    f=open("words.txt",'r')
    for line in f:

        words.append([line.split(",")[0],int(line.split(",")[1])])

    return words

def draw_chart(X, Y, slope, intercept):
    fig = plt.figure()

    # 여기에 내용을 채우세요.
    ax=fig.add_subplot(111)
    plt.scatter(X,Y)

    min_X=min(X)
    max_X=max(X)
    min_Y=min_X*slope+intercept
    max_Y=max_X*slope+intercept
    plt.plot([min_X,max_X],[min_Y,max_Y],color='red',linestyle='--',linewidth=3.0)
    ax.text(min_X,min_Y+0.1,r'$y=%.2lfx+%2lf$'%(slope,intercept),fontsize=15)
    
    plt.savefig('chart.png')
    elice_utils.send_image('chart.png')

def do_linear_regression(X, Y):
    # 여기에 내용을 채우세요.

    #X=np.array(X).reshape(-1,1)
    slope, intercept, r_value, p_value, std_err=linregress(X,Y)


    return slope, intercept

if __name__ == "__main__":
    main()
