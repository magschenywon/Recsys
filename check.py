import numpy as np
import pandas as pd

def minkowski(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sqrt(np.sum(np.abs(index1-index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

def lorentzian(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.log(1 + np.abs(index1-index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

def canberra(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.abs(index1-index2)/(np.abs(index1)+np.abs(index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

def sorensen(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.abs(index1-index2))/np.sum(index1+index2)
            inner_count +=1
        outer_count +=1
    return sim_arr

#start from here:
def soergel(data):
    dims=len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.abs(index1-index2))/np.max((index1,index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#problem:ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
def kulczynski(data):
    dims=len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.abs(index1-index2))/np.sum(np.min((index1,index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def meancharacter(data):
    dims=len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.abs(index1-index2))/dims
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def nonintersection(data):
    dims=len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.abs(index1-index2))*(1/2)
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def jaccard(data):
    dims=len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.abs(np.square(index1-index2)))/(np.sum(np.square(index1))+np.sum(np.square(index2))-np.sum(np.abs(index1*index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def cosine(data):
    dims=len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = 1-np.sum(np.abs(index1*index2))/(np.sum(np.sqrt(index1))*np.sum(np.sqrt(index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def dice(data):
    dims=len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = 1-(2*np.sum(np.abs(index1*index2)))/(np.sum(np.square(index1))+np.sum(np.square(index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def chord(data):
    dims=len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sqrt(2-2*(np.sum(np.abs(index1*index2))/(np.sum(np.square(index1))*np.sum(np.square(index2)))))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def bhattacharyya(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = -1*np.log(np.sum(np.sqrt(index1*index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def squaredchord(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.square(np.sqrt(index1)-np.sqrt(index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def matusita(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sqrt(np.sum(np.square(np.sqrt(index1)-np.sqrt(index2))))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def hellinger(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sqrt(2*np.sum(np.square(np.sqrt(index1)-np.sqrt(index2))))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def squaredeuclidean(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.square(index1-index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def clark(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sqrt(np.sum(np.square((index1-index2)/((np.abs(index1)+(np.abs(index2)))))))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def neyman(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.square(index1-index2)/index1)
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def pearson(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.square(index1-index2)/index2)
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def SquD(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = np.sum(np.square(index1-index2)/(index1+index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked    
def ProbabilisticSymmetric(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = 2*np.sum(np.square(index1-index2)/(index1+index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def divergence(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = 2*np.sum(np.square(index1-index2)/np.square(index1+index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def additiveSymmetric(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] = 2*np.sum(((np.square(index1-index2))*(index1*index2))/(index1*index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def average(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sqrt(1/dims*np.sum(np.square(index1-index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#error
def MCED(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sqrt(np.sum(np.square(index1-index2))/np.sum(np.square))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked (mind zero countered in log)
def kullbackleibler(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum(index1*np.log(index1/index2))
            inner_count +=1
        outer_count +=1
    return sim_arr
￼
#worked (mind zero countered in log)
def jeffreys(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum((index1-index2)*np.log(index1/index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked (mind zero countered in log)
def kdd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum(index1*log((2*index1)/(index1+index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked (mind zero countered in log)
def topsoe(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum(index1*np.log((2*index1)/(index1+index2)))+np.sum(index2*np.log((2*index2)/(index1+index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked (mind zero countered in log)
def jensenshannon(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =(1/2)*(np.sum(index1*np.log((2*index1)/(index1+index2)))+np.sum(index2*np.log((2*index2)/(index1+index2))))
            inner_count +=1
        outer_count +=1
    return sim_arr

#error：ValueError: setting an array element with a sequence
def jdd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =(1/2)*(np.sum((index1*log(index1)+index2*log(index2))/2)-(index1/2+index2/2)*log(index1/2+index2/2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#problem: ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
def vwhd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum(np.abs(index1-index2)/min(index1,index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#problem: ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
def vsdf1(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum(np.square(index1-index2)/np.square(min(index1,index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#problem: ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
def vsdf2(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum(np.square(index1-index2)/min(index1,index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#problem: ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
def vsdf3(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum(np.square(index1-index2)/max(index1,index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def mscd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.max(np.sum((np.square(index1-index2)/index1)),np.sum((np.square(index1-index2)/index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def miscd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.min(np.sum((np.square(index1-index2)/index1)),np.sum((np.square(index1-index2)/index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def avgD(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =(1/2)*np.sum(np.abs(index1-index2))+max(np.abs(index1-index2))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def kjd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum(np.square(np.square(index1)+np.square(index2))/(2*(index1*index2)**(3/2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#error:ValueError: setting an array element with a sequence.
def tanD(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum((index1+index2)/2)*np.log((index1+index2)/(2*np.sqrt(index1*index2)))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def pearsond(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            x_bar =  np.mean(index1)
            y_bar = np.mean(index2)
            sim_arr[outer_count,inner_count] = 1 - (np.sum((index1-x_bar)*(index2-y_bar))/np.sqrt((np.sum(np.square(index1-x_bar)))*(np.sum(np.square(index2-y_bar)))))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def correlationd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            x_bar =  np.mean(index1)
            y_bar = np.mean(index2)
            sim_arr[outer_count,inner_count] = 0.5*(1 - (np.sum((index1-x_bar)*(index2-y_bar))/np.sqrt((np.sum(np.square(index1-x_bar)))*(np.sum(np.square(index2-y_bar))))))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def squaredpearsond(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            x_bar =  np.mean(index1)
            y_bar = np.mean(index2)
            sim_arr[outer_count,inner_count] = 1 -(np.square(np.sum((index1-x_bar)*(index2-y_bar))/np.sqrt((np.sum(np.square(index1-x_bar)))*(np.sum(np.square(index2-y_bar))))))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def hammingd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            sim_arr[outer_count,inner_count] =np.sum(np.abs(index1-index2).astype(bool))
            inner_count +=1
        outer_count +=1
    return sim_arr

#hausdroff_distance

#worked
def statisticsd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            mean=(index1+index2)/2
            sim_arr[outer_count,inner_count] = np.sum((index1-mean)/mean)
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def whittaker(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            x1=index1/np.sum(index1)
            y1=index2/np.sum(index2)
            sim_arr[outer_count,inner_count] = (1/2)*np.sum(np.abs(x1-y1))
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def motykad(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            mxs = [x if x>y else y for x,y in np.array(list(zip(index1,index2)))]
            mxs = np.array(mxs)
            sim_arr[outer_count,inner_count] = np.sum(np.max(mxs))/np.sum(index1+index2)
            inner_count +=1
        outer_count +=1
    return sim_arr

#worked
def hassanatd(data):
    dims = len(data.index)
    sim_arr = np.zeros((dims,dims))
    outer_count = 0
    for row,index1 in data.iterrows():
        inner_count = 0
        for row2,index2 in data.iterrows():
            mxs = [x if x>y else y for x,y in np.array(list(zip(index1,index2)))]
            mxs = np.array(mxs)
            if np.min((index1,index2))>=0:
                1-(1+np.min(mxs))/(1+np.max(mxs))
            else:
                1-(1+np.min(mxs)+np.abs(np.min(mxs)))/(1+np.max(mxs)+np.abs(np.max(mxs)))
        outer_count +=1
    return sim_arr
    
test_data = pd.DataFrame({'A':[4,0,2],'B':[3,2,1],'C':[2,1,3]})
print(pearsond(test_data))
~                                                                                                                                                                                                           
                                                                                                                                                                                                       
                                                                                                                                                                                          1,1           All
