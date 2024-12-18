import torch 
import numpy as np
import math
import matplotlib.pyplot as plt

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

x_data = torch.tensor(data)


def TSC1_demo(A, subsequenceLength, anchor = 0):

    exclusionZone = round(subsequenceLength/4)


    if subsequenceLength > len(A/4):
        raise TypeError("Error: Time series is too short relative to desired subsequence length")
    
    if subsequenceLength < 4: 
        raise TypeError('Error: Subsequence length must be at least 4')

    if A.dim() == 2 and A.shape[0] == 1:
        A = A.T  

    # initialization
    MatrixProfileLength = A.dim() - subsequenceLength + 1
    MPLeft = torch.ones(MatrixProfileLength,1) * torch.finfo(torch.float32).max
    MPindexLeft = torch.zeros(MatrixProfileLength, 1)
    MPRight = torch.ones(MatrixProfileLength,1) * torch.finfo(torch.float32).max
    MPindexRight = torch.zeros(MatrixProfileLength, 1)
    X, n, sumx2, sumx, meanx, sigmax2, sigmax = fastfindNNPre(A, subsequenceLength)

    pickedIdx = A[:MatrixProfileLength]; 
    dropval= 0
    distanceProfile = torch.zeros(MatrixProfileLength,1)
    lastz = torch.zeros(MatrixProfileLength,1)
    updatePosRight= torch.zeros(MatrixProfileLength, dtype=torch.bool)
    updatePosLeft= torch.zeros(MatrixProfileLength, dtype=torch.bool)

    for i in range(pickedIdx.shape[0]):
        idx = pickedIdx[i]
        subsequence = A[idx:idx+subsequenceLength-1]

        if i == 0:
            distanceProfile[:,1], lastz ,dropval ,lastsumy, lastsumy2 = fastfindNN(X, subsequence, n, subsequenceLength,sumx2, sumx, meanx, sigmax2, sigmax)
            distanceProfile[:,1] = torch.abs(distanceProfile)
            firstz=lastz
        else:
            lastsumy = lastsumy-dropval+subsequence[-1]
            lastsumy2 = lastsumy2 - dropval**2 + subsequence[-1]**2
            meany= lastsumy/subsequenceLength
            sigmay2 = lastsumy2/subsequenceLength-meany**2
            sigmay = math.sqrt(sigmay2)
            lastz[1:n-subsequenceLength]=lastz[:n-(subsequenceLength-1)]-A[:n-(subsequenceLength-1)]*dropval+A[subsequenceLength:n]*subsequence[subsequenceLength-1]
            lastz[1]=firstz[i]
            distanceProfile[:1] = math.sqrt(2*(subsequenceLength-(lastz-subsequenceLength*meanx*meany)/(sigmax*sigmay)))
            dropval=subsequence[1]

    # apply exclusion zone
    exclusionZoneStart = torch.max(1, idx-exclusionZone)
    exclusionZoneEnd = torch.min(MatrixProfileLength, idx+exclusionZone)
    distanceProfile[exclusionZoneStart:exclusionZoneEnd] = torch.finfo(torch.float32).max
    
    # figure out and store the neareest neighbor
    updatePosLeft[1:(idx-1)] = False
    updatePosLeft[idx:MatrixProfileLength] = distanceProfile[idx:MatrixProfileLength] < MPLeft[idx:MatrixProfileLength]
    MPLeft[updatePosLeft] = distanceProfile[updatePosLeft]
    MPindexLeft[updatePosLeft] = idx
    
    updatePosRight[(idx+1):MatrixProfileLength] = False
    updatePosRight[1:idx] = distanceProfile[1:idx] < MPRight[1:idx]
    MPRight[updatePosRight] = distanceProfile(updatePosRight)
    MPindexRight[updatePosRight] = idx

    if i%1000 == 0:
        None
    

    ChainPos = torch.zeros(MatrixProfileLength, dtype=torch.bool)
    ChainLength = torch.zeros(MatrixProfileLength,1)

    for i in range(MatrixProfileLength-1):
        if ChainPos[i] == False:
            cur = i
            count=1
            while MPindexRight[cur]>0 and MPindexLeft[MPindexRight[cur]]==cur:
                ChainPos[cur]= True
                cur=MPindexRight[cur]
                count += 1

            ChainLength[i] = count
    
    ChainStart=0

    if anchor == 0:
        ChainStart, _ = torch.max(ChainLength, dim=0) 
    else:
        ChainStart = anchor

    count=ChainLength[ChainStart]




def fastfindNNPre(x, m):
    n = x.shape[0]
    
    x_padded = torch.cat((x, torch.zeros(n)))

    X = torch.fft.fft(x_padded)

    cum_sumx = torch.cumsum(x_padded, dim=0)
    cum_sumx2 = torch.cumsum(x_padded**2, dim=0)

    
    sumx2 = cum_sumx2[m-1:n] - torch.cat((torch.zeros(1, dtype=cum_sumx2.dtype), cum_sumx2[:n-m]), dim=0)
    sumx = cum_sumx[m-1:n] - torch.cat((torch.zeros(1, dtype=cum_sumx2.dtype), cum_sumx[:n-m]), dim=0)
    meanx = sumx / m
    sigmax2 = (sumx2 / m) - (meanx**2)
    sigmax = torch.sqrt(sigmax2)

    return X, n, sumx2, sumx, meanx, sigmax2, sigmax

def fastfindNN(X, y, n, m, sumx2, sumx, meanx, sigmax2, sigmax):
    dropval= y[1]
    y = torch.flip(y)
    y_padded = torch.cat(y, torch.zeros(m))

    #The main trick of getting dot products in O(n log n) time
    Y = torch.fft(y)
    Z = X*Y
    z = torch.fft.ifft(Z)

    #compute y stats -- O(n)
    sumy = torch.sum(y_padded)
    sumy2 = sum(y_padded**2)
    meany=sumy/m
    sigmay2 = sumy2/m-meany**2
    sigmay = math.sqrt(sigmay2)

    dist = 2*(m-(z[m:n]-m*meanx*meany)/(sigmax*sigmay))
    dist = math.sqrt(dist)
    lastz=torch.real(z[m:n])

    return dist ,lastz ,dropval ,sumy ,sumy2
