import torch 
import numpy as np
import matplotlib.pyplot as plt

data = [[1, 2], [3, 4]]
np_data = np.array(data)
x_data = torch.tensor(np_data)


print(torch.rand(55))

def TSC1_demo(A, subsequenceLength, anchor = 0):


    if subsequenceLength > len(A/4):
        raise TypeError("Error: Time series is too short relative to desired subsequence length")
    
    if subsequenceLength < 4: 
        raise TypeError('Error: Subsequence length must be at least 4')



"""
# x axis values
x = [1,2,3]
# corresponding y axis values
y = [2,4,1]

# plotting the points 
plt.plot(x, y)

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('My first graph!')

# function to show the plot
plt.show()
"""
