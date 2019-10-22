import numpy as np
import math
import pickle

data = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(data)
""" - infront of index means count from the end, not start
    : is the slice operator, means <start of slice>:<end of slice>
"""
x = data[:,-2:]     # from right count 2, then remaining to the right
print("this is x")
print(x)
y = data[:,:-2]     # from right count 2, everything up until this column
print("this is y")
print(y)
print("\n")

z = data[:,:2]      # from left count 2, everything up until this column
print("this is z")
print(z)
a = data[:,-1:]
print("this is a")  # from left count 1,
print(a)


maxSequenceLength = 1000
DS = np.empty([1, maxSequenceLength, 388])
fullList = np.load('./npsave.npy')

frames = math.ceil(fullList.shape[0] / maxSequenceLength)
remainder = fullList.shape[0] % maxSequenceLength

if frames == 1:
    if remainder == 0:
        unsqueezed = np.expand_dims(fullList, axis=0)  # add dimension
        DS = np.append(DS, unsqueezed, axis=0)  # append to DS
    else:
        # append with zeros to fill single frame
        appended = np.append(fullList,
                             np.zeros([(maxSequenceLength - remainder), 388]), axis=0)
        unsqueezed = np.expand_dims(appended, axis=0)   # add dimension
        DS = np.append(DS, unsqueezed, axis=0)  # append to DS
else:
    if remainder == 0:
        unsqueezed = np.expand_dims(fullList, axis=0)  # add dimension
        unsqueezed = unsqueezed.reshape([-1, maxSequenceLength, 388])
        # append to DS for each frame, requires expand_dims to have 3-D shape for append
        for i in range(frames):
            DS = np.append(DS, np.expand_dims(unsqueezed[i], axis=0), axis=0)
    else:
        # append with zeros to allow reshape
        appended = np.append(fullList,
                             np.zeros([(maxSequenceLength - remainder), 388]), axis=0)
        unsqueezed = np.expand_dims(appended, axis=0)   # add dimension
        unsqueezed = unsqueezed.reshape([-1, maxSequenceLength, 388])
        # append to DS for each frame, requires expand_dims to have 3-D shape for append
        for i in range(frames):
            DS = np.append(DS, np.expand_dims(unsqueezed[i], axis=0), axis=0)

DS = np.delete(DS, 0, axis=0)   # removes originial empty array (i.e. array[0, :, :])

# listtest = []
# for i in range(30):
#     listtest.append(i)
#
# with open('listtest.data','wb') as filehandle:
#     pickle.dump(listtest, filehandle)