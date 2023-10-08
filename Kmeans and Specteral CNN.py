import numpy as np
import matplotlib.pyplot as plt
plt.set_cmap('jet')
X=np.load('FFT_2361/NEW_FBH_2361.npy')
X_Ref=np.load('FFT_2361/NEW_FBH_2361_ref.npy')
X=np.reshape(X, (85*51,2361))
X_Ref=np.reshape(X_Ref,(85*51,2361))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#FIR=np.divide(X, X_Ref)



# In[1]:
# Elbow Method
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 6), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Clustring 
kmeans=[]
y_kmeans=[]
for i in range(1,6):
    kmeans.append(KMeans(n_clusters = i+1, init = 'k-means++', random_state = 42))
    y_kmeans.append(kmeans[-1].fit_predict(X))



# Visualization the k-means results
plt.imshow(np.reshape(y_kmeans[4],(85,51)))
plt.show()


# # saving the y_kmeans
# data = np.array(y_kmeans)
# np.save("y_kmeans_20_cluster", y_kmeans)

# In[2]:

# def FBD(f1,f2):      #This function compute Frequency Band Data. The maximum of F2 is 2360. 
#     FBDi=np.zeros([85,51])
#     F_Res=125
#     for i in range(85):
#         for j in range(51):
#     #         if i>=17 and j>=42 :
#     #             FBDi[i,j]=0
#     #         else:
#             FBDi[i,j]=( np.sum(np.reshape(X,(85,51,2361))[i,j,f1:f2+1]/np.reshape(X_Ref,(85,51,2361))[i,j,f1:f2+1]) ) * (F_Res/(f2-f1))
#     return FBDi


def FBD(f1,f2):     
    FBDi=np.zeros([85,51])
    F_Res=125
    for i in range(85):
        for j in range(51):
            FBDi[i,j]=( np.sum(X[i,j,f1:f2+1]/X_Ref[i,j,f1:f2+1]) ) * (F_Res/(f2-f1))
    return FBDi

FBD_result=FBD(0,2360)
plt.imshow(FBD_result)
plt.show()


y=FBD_result>=0.44
y=1*y
plt.imshow(y)


y=T>=0.28
y=1*y
plt.imshow(y)

# In[3]:
# HC clustering 
# Feature Scaling


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(FIR, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
plt.imshow(y_hc)

y_hc=np.reshape(y_hc, (85,51))
y_hc[0:55,25:51]=1
plt.imshow(y_hc)
Defect=np.argwhere(np.reshape(y_hc,(85*51,))==0)


# plt.imshow(np.reshape(y_hc,(85,51)))
# Defect=np.argwhere(y_hc==0)
# print(Defect.shape)
# z=y_hc==0
# # z=1*z
# # plt.imshow(z)
# arr[z]=1.
# np.save('Train_y',z)


# In[4]

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

plt.plot(range(1, 2361), explained_variance[0:2360])
plt.title('PCA')
plt.xlabel('n_components')
plt.ylabel('variance_ratio')
plt.show()

kmeans=KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans=kmeans.fit_predict(X_train)

plt.imshow(np.reshape(y_kmeans,(85,51)))

# In[5]


import numpy as np
import matplotlib.pyplot as plt

Targets=np.load("Whole_Testset.npy")
X=np.load('FFT_2361/NEW_FBH_2361.npy')
X=X.reshape(1,85,51,2361)

plt.imshow(Targets)

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    
    ### START CODE HERE ### (≈ 1 line)
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))
    ### END CODE HERE ###
    
    return X_pad

# GRADED FUNCTION: conv_forward

def conv_forward(A_prev,stride,pad,f,n_c):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    Z=[]
    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev) = A_prev.shape[0],A_prev.shape[1],A_prev.shape[2]
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f) = f,f
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = stride
    pad = pad
    n_c=n_c
    # Compute the dimensions of the CONV output volume using the formula given above. 
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    n_H = int((n_H_prev-f+2*pad)/(stride))+1
    n_W =  int((n_W_prev-f+2*pad)/(stride))+1
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev,pad)
    
    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i,:,:,:]               # Select ith training example's padded activation
        for h in range(n_H):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = h*stride
            vert_end = h*stride+f
            
            for w in range(n_W):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice" (≈2 lines)
                horiz_start = w*stride
                horiz_end = w*stride+f
                
                for c in range(n_c):   # loop over channels (= #filters) of the output volume
                                        
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    Z.append(a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:])
                    
    return Z

Data=conv_forward(A_prev=X,stride=1,pad=3,f=7,n_c=1) # the whole dataset
print(Data[0].shape)
print(len(Data))

Data_5D=np.zeros([4335,7,7,2361])
for i in range(len(Data)):
    Data_5D[i,:,:,:]=Data[i][:,:,:]
Data_5D=Data_5D.reshape(4335,7,7,2361,1)
Data_5D.shape
Data_5D=np.reshape(Data_5D,(4335,7,7,2361))
Data_5D[2001,3,3,123,0]==X.reshape(85*51,2361)[2001,123] #Checking

np.save('Whole_Inputs',Data_5D)

# In[6] 
import numpy as np


Targets=np.load("Whole_Testset.npy")
Inputs=np.load('Whole_Inputs.npy')

Defect_index= np.argwhere(Targets == 1)
Train_x = np.zeros([68,7,7,2361])
for i in range (Train_x.shape[0]):
    x=Defect_index[i,0]
    Train_x[i]=Inputs[x]
# Train_x_defect = Train_x 
# Train_x[0,3,3,200] == Inputs[271,3,3,200] checking 
np.save("Inputs_Defects",Train_x_defect)


nonDefect_index= np.argwhere(Targets == 0)
Train_x = np.zeros([3339,7,7,2361])
for i in range (Train_x.shape[0]):
    x=nonDefect_index[i,0]
    Train_x[i]=Inputs[x]
Train_x_nondefect = Train_x
np.save("Inputs_nonDefects",Train_x_nondefect) 


I_index= np.argwhere(Targets == 2)
Train_x = np.zeros([928,7,7,2361])
for i in range (Train_x.shape[0]):
    x=nonDefect_index[i,0]
    Train_x[i]=Inputs[x]
Train_x_I = Train_x
np.save("Inputs_I",Train_x_I) 


# In[7]
import pandas as pd
Inputs_ndft=np.load('Inputs_nonDefects.npy')
train_y=np.zeros([3339,])
train_y=pd.Series(train_y)
majority_class_indices=train_y[train_y.values== 0].index
np.random.seed(37)
random_majority_indices=np.random.choice(majority_class_indices,68,replace=False)
print(len(random_majority_indices))  


Train_x = np.zeros([68,7,7,2361])
for i in range (Train_x.shape[0]):
    x=random_majority_indices[i,]
    Train_x[i]=Inputs_ndft[x]
under_sample_Inputs_ndft=Train_x
np.save("Inputs_nondefects_undersample",under_sample_Inputs_ndft) 


# In[8]

# from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
# from keras.layers import Dropout, Input, BatchNormalization,LeakyReLU
# from sklearn.metrics import confusion_matrix, accuracy_score
# from keras.losses import categorical_crossentropy,binary_crossentropy
# from keras.optimizers import Adadelta,Adam,RMSprop
# from keras.models import Model
# from keras.activations import relu
# import keras
# import scipy.io as spio
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from keras.models import load_model
# import keras_metrics as km 



import numpy as np
from keras import models
from keras import layers
import keras_metrics as km

nondfct = np.load('Inputs_nondefects_undersample.npy')
dfct = np.load('Inputs_Defects.npy')
Data = np.load('Whole_Inputs.npy')


X= np.concatenate((nondfct,dfct), axis=0)
Y= np.concatenate(((np.zeros((68,),dtype=np.int64)),(np.ones((68,),dtype=np.int64))), axis = 0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(7,7,2361)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy', km.binary_f1_score()])
model.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=50, batch_size=4)

c=model.predict(Data)

c=np.reshape(c,(85,51))

import matplotlib.pyplot as plt

plt.imshow(c)

model.save('CNN.h5') 
np.save('Result_CNN.npy',c)


