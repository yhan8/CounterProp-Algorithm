import numpy as np
import matplotlib.pyplot as plt
import copy
import operator
from scipy.spatial import distance
import math
from array import array
import random
import seaborn as sns



l=[]
d=[]
new_inputs=[]
loss = []

### Training ###

#Initialize inputs and targeted outputs
inputs=[[0.227, 0.289],[0.182, 0.789],[0.613, 0.553], [0.682, 0.842], [0.72, 0.342], [0.182, 0.684]]
targets=[[0,0,0,0,1],[0,0,1,0,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,0,1,0],[1,0,0,0,0]]

def l_2(inputs):
    d = []
    l = []
    inputs = np.asarray(inputs)
    
    #l2 normalization
    for i in range(0, len(inputs)):
        l.append(math.sqrt(inputs[i][0]**2 + inputs[i][1]**2))
    
    N_index = np.argmax(l)
    N = l[N_index] + 0.01
    
    for i in range(0,len(l)):
        d.append(N**2 - l[i]**2)
        
    d = np.reshape(d,(len(d),1))
    inputs = np.append(inputs,d,1)
    inputs = inputs/N
    
    return inputs


new_inputs = l_2(inputs)
    
#Initialize weights from i to j with three inputs and 25 hidden nodes
weight_inputs=np.random.uniform(low=0, high=1, size=(3,10) )

#Initialize weights from j to k with 25 hidden nodes and 5 outputs
weight_outputs=np.random.uniform(low=0, high=1, size=(10,5) )  
loss =[]

for epoch in range(600):
    predictions = []
    current_loss = 0
    for i in range(0, len(new_inputs)):
        #winner takes all
        current_input = new_inputs[i]
        current_input = np.reshape(current_input,(1,3))
        hidden_inputs=np.dot(current_input,weight_inputs)
        output = copy.deepcopy(hidden_inputs)
        w_ij_win_index=np.argmax(hidden_inputs)
        hidden_inputs = np.zeros((1,10))
        hidden_inputs[0,w_ij_win_index] = 1
        
        #define predicted output
        predicted_output=np.dot(hidden_inputs,weight_outputs) 
        predictions.append(predicted_output)
        pred = np.argmax(predicted_output)
        
        ##update weights
        current_input = np.reshape(current_input,(1,3))
        
        update = 0.1 * (current_input - weight_inputs[:,w_ij_win_index])
        update = np.reshape(update,(3,))
        weight_inputs[:,w_ij_win_index] +=  update
        
        update = 0.1*  (targets[i] - predicted_output)
        update = np.reshape(update,(5,))
        weight_outputs[w_ij_win_index,:] += update
        
    #calculate loss 
    predictions = np.asarray(predictions)
    predictions = np.reshape(predictions,(6,5))
    
    targets_array = np.asarray(targets)
    targets_array = np.reshape(targets_array, (6,5))
    
    for i in range(0,len(predictions)):
        current_loss += sum((np.asarray(predictions[i]) - np.asarray(targets[i]))**2)
    loss.append(current_loss)
    
#plot loss function    
plt.plot(np.arange(len(loss)),loss)
plt.show()


### Print Predicted Output Using Trained Weights ###
def predict(x):
    predictions = []
    
    for i in range(0,len(x)):
        current_input = x[i]
        current_input = np.reshape(current_input,(1,3))
        hidden_inputs=np.dot(current_input,weight_inputs)
        output = copy.deepcopy(hidden_inputs)
        w_ij_win_index=np.argmax(hidden_inputs)
        hidden_inputs = np.zeros((1,10))
        hidden_inputs[0,w_ij_win_index] = 1
        predicted_output=np.dot(hidden_inputs,weight_outputs) 
        predictions.append(predicted_output)
    return predictions

pred = predict(new_inputs)
print(np.round(pred,2))

### Heat Map ###       
   
#Initialize inputs and targeted outputs
inputs=[[0.227, 0.289],[0.182, 0.789],[0.613, 0.553], [0.682, 0.842], [0.72, 0.342], [0.182, 0.684]]
targets=[[0,0,0,0,1],[0,0,1,0,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,0,1,0],[1,0,0,0,0]]

raw_data = np.asarray(inputs)
colors = []
for i in range(0,len(targets)):
    colors.append(np.argmax(targets[i]))
    
for i in range(0,len(raw_data)):
     plt.text(raw_data[i,0]*10,raw_data[i,1]*10,colors[i],ha="center", va="center",fontsize=20,color='r')
   
x = np.linspace(0,1,11)
matrix = []
cords_temp = []

for i in range(0,len(x)):
    row = []
    row_cords =[]
    for j in range(0,len(x)):
        cords_temp.append([x[i],x[j]])

cords_temp = l_2(cords_temp)
cords = []
counter = 0
matrix = []

for i in range(0,len(x)):
    row_cords = []
    matrix_temp = []
    for j in range(0,len(x)):
        row_cords.append(cords_temp[counter])
        temp = np.reshape(cords_temp[counter],(1,3))
        pred = predict([temp])
        value = np.argmax(pred[0])
        matrix_temp.append(value)
        counter +=1
    matrix.append(matrix_temp)
    cords.append(row_cords)

matrix = np.asarray(matrix)
cords= np.asarray(cords)
plt.imshow(matrix.T)
plt.show()





