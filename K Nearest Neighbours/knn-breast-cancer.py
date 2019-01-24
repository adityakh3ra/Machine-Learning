#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  12 23:04:14 2019

@author: adityakhera
"""

import numpy as np
from collections import Counter
import pandas as pd
import random


def nearest_neighbours(data, predict, k = 3) :
    if len(data) >= k :
        print("K value invalid")
        return
        
    distances = []
    
    for group in data :
        for features in data[group] :
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance , group])
        
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    confidence = Counter(votes).most_common(1)[0][1] / k

    #print(confidence)
        
    return vote_result
            
    
    
df = pd.read_csv("breast-cancer-wisconsin.data.txt", sep=',',  header = None,  encoding='latin-1')
df.replace('?' , -99999, inplace = True )
df.drop([0], 1 , inplace = True)

data = df.astype(float).values.tolist() 

random.shuffle(data)

test_size = 0.2
train_set = {2:[] , 4:[]}
test_set = {2:[] , 4:[]}

train_data = data[: -int(test_size*len(data))]   
test_data = data[-int(test_size*len(data)) :]   

for i in train_data :
    train_set[i[-1]].append(i[:-1])
    
for i in test_data :
    test_set[i[-1]].append(i[:-1])
    
correct , total = 0,0

# ----TESTING----

for group in test_set :
    for features in test_set[group] :
        vote = nearest_neighbours(train_set ,features, k = 5)
        if group == vote :
            correct += 1
        total += 1
        
print (correct / total)

