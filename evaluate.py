# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:45:55 2017

@author: 14094
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:58:04 2017

@author: 14094
"""

#this code has been checked and all function are no problem
#author:zesen.chen
#email:seu_czs@126.com
#Checked on Mon Feb 27 2017

import numpy as np

def find(instance, label1, label2):
    index1 = []
    index2 = []
    for i in range(instance.shape[0]):
        if instance[i] == label1:
            index1.append(i)
        if instance[i] == label2:
            index2.append(i)
    return index1, index2

def findmax(outputs):
    Max = -float("inf")    
    index = 0
    for i in range(outputs.shape[0]):
        if outputs[i] > Max:
            Max = outputs[i]
            index = i
    return Max,index

def sort(x):
    temp = np.array(x)
    length = temp.shape[0]
    index = []
    sortX = []
    for i in range(length):
        Min = float("inf")
        Min_j = i
        for j in range(length):
            if temp[j] < Min:
                Min = temp[j]
                Min_j = j        
        sortX.append(Min)
        index.append(Min_j)
        temp[Min_j] = float("inf")
    return sortX,index

def findIndex(a, b):
    for i in range(len(b)):
        if a == b[i]:
            return i

def avgprec(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)            
            labels_index.append(index1)
            not_labels_index.append(index2)
    
    aveprec = 0
    for i in range(instance_num):
        tempvalue, index = sort(temp_outputs[i])
        indicator = np.zeros((class_num,))     
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            indicator[loc] = 1
        summary = 0
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            #print(loc)
            summary = summary + sum(indicator[loc:class_num])*1.0/(class_num-loc);
        aveprec = aveprec + summary*1.0/labels_size[i]
    return aveprec*1.0/test_data_num
            
def Coverage(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        labels_size.append(sum(test_target[i] == 1))
        index1, index2 = find(test_target[i], 1, 0)
        labels_index.append(index1)
        not_labels_index.append(index2)
    
    cover = 0
    for i in range(test_data_num):
        tempvalue,index = sort(outputs[i])
        temp_min = class_num + 1
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            if loc < temp_min:
                temp_min = loc
        cover = cover + (class_num - temp_min)
    return (cover*1.0/test_data_num - 1)*1.0/class_num
    
def HammingLoss(predict_labels, test_target):
    labels_num = predict_labels.shape[1]
    test_data_num = predict_labels.shape[0]
    hammingLoss = 0    
    for i in range(test_data_num):
        notEqualNum = 0
        for j in range(labels_num):
            if predict_labels[i][j] != test_target[i][j]:
                notEqualNum = notEqualNum + 1
        hammingLoss = hammingLoss + notEqualNum*1.0/labels_num
    hammingLoss = hammingLoss*1.0/test_data_num
    return hammingLoss        

def OneError(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    num = 0
    one_error = 0    
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            Max,index = findmax(outputs[i])
            num = num + 1
            if test_target[i][index] != 1:
                one_error = one_error + 1
    return one_error*1.0/num
    
def rloss(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)            
            labels_index.append(index1)
            not_labels_index.append(index2)
    
    rankloss = 0   
    for i in range(instance_num):
        m = labels_size[i]
        n = class_num - m
        temp = 0
        for j in range(m):
            for k in range(n):
                if temp_outputs[i][labels_index[i][j]] < temp_outputs[i][not_labels_index[i][k]]:
                    temp = temp + 1
        rankloss = rankloss + temp*1.0/(m*n)
    
    rankloss = rankloss*1.0/instance_num
    return rankloss
    
def SubsetAccuracy(predict_labels, test_target):
    test_data_num = predict_labels.shape[0]
    class_num = predict_labels.shape[1]
    correct_num = 0
    for i in range(test_data_num):
        for j in range(class_num):
            if predict_labels[i][j] != test_target[i][j]:
                break
        if j == class_num - 1:
            correct_num = correct_num + 1
            
    return correct_num*1.0/test_data_num

def MacroAveragingAUC(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    P = []
    N = []
    labels_size = []
    not_labels_size = []
    AUC = 0
    for i in range(class_num):
        P.append([])
        N.append([])
    
    for i in range(test_data_num):#得到Pk和Nk
            for j in range(class_num):
                if test_target[i][j] == 1:
                    P[j].append(i)
                else:
                    N[j].append(i)
    
    for i in range(class_num):
        labels_size.append(len(P[i]))
        not_labels_size.append(len(N[i]))
    
    for i in range(class_num):
        auc = 0
        for j in range(labels_size[i]):
            for k in range(not_labels_size[i]):
                pos = outputs[P[i][j]][i]
                neg = outputs[N[i][k]][i]
                if pos > neg:
                    auc = auc + 1
        AUC = AUC + auc*1.0/(labels_size[i]*not_labels_size[i])
    return AUC*1.0/class_num
    
