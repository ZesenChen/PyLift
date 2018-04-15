# PyLIFT

LIFT algorithm developed in python, this is an examplar file on how the LIFT program could be used.

﻿### Coder

author: Zesen Chen

email: seu.chenzesen@gmail.com 

### Requirements

python version: 2.7

lib: numpy, mlab, libsvm, sklearn, scipy

### Description

***ratio*** - The number of clusters (i.e. k1 for positive examples, k2 for negative examples) considered for the i-th class is set to k2=k1=min(ratio*num_pi,ratio*num_ni), where num_pi and num_ni are the number of positive and negative examples for the i-th class respectively.
***The default configuration is ratio=0.1***

***svm*** - A struct variable with svm_type,which gives the kernel type, which can take the value of 1 to 3.
1) if svm_type==1,then the LIFT algorithm run with linear svm.

2) if svm_type==2,then the LIFT algorithm run with RBF kernel ,which is exp(-gamma*|x(i)-x(j)|^2)

3) if svm_type==3,then the LIFT algorithm run with Poly kernel

***The default configuration of svm_type is 1***

### Example

1、Change the basepath of your datasets and give the list of them;

2、Set the ratio parameter and svm_type;

3、Ensure that the range() correspond to your datasets;

4、Run the code in your cmd windows.

```python
path = 'F:/PyLift/dataset/'
dataset = ['CAL500','corel5k','emotions']
ratio = 0.1
svm_type = 1#Linear
for i in range(2,3):#correspond to 'emotions' dataset.
    dataset_path = path + dataset[i] + '.mat'
        tmp = sio.loadmat(dataset_path)
        data,target = tmp['data'],tmp['target'].T
        target[target==-1] = 0
        kf = KFold(n_splits=10, shuffle=True, random_state=2017)
        result = []
        #Ten fold cross validation
        for dev_index,val_index in kf.split(data):
            train_data, test_data = data[dev_index], data[val_index]
            train_target, test_target = target[dev_index], target[val_index]
            lift = LIFT(ratio,svm_type)
            lift.fit(train_data,train_target)
            result.append(lift.predict(test_data,test_target))     
        res_mean = np.mean(result,0)
        res_std = np.std(result,0)
        print('Hamming Loss:',res_mean[0],'+-',res_std[0])
        print('Ranking Loss:',res_mean[1],'+-',res_std[1])
        print('Coverage:',res_mean[2],'+-',res_std[2])
        print('One Error:',res_mean[3],'+-',res_std[3])
        print('Average Precision:',res_mean[4],'+-',res_std[4])
        print('Macro Averaging AUC:',res_mean[5],'+-',res_std[5])
```



