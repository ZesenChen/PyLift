import numpy as np
from mlab.releases import latest_release as mb
import evaluate as ev
import sys
from sklearn.preprocessing import scale
import scipy.io as sio
from Clustering import *
from copy import copy
from sklearn.model_selection import KFold

sys.path.append('.\libsvm-3.22\python')

from svmutil import *

class LIFT(object):
    def __init__(self, ratio, svm_type):
    #svm_type:
    #1:Linear;2:RBF;3:Polynomial
        self.ratio = ratio
        self.type = svm_type
        self.svm = []
        self.binary = []

    def fit(self, X, Y):
        train_data,train_target = copy(X),copy(Y)
        num_train, dim = train_data.shape
        tmp_value, num_class = train_target.shape
        self.P_Centers,self.N_Centers = [],[]
        for i in range(num_class):
            print('Performing clustering for the', i+1, '-th class')
            p_idx = train_target[:,i]==1
            n_idx = train_target[:,i]==0
            p_data = train_data[p_idx,:]
            n_data = train_data[n_idx,:]
            k1 = int(mb.ceil(min(sum(p_idx)*self.ratio,
                     sum(n_idx)*self.ratio)))
            k2 = k1
            if k1==0:
                label = 1 if sum(p_idx)>sum(n_idx) else 0
                self.binary.append(label)
                POS_C = []
                NEG_C = []
            else:
                self.binary.append(2)
                if p_data.shape[0] == 1:
                    POS_C = p_data
                else:
                    POS_IDX,POS_C = KMeans(p_data,k1)

                if n_data.shape[0] == 1:
                    NEG_C = n_data
                else:
                    NEG_IDX,NEG_C = KMeans(n_data,k2)
            self.P_Centers.append(POS_C)
            self.N_Centers.append(NEG_C)

        for i in range(num_class):
            print('Building classifiers: ',i+1,'/',num_class)
            if self.binary[i] != 2:
                self.svm.append([])
            else:
                centers = np.vstack((self.P_Centers[i],
                                     self.N_Centers[i]))
                num_center = centers.shape[0]
                if num_center >= 5000:
                    print('Too many cluster centers!')
                    sys.exit()
                else:
                    blocksize = 5000-num_center
                    num_block = int(mb.ceil(num_train*1.0/blocksize))
                    data = np.zeros((num_train,num_center))
                    for j in range(1,num_block):
                        low = (j-1)*blocksize
                        high = j*blocksize
                        tmp_mat = np.vstack((centers,train_data[low:high,:]))
                        Y = mb.pdist(tmp_mat)
                        Z = mb.squareform(Y)
                        data[low:high,] = Z[num_center:num_center+blocksize,0:num_center]
                        #data.append(Z[num_center:num_center+blocksize,0:num_center].tolist())
                    low = (num_block-1)*blocksize
                    high = num_train
                    tmp_mat = np.vstack((centers,train_data[low:high,:]))
                    Y = mb.pdist(tmp_mat)
                    Z = mb.squareform(Y)
                    data[low:high,] = Z[num_center:num_center+blocksize,0:num_center]
                    #data.append(Z[num_center:num_center+blocksize,0:num_center].tolist())
                training_instance = copy(data)    
                training_label = train_target[:,i]
                prob = svm_problem(training_label.tolist(), \
                                   training_instance.tolist())
                param = svm_parameter()
                if self.type == 1:
                    param.kernel_type = LINEAR
                elif self.type == 2:
                    param.kernel_type = RBF
                else:
                    param.kernel_type = POLY
                param.C = 1
                param.probability = 1
                self.svm.append(svm_train(prob,param))
                
    def predict(self, test_X, test_Y):
        test_data,test_target = copy(test_X),copy(test_Y)
        num_test, num_class = test_target.shape
        Pre_Labels = np.zeros(test_target.shape)
        Outputs = copy(Pre_Labels)
        for i in range(num_class):
            if self.binary[i] != 2:
                Pre_Labels[:,i] = self.binary[i]
                Outputs[:,i] = self.binary[i]
            else:
                centers = np.vstack((self.P_Centers[i],
                                     self.N_Centers[i]))
                num_center = centers.shape[0]
                data = np.zeros((num_test,num_center))
                if num_center >= 5000:
                    print('Too many cluster centers!')
                    sys.exit()
                else:
                    blocksize = 5000-num_center
                    num_block = int(mb.ceil(num_test*1.0/blocksize))
                    for j in range(1,num_block):
                        low = (j-1)*blocksize
                        high = j*blocksize
                        tmp_mat = np.vstack((centers,test_data[low:high,:]))
                        Y = mb.pdist(tmp_mat)
                        Z = mb.squareform(Y)
                        data[low:high,] = Z[num_center:num_center+blocksize,0:num_center]
                        #data.append(Z[num_center:num_center+blocksize,0:num_center].tolist())
                    low = (num_block-1)*blocksize
                    high = num_test
                    tmp_mat = np.vstack((centers,test_data[low:high,:]))
                    Y = mb.pdist(tmp_mat)
                    Z = mb.squareform(Y)
                    data[low:high,] = Z[num_center:num_center+blocksize,0:num_center]
                    #data.append(Z[num_center:num_center+blocksize,0:num_center].tolist())
                new_test_data = copy(data)
                Pre_Labels[:,i],tmp,tmpoutputs = svm_predict(test_target[:,i].tolist(), \
                                                             new_test_data.tolist(), \
                                                             self.svm[i],'-b 1')
                pos_index = 0 if self.svm[i].label[0]==1 else 1
                Outputs[:,i] = np.array(tmpoutputs)[:,pos_index]

        return [ev.HammingLoss(Pre_Labels,test_target),
                ev.rloss(Outputs,test_target),
                ev.Coverage(Outputs,test_target),
                ev.OneError(Outputs,test_target),
                ev.avgprec(Outputs,test_target),
                ev.MacroAveragingAUC(Outputs,test_target)]



if __name__ == '__main__':
    path = 'F:/PyLift/dataset/'
    dataset = ['birds','CAL500','corel5k','emotions',
               'enron','genbase','Image','languagelog',
               'recreation','scene','slashdot','yeast']
    ratio = 0.1
    svm_type = 1#Linear
    for i in range(3,4):
        dataset_path = path + dataset[i] + '.mat'
        tmp = sio.loadmat(dataset_path)
        data,target = tmp['data'],tmp['target'].T
        target[target==-1] = 0
        kf = KFold(n_splits=10, shuffle=True, random_state=2017)
        result = []
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
        
