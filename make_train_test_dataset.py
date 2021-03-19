import numpy as np


class MakeTrainTestDataset:
    ##Make and Join a Training and a Testing dataset.
    ##Run split() to split the dataset to the Training and the Testing.
    ##Run join() to join the Training and Testing into one.
    def __init__(self, m_list):
        self.m_list = m_list
        self.split_idx = None
        
    def split(self, dataset, train_ratio):
        ##Split the (normalized) dataset for Training and Testing 
        ##The percentage of the Training is defined by train_ratio.
        ##The rest of the dataset is split for Testing.
        split_idx = {}
        train, test = {}, {}
        for m_key in self.m_list:
            split_idx[m_key] = [0] * len(dataset[m_key])
            np.random.seed(1)  ##Fixed random-seed so the result of splitting is always equal.
            ##First, the rnd is defined as a series-of-number from 0 to len(dataset)-1.
            rnd = np.arange(len(dataset[m_key]))
            ##Second, the rnd is to be a sorted list of len(dataset) * train_ratio randomly extracted values from the above rnd.
            rnd = sorted(np.random.choice(rnd, size = int(len(dataset[m_key]) * train_ratio), replace = False))
            ##Third, set the element of the split_idx[m_key]'s index that is included in rnd to 1.
            for rnd_i in rnd:
                split_idx[m_key][rnd_i] = 1
            train[m_key], test[m_key] = [], []
            ##Finally, refer to the element of split_idx[m_key] as sp_idx in order from the beginning.
            for sp_idx in split_idx[m_key]:
                ##If sp_idx is 1, pop the first element of the dataset to the train(==Training).
                if sp_idx == 1:
                    train[m_key].append(dataset[m_key].pop(0))
                ##If sp_idx is 0, pop the first element of the dataset to the test(==Testing).
                if sp_idx == 0:
                    test[m_key].append(dataset[m_key].pop(0))
        self.split_idx = split_idx
        return train, test
    
    def join(self, train, test):
        ##Join the train and test in their original order.
        dataset = {}
        for m_key in self.m_list:
            dataset[m_key] = []
            for sp_idx in self.split_idx[m_key]:
                if sp_idx == 1:
                    dataset[m_key].append(train[m_key].pop(0))
                if sp_idx == 0:
                    dataset[m_key].append(test[m_key].pop(0))
        return dataset
