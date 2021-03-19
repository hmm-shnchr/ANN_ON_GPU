from multilayer_extend_gpu import MultiLayerNetExtend
from reshape_merger_tree import ReshapeMergerTree
from normalization import Normalization
from optimizer_gpu import set_optimizer
import matplotlib.pyplot as plt
import numpy as np
import cupy
import copy as cp
import os, sys, shutil


class ArtificialNeuralNetwork:
    def __init__(self, input_size, hidden, act_func, weight_init, batch_norm, output_size, lastlayer_identity, loss_func, is_epoch_in_each_mlist = False):
        self.input_size = input_size
        self.hidden, self.act_func, self.weight_init = hidden, act_func, weight_init
        self.batch_norm = batch_norm
        self.output_size = output_size
        self.loss_func = loss_func
        self.network = MultiLayerNetExtend(input_size*2, hidden, act_func, weight_init, batch_norm, output_size, lastlayer_identity, loss_func)
        self.is_epoch_in_each_mlist = is_epoch_in_each_mlist
        if self.is_epoch_in_each_mlist:
            self.loss_val = {}
            self.train_acc, self.test_acc = {}, {}
        else:
            self.loss_val = []
            self.train_acc, self.test_acc = [], []
        self.Norm_input_, self.Norm_output_ = None, None
        
    def learning(self, train, test, opt, lr, batchsize_denominator, epoch, m_list, norm_format):
        ##Initialize the self-variables.
        if self.is_epoch_in_each_mlist:
            for m_key in m_list:
                self.loss_val[m_key] = []
                self.train_acc[m_key], self.test_acc[m_key] = [], []
        ##Make input/output dataset.
        RMT_train, RMT_test = {}, {}
        train_input, train_output = {}, {}
        test_input, test_output = {}, {}
        train_input_, train_output_ = None, None
        test_input_, test_output_ = None, None

        if self.is_epoch_in_each_mlist:
            Norm_train_input, Norm_train_output = {}, {}
            Norm_test_input, Norm_test_output = {}, {}
        for m_key in m_list:
            RMT_train[m_key] = ReshapeMergerTree()
            RMT_test[m_key] = ReshapeMergerTree()
            train_input[m_key], train_output[m_key] = RMT_train[m_key].make_dataset(train[m_key], self.input_size, self.output_size)
            test_input[m_key], test_output[m_key] = RMT_test[m_key].make_dataset(test[m_key], self.input_size, self.output_size)
            if train_input_ is None:
                train_input_, train_output_ = cp.deepcopy(train_input[m_key]), cp.deepcopy(train_output[m_key])
                test_input_, test_output_ = cp.deepcopy(test_input[m_key]), cp.deepcopy(test_output[m_key])
            else:
                train_input_, train_output_ = np.concatenate([train_input_, train_input[m_key]], axis = 0), np.concatenate([train_output_, train_output[m_key]], axis = 0)
                test_input_, test_output_ = np.concatenate([test_input_, test_input[m_key]], axis = 0), np.concatenate([test_output_, test_output[m_key]], axis = 0)
            if self.is_epoch_in_each_mlist:
                Norm_train_input[m_key], Norm_train_output[m_key] = Normalization(norm_format), Normalization(norm_format)
                Norm_test_output[m_key], Norm_test_output[m_key] = Normalization(norm_format), Normalization(norm_format)
                train_input[m_key] = Norm_train_input[m_key].run(train_input[m_key])
                train_output[m_key] = Norm_train_output[m_key].run(train_output[m_key])
                test_input[m_key] = Norm_test_input[m_key].run(test_input[m_key])
                test_output[m_key] = Norm_test_input[m_key].run(test_output[m_key])
                train_mask = (train_output[m_key] == 0.0)
                test_mask = (test_output[m_key] == 0.0)
                train_output[m_key][train_mask] += 1e-7
                test_output[m_key][test_mask] += 1e-7
        Norm_train_input_, Norm_train_output_ = Normalization(norm_format), Normalization(norm_format)
        Norm_test_input_, Norm_test_output_ = Normalization(norm_format), Normalization(norm_format)
        train_input_ = Norm_train_input_.run(train_input_)
        train_output_ = Norm_train_output_.run(train_output_)
        test_input_ = Norm_test_input_.run(test_input_)
        test_output_ = Norm_test_output_.run(test_output_)
        self.Norm_input_ = Norm_train_input_
        self.Norm_output_ = Norm_train_output_
        train_mask = (train_output_ == 0.0)
        test_mask = (test_output_ == 0.0)
        train_output_[train_mask] += 1e-7
        test_output_[test_mask] += 1e-7
        print("Make a train/test dataset.")
        print("Train dataset size : {}\nTest dataset size : {}".format(train_input_.shape[0], test_input_.shape[0]))
        ##Define the optimizer.
        learning_rate = float(lr)
        optimizer = set_optimizer(opt, learning_rate)
        ##Define the number of iterations.
        rowsize_train = train_input_.shape[0]
        batch_mask_arange = np.arange(rowsize_train)
        batch_size = int(rowsize_train/batchsize_denominator)
        iter_per_epoch = int(rowsize_train/batch_size)
        iter_num = iter_per_epoch * epoch
        print("Mini-batch size : {}\nIterations per 1epoch : {}\nIterations : {}".format(batch_size, iter_per_epoch, iter_num))
        ##Start learning.
        for i in range(iter_num):
            ##Make a mini batch.
            batch_mask = np.random.choice(batch_mask_arange, batch_size)
            batch_input, batch_output = cupy.asarray(train_input_[batch_mask, :]), cupy.asarray(train_output_[batch_mask, :])
            ##Update the self.network.params with grads.
            grads = self.network.gradient(batch_input, batch_output, is_training = True)
            params_network = self.network.params
            optimizer.update(params_network, grads)
            ##When the iteration i reaches a multiple of iter_per_epoch,
            ##Save loss_values, train/test_accuracy_value of the self.network to self.loss_val, self.train_acc, self.test_acc.
            if i % iter_per_epoch == 0:
                if self.is_epoch_in_each_mlist:
                    for m_key in m_list:
                        loss_val = self.network.loss(cupy.asarray(train_input[m_key]), cupy.asarray(train_output[m_key]), is_training = False)
                        self.loss_val[m_key].append(loss_val)
                        train_acc = self.network.accuracy(cupy.asarray(train_input[m_key]), cupy.asarray(train_output[m_key]), is_training = False)
                        self.train_acc[m_key].append(train_acc)
                        test_acc = self.network.accuracy(cupy.asarray(test_input[m_key]), cupy.asarray(test_output[m_key]), is_training = False)
                        self.test_acc[m_key].append(test_acc)
                else:
                    loss_val = self.network.loss(cupy.asarray(train_input_), cupy.asarray(train_output_), is_training = False)
                    self.loss_val.append(loss_val)
                    train_acc = self.network.accuracy(cupy.asarray(train_input_), cupy.asarray(train_output_), is_training = False)
                    self.train_acc.append(train_acc)
                    test_acc = self.network.accuracy(cupy.asarray(test_input_), cupy.asarray(test_output_), is_training = False)
                    self.test_acc.append(test_acc)
                    if i % (10 * iter_per_epoch) == 0:
                        print("{}epoch.".format(int(i / iter_per_epoch)))
                        print("loss_val : {}\ntrain_acc : {}\ntest_acc : {}".format(loss_val, train_acc, test_acc))
                    
    def predict(self, dataset, RMT_flag = True):
        m_list = dataset.keys()
        prediction = {}
        data_input = {}
        if RMT_flag:
            RMT = {}
        for m_key in m_list:
            if RMT_flag:
                RMT[m_key] = ReshapeMergerTree()
                data_input[m_key], _ = RMT[m_key].make_dataset(dataset[m_key], self.input_size, self.output_size)
            else:
                data_input[m_key] = dataset[m_key]
            data_input[m_key] = self.Norm_input_.run_predict(data_input[m_key])
            data_input[m_key] = cupy.asarray(data_input[m_key])
            prediction[m_key] = self.network.predict(data_input[m_key], is_training = False)
            prediction[m_key] = cupy.asnumpy(prediction[m_key])
            prediction[m_key] = self.Norm_output_.inv_run_predict(prediction[m_key])
            if RMT_flag:
                prediction[m_key] = RMT[m_key].restore_mergertree(prediction[m_key])
        return prediction
    
    def plot_figures(self, save_dir, save_fig_type):
        fontsize = 26
        labelsize = 15
        length_major = 20
        length_minor = 15
        linewidth = 2.5
        if self.is_epoch_in_each_mlist:
            m_list = self.loss_val.keys()
            for m_key in m_list:
                ##Plot and save the self.loss_val.
                epochs = np.arange(len(self.loss_val[m_key]))
                fig = plt.figure(figsize = (8, 5))
                ax_loss = fig.add_subplot(111)
                ax_loss.plot(epochs, self.loss_val[m_key], label = "Loss Function", color = "red", linewidth = linewidth)
                ax_loss.set_yscale("log")
                ax_loss.set_xlabel("Epoch", fontsize = fontsize)
                ax_loss.set_ylabel("Loss Function", fontsize = fontsize)
                ax_loss.legend(loc = "best", fontsize = int(fontsize * 0.6))
                ax_loss.tick_params(labelsize = labelsize, length = length_major, direction = "in", which = "major")
                ax_loss.tick_params(labelsize = labelsize, length = length_minor, direction = "in", which = "minor")
                plt.title("Loss Function({})".format(m_key[11:16]), fontsize = fontsize)
                plt.tight_layout()
                plt.savefig("{}fig_loss_{}{}".format(save_dir, m_key[11:16], save_fig_type))
                np.savetxt("{}data_loss_{}.csv".format(save_dir, m_key[11:16]), self.loss_val[m_key], delimiter = ",")
                ##Plot and save the self.train/test_acc.
                epochs = np.arange(len(self.train_acc[m_key]))
                fig = plt.figure(figsize = (8, 5))
                ax_acc = fig.add_subplot(111)
                ax_acc.plot(epochs, self.train_acc[m_key], label = "Training", color = "orange", linewidth = linewidth)
                ax_acc.plot(epochs, self.test_acc[m_key], label = "Testing", color = "blue", linewidth = linewidth)
                ax_acc.set_yscale("log")
                ax_acc.set_xlabel("Epoch", fontsize = fontsize)
                ax_acc.set_ylabel("Accuracy", fontsize = fontsize)
                ax_acc.legend(loc = "best", fontsize = int(fontsize * 0.6))
                ax_acc.tick_params(labelsize = labelsize, length = length_major, direction = "in", which = "major")
                ax_acc.tick_params(labelsize = labelsize, length = length_minor, direction = "in", which = "minor")
                plt.title("Training and Testing Accuracy({})".format(m_key[11:16]), fontsize = fontsize)
                plt.tight_layout()
                plt.savefig("{}fig_acc_{}{}".format(save_dir, m_key[11:16], save_fig_type))
                np.savetxt("{}data_acc_train_{}.csv".format(save_dir, m_key[11:16]), self.train_acc[m_key], delimiter = ",")
                np.savetxt("{}data_acc_test_{}.csv".format(save_dir, m_key[11:16]), self.test_acc[m_key], delimiter = ",")
        else:
            ##Plot and save the self.loss_val.
            epochs = np.arange(len(self.loss_val))
            fig = plt.figure(figsize = (8, 5))
            ax_loss = fig.add_subplot(111)
            ax_loss.plot(epochs, self.loss_val, label = "Loss Function", color = "red", linewidth = linewidth)
            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Epoch", fontsize = fontsize)
            ax_loss.set_ylabel("Loss Function", fontsize = fontsize)
            ax_loss.legend(loc = "best", fontsize = int(fontsize * 0.6))
            ax_loss.tick_params(labelsize = labelsize, length = length_major, direction = "in", which = "major")
            ax_loss.tick_params(labelsize = labelsize, length = length_minor, direction = "in", which = "minor")
            plt.title("Loss Function", fontsize = fontsize)
            plt.tight_layout()
            plt.savefig("{}fig_loss{}".format(save_dir, save_fig_type))
            np.savetxt("{}data_loss.csv".format(save_dir), self.loss_val, delimiter = ",")
            ##Plot and save the self.train/test_acc.
            epochs = np.arange(len(self.train_acc))
            fig = plt.figure(figsize = (8, 5))
            ax_acc = fig.add_subplot(111)
            ax_acc.plot(epochs, self.train_acc, label = "Training", color = "orange", linewidth = linewidth)
            ax_acc.plot(epochs, self.test_acc, label = "Testing", color = "blue", linewidth = linewidth)
            ax_acc.set_yscale("log")
            ax_acc.set_xlabel("Epoch", fontsize = fontsize)
            ax_acc.set_ylabel("Accuracy", fontsize = fontsize)
            ax_acc.legend(loc = "best", fontsize = int(fontsize * 0.6))
            ax_acc.tick_params(labelsize = labelsize, length = length_major, direction = "in", which = "major")
            ax_acc.tick_params(labelsize = labelsize, length = length_minor, direction = "in", which = "minor")
            plt.title("Training and Testing Accuracy", fontsize = fontsize)
            plt.tight_layout()
            plt.savefig("{}fig_acc{}".format(save_dir, save_fig_type))
            np.savetxt("{}data_acc_train.csv".format(save_dir), self.train_acc, delimiter = ",")
            np.savetxt("{}data_acc_test.csv".format(save_dir), self.test_acc, delimiter = ",")    
