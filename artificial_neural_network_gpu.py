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
        self.lastlayer_identity = lastlayer_identity
        self.loss_func = loss_func
        self.network = None
        self.is_epoch_in_each_mlist = is_epoch_in_each_mlist
        if self.is_epoch_in_each_mlist:
            self.loss_val = {}
            self.train_acc, self.test_acc = {}, {}
        else:
            self.loss_val = []
            self.train_acc, self.test_acc = [], []
        self.norm_format = None
        self.Norm_input, self.Norm_output = None, None

    def __set_dataset1(self, train, test):
        m_list = list(train.keys())
        RMT = ReshapeMergerTree()

        ##Make train/test_input/output arrays.
        for m_key in m_list:
            train_input_, train_output_ = RMT.make_dataset(train[m_key], self.input_size, self.output_size)
            test_input_, test_output_ = RMT.make_dataset(test[m_key], self.input_size, self.output_size)
            if m_key == m_list[0]:
                train_input, train_output = train_input_, train_output_
                test_input, test_output = test_input_, test_output_
            else:
                train_input, train_output = np.concatenate([train_input, train_input_], axis = 0), np.concatenate([train_output, train_output_], axis = 0)
                test_input, test_output = np.concatenate([test_input, test_input_], axis = 0), np.concatenate([test_output, test_output_], axis = 0)

        ##Normalize these input/output arrays.
        ##The test-array is normalized by train-array's normalization parameters.
        Norm_input, Norm_output = Normalization(self.norm_format), Normalization(self.norm_format)
        train_input, train_output = Norm_input.run(train_input), Norm_output.run(train_output)
        test_input, test_output = Norm_input.run_predict(test_input), Norm_output.run_predict(test_output)
        self.Norm_input, self.Norm_output = Norm_input, Norm_output

        ##Masking process to prevent division by zero.
        mask = (train_output == 0.0)
        train_output[mask] += 1e-7
        mask = (test_output == 0.0)
        test_output[mask] += 1e-7

        return train_input, train_output, test_input, test_output


    def __set_dataset2(self, train, test):
        m_list = list(train.keys())
        RMT = ReshapeMergerTree()

        ##Make train/test_input/output arrays in each m_list.
        train_input_dict, train_output_dict = {}, {}
        test_input_dict, test_output_dict = {}, {}
        for m_key in m_list:
            train_input_dict[m_key], train_output_dict[m_key] = RMT.make_dataset(train[m_key], self.input_size, self.output_size)
            test_input_dict[m_key], test_output_dict[m_key] = RMT.make_dataset(test[m_key], self.input_size, self.output_size)
            if m_key == m_list[0]:
                train_input, train_output = train_input_dict[m_key], train_output_dict[m_key]
            else:
                train_input, train_output = np.concatenate([train_input, train_input_dict[m_key]], axis = 0), np.concatenate([train_output, train_output_dict[m_key]], axis = 0)

        ##Normalize these input/output arrays.
        ##The dict-array is normalized by train-array's normalization parameters.
        Norm_input, Norm_output = Normalization(self.norm_format), Normalization(self.norm_format)
        train_input, train_output = Norm_input.run(train_input), Norm_output.run(train_output)
        for m_key in m_list:
            train_input_dict[m_key], train_output_dict[m_key] = Norm_input.run_predict(train_input_dict[m_key]), Norm_output.run_predict(train_output_dict[m_key])
            test_input_dict[m_key], test_output_dict[m_key] = Norm_input.run_predict(test_input_dict[m_key]), Norm_output.run_predict(test_output_dict[m_key])
        self.Norm_input, self.Norm_output = Norm_input, Norm_output

        ##Masking process to prevent division by zero.
        mask = (train_output == 0.0)
        train_output[mask] += 1e-7
        for m_key in m_list:
            mask = (train_output_dict[m_key] == 0.0)
            train_output_dict[m_key][mask] += 1e-7
            mask = (test_output_dict[m_key] == 0.0)
            test_output_dict[m_key][mask] += 1e-7

        return train_input, train_output, train_input_dict, train_output_dict, test_input_dict, test_output_dict
        
        
    def learning(self, train, test, opt, lr, batchsize_denominator, epoch, m_list, norm_format):
        self.norm_format = norm_format
        m_list = list(train.keys())

        ##Initialize the self-variables.
        ##Make datasets.
        print("Make a train/test dataset.")
        if self.is_epoch_in_each_mlist:
            for m_key in m_list:
                self.loss_val[m_key] = []
                self.train_acc[m_key], self.test_acc[m_key] = [], []
            train_input, train_output, train_input_dict, train_output_dict, test_input_dict, test_output_dict = self.__set_dataset2(train, test)
            print("Train dataset size : {}".format(train_input.shape[0]))
        else:
            train_input, train_output, test_input, test_output = self.__set_dataset1(train, test)
            print("Train dataset size : {}\nTest dataset size : {}".format(train_input.shape[0], test_input.shape[0]))

        ##Define Machine Learning Model.
        self.network = MultiLayerNetExtend(train_input.shape[1], self.hidden, self.act_func, self.weight_init, self.batch_norm, train_output.shape[1], self.lastlayer_identity, self.loss_func)

        ##Define the optimizer.
        learning_rate = float(lr)
        optimizer = set_optimizer(opt, learning_rate)

        ##Define the number of iterations.
        rowsize_train = train_input.shape[0]
        batch_mask_arange = np.arange(rowsize_train)
        batch_size = int(rowsize_train/batchsize_denominator)
        iter_per_epoch = int(rowsize_train/batch_size)
        iter_num = iter_per_epoch * epoch
        print("Mini-batch size : {}\nIterations per 1epoch : {}\nIterations : {}".format(batch_size, iter_per_epoch, iter_num))

        ##Start learning.
        for i in range(iter_num):
            ##Make a mini batch.
            batch_mask = np.random.choice(batch_mask_arange, batch_size)
            batch_input, batch_output = cupy.asarray(train_input[batch_mask, :]), cupy.asarray(train_output[batch_mask, :])

            ##Update the self.network.params with grads.
            grads = self.network.gradient(batch_input, batch_output, is_training = True)
            params_network = self.network.params
            optimizer.update(params_network, grads)

            ##When the iteration i reaches a multiple of iter_per_epoch,
            ##Save loss_values, train/test_accuracy_value of the self.network to self.loss_val, self.train_acc, self.test_acc.
            if i % iter_per_epoch == 0:
                if self.is_epoch_in_each_mlist:
                    if i % (10 * iter_per_epoch) == 0:
                        print("{}epoch.".format(int(i / iter_per_epoch)))
                    for m_key in m_list:
                        loss_val = self.network.loss(cupy.asarray(train_input_dict[m_key]), cupy.asarray(train_output_dict[m_key]), is_training = False)
                        self.loss_val[m_key].append(loss_val)
                        train_acc = self.network.accuracy(cupy.asarray(train_input_dict[m_key]), cupy.asarray(train_output_dict[m_key]), is_training = False)
                        self.train_acc[m_key].append(train_acc)
                        test_acc = self.network.accuracy(cupy.asarray(test_input_dict[m_key]), cupy.asarray(test_output_dict[m_key]), is_training = False)
                        self.test_acc[m_key].append(test_acc)
                        if i % (10 * iter_per_epoch) == 0:
                            print("-----{}-----".format(m_key[11:16]))
                            print("loss_val : {}\ntrain_acc : {}\ntest_acc : {}".format(loss_val, train_acc, test_acc))
                else:
                    loss_val = self.network.loss(cupy.asarray(train_input), cupy.asarray(train_output), is_training = False)
                    self.loss_val.append(loss_val)
                    train_acc = self.network.accuracy(cupy.asarray(train_input), cupy.asarray(train_output), is_training = False)
                    self.train_acc.append(train_acc)
                    test_acc = self.network.accuracy(cupy.asarray(test_input), cupy.asarray(test_output), is_training = False)
                    self.test_acc.append(test_acc)
                    if i % (10 * iter_per_epoch) == 0:
                        print("{}epoch.".format(int(i / iter_per_epoch)))
                        print("loss_val : {}\ntrain_acc : {}\ntest_acc : {}".format(loss_val, train_acc, test_acc))
                    
    def predict(self, dataset, is_RMT = True):
        m_list = dataset.keys()
        prediction = {}
        if is_RMT:
            RMT = {}
        for m_key in m_list:
            if is_RMT:
                RMT[m_key] = ReshapeMergerTree()
                data_input, _ = RMT[m_key].make_dataset(dataset[m_key], self.input_size, self.output_size)
            else:
                data_input = dataset[m_key]
            data_input = self.Norm_input.run_predict(data_input)
            data_input = cupy.asarray(data_input)
            prediction[m_key] = self.network.predict(data_input, is_training = False)
            prediction[m_key] = cupy.asnumpy(prediction[m_key])
            prediction[m_key] = self.Norm_output.inv_run_predict(prediction[m_key])
            if is_RMT:
                prediction[m_key] = RMT[m_key].restore_mergertree(prediction[m_key])
        return prediction
    
    def plot_figures(self, save_dir, save_fig_type, fontsize = 26, labelsize = 15, length_major = 20, length_minor = 13, linewidth = 2.5, figsize = (8, 5)):
        if self.is_epoch_in_each_mlist:
            m_list = self.loss_val.keys()
            for m_key in m_list:
                ##Plot and save the self.loss_val.
                epochs = np.arange(len(self.loss_val[m_key]))
                fig = plt.figure(figsize = figsize)
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
                fig = plt.figure(figsize = figsize)
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
            fig = plt.figure(figsize = figsize)
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
            fig = plt.figure(figsize = figsize)
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
