import os, sys

def about_dir(save_dir, m_list, use_param, LP):
    with open(save_dir + "AboutThisDirectory.txt", mode = "w") as f:
        ##Mainbranch.
        line = "Mainbranch :"
        for m_key in m_list:
            line += " {}".format(m_key)
            if m_key != m_list[-1]:
                line += ","
        line += "\n"
        f.write(line)
        ##Parameters.
        line = "Used parameter.pickle : {}\n".format(use_param)
        f.write(line)
        line = "Predicted parameter : {}\n".format(LP.param_kind)
        f.write(line)
        line = "Add bias to dataset : {}\n".format(LP.bias)
        f.write(line)
        if LP.bias:
            line = "    The bias value : {}\n".format(LP.eps)
            f.write(line)
        ##ANN
        line = "Input size : {}, Output size : {}\n".format(LP.input_size, LP.output_size)
        f.write(line)
        line = "Hidden layers : "
        neuron_variety = []
        for h_elem in LP.hidden:
            if len(neuron_variety) == 0 or neuron_variety[-1] != h_elem:
                neuron_variety.append(h_elem)
        cnt = [1] * len(neuron_variety)
        cnt_i = 0
        for i in range(1, len(LP.hidden)):
            if LP.hidden[i-1] == LP.hidden[i]:
                cnt[cnt_i] += 1
            else:
                cnt_i += 1
        for i in range(len(neuron_variety)):
            line += str(neuron_variety[i])
            if cnt[i] != 1:
                line += "*{}".format(cnt[i])
            if i != len(neuron_variety) - 1:
                line += " + "
        line += "\n"
        f.write(line)
        line = "Batch normalization : "
        if LP.batch_normalization:
            line += "True\n"
        else:
            line += "False\n"
        f.write(line)
        line = "Batch size denominator : {}\n".format(LP.batchsize_denominator)
        f.write(line)
        line = "Activation function : {}\n".format(LP.activation_func)
        f.write(line)
        line = "Weight initialize condition : {}\n".format(LP.weight_init)
        f.write(line)
        line = "Lastlayer's activation function is identity : "
        if LP.lastlayer_identity:
            line += "True\n"
        else:
            line += "False\n"
        f.write(line)
        line = "Loss function : {}\n".format(LP.loss_func)
        f.write(line)
        line = "Optimizer : {}\n".format(LP.optimizer)
        f.write(line)
        line = "Learning rate : {}\n".format(LP.learning_rate)
        f.write(line)
        line = "Epoch : {}\n".format(LP.epoch)
        f.write(line)
        line = "Normalization format of dataset : {}\n".format(LP.normalize_format)
        f.write(line)
        line = "Extracted dataset : {}\n".format(LP.extract_dataset)
        f.write(line)
        line = "Ratio of training dataset to all dataset : {}\n".format(LP.train_ratio)
        f.write(line)
