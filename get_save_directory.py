def get_save_dir(LP):
    dirname = str(LP.input_size) + "in_" + str(LP.output_size) + "out/"
    dirname += LP.param_kind + "_"
    dirname += str(len(LP.hidden)) + "layers_" + str(LP.hidden[0]) + "neurons_"
    if LP.batch_normalization:
        dirname += "BatchNorm_"
    dirname += LP.loss_func + "_"
    dirname += LP.learning_rate + "lr_"
    dirname += str(LP.epoch) + "epoch_"
    if LP.normalize_format != "None":
        dirname += LP.normalize_format + "_"
    dirname += LP.extract_dataset + "_"
    dirname += str(LP.learn_num) + "/"
    return dirname
