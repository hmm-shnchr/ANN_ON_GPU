from scipy import interpolate as ip
import numpy as np
from reshape_merger_tree import ReshapeMergerTree


def interpolate(dataset, input_size, output_size, interp_kind, RMT_flag = True):
    m_list = dataset.keys()
    ##Make input dataset.
    data_input = {}
    if RMT_flag:
        RMT = {}
        prediction = {}
        for m_key in m_list:
            RMT[m_key] = ReshapeMergerTree()
            data_input[m_key], _ = RMT[m_key].make_dataset(dataset[m_key], input_size, output_size)
    else:
        for m_key in m_list:
            data_input[m_key] = dataset[m_key]

    prediction = {}
    interp_input_step = np.concatenate([np.arange(input_size), np.arange(input_size + output_size, input_size * 2 + output_size)])
    interp_unit_size = np.arange(input_size * 2 + output_size)
    for m_key in m_list:
        interp_func = ip.interp1d(interp_input_step, data_input[m_key], kind = interp_kind)
        prediction[m_key] = interp_func(interp_unit_size)[:, input_size:input_size + output_size]
    
    if RMT_flag:
        for m_key in m_list:
            prediction[m_key] = RMT[m_key].restore_mergertree(prediction[m_key])
    
    return prediction
