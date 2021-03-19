import numpy as np

class ReshapeMergerTree:

    def __init__(self):
        self.array_input = None
        self.array_output = None
        self.surplus_list = []
        self.rowsize_list = []
        self.skiprow_dict = {}
        self.input_size = None

    def make_dataset(self, data, input_size, output_size):
        self.input_size = input_size
        unity_size = input_size+output_size

        for idx in range(len(data)):

            if len(data[idx]) < unity_size+input_size:
                self.skiprow_dict[idx] = data[idx]
                continue

            all_size = int(len(data[idx])/unity_size)
            use_size = all_size*unity_size+input_size
            if len(data[idx])-use_size < 0:
                use_size -= unity_size

            ext_data = data[idx][-use_size:]
            self.surplus_list.append(data[idx][:-use_size])
            rowsize = int((use_size-input_size)/unity_size)
            if len(self.rowsize_list) == 0:
                self.rowsize_list.append(rowsize)
            else:
                self.rowsize_list.append(self.rowsize_list[-1]+rowsize)

            colsize_input = 2*input_size
            colsize_output = output_size
            array_input = np.empty((rowsize, colsize_input))
            array_output = np.empty((rowsize, colsize_output))
            col = 0
            for row in range(rowsize):
                array_input[row, 0:input_size] = ext_data[col:col+input_size]
                array_output[row, 0:output_size] = ext_data[col+input_size:col+input_size+output_size]
                array_input[row, input_size:2*input_size] = ext_data[col+input_size+output_size:col+2*input_size+output_size]
                col += unity_size
            if self.array_input is None:
                self.array_input = array_input
                self.array_output = array_output
            else:
                self.array_input = np.concatenate([self.array_input, array_input], axis = 0)
                self.array_output = np.concatenate([self.array_output, array_output], axis = 0)

        return self.array_input, self.array_output

    def restore_mergertree(self, array_output):
        restore_list = []
        for idx in range(len(self.surplus_list)):
            if idx == 0:
                restore_input = self.array_input[0:self.rowsize_list[idx]]
                restore_output = array_output[0:self.rowsize_list[idx]]
            else:
                restore_input = self.array_input[self.rowsize_list[idx-1]:self.rowsize_list[idx]]
                restore_output = array_output[self.rowsize_list[idx-1]:self.rowsize_list[idx]]
            end = restore_input[-1, self.input_size:]

            restore_array = np.concatenate([restore_input[:, 0:self.input_size], restore_output], axis = 1).reshape(-1)
            restore_array = np.concatenate([self.surplus_list[idx], restore_array, end])
            restore_list.append(restore_array)

        for skip_idx, skip_row in self.skiprow_dict.items():
            restore_list.insert(skip_idx, np.array(skip_row))

        return restore_list
