import os
import numpy as np
import time

def read_numerical_csv_file(file_path, num_type=float):
    file = open(file_path)
    data_raw = []
    for line in file:
        data_raw.append(line.strip().split(","))
    data = []
    for line in data_raw[1:]:
        line_data = []
        for item in line:
            line_data.append(num_type(item))
        data.append(line_data)
    file.close()
    return np.array(data)

def read_csv_file(file_path, data_types):
    file = open(file_path, "r", encoding="utf-8")
    data_raw = []
    for line in file:
        data_raw.append(line.strip().split(","))
    data = []
    for line in data_raw[1:]:
        line_data = []
        for i,item in enumerate(line):
            line_data.append(data_types[i](item))
        data.append(line_data)
    file.close()
    return data

def write_csv_file(file_path, header, data):
    file = open(file_path, "w", encoding="utf-8")
    file.write(header+"\n")
    for line in data:
        last_item_index =  len(line)-1
        for i,item in enumerate(line):
            file.write(str(item)+ ("\n" if i==last_item_index else ","))
    file.close()


def write_string(file_path, string_to_write):
    file = open(file_path, "w", encoding="utf-8")
    file.write(string_to_write)
    file.close()