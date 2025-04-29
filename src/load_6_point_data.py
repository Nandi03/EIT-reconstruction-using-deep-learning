import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_output_array(filename):
    output_array = []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)  
        for row in reader:
            output_array.append(row)
    output_array = np.array(output_array, dtype=np.float32)
    output_array = np.concatenate((output_array, output_array), axis=0)

    np.savetxt("6_point_outputs.csv", output_array, delimiter=",", fmt="%.0f")
    return output_array


def is_adjacent(a, b, n=8):
    return (a % n) + 1 == b

def is_valid_config(I_source, I_sink, V_p, V_n):
    stim_adjacent = is_adjacent(I_source, I_sink)
    volt_adjacent = is_adjacent(V_p, V_n)
    no_overlap = len({I_source, I_sink}.intersection({V_p, V_n})) == 0
    return stim_adjacent and volt_adjacent and no_overlap

def find_adjacent_indices(csv_file):
    df = pd.read_csv(csv_file)  # header row = [0, 1, 2, 3]
    
    adjacent_indices = df[
        df.apply(lambda row: is_valid_config(row[0], row[1], row[2], row[3]), axis=1)
    ].index.tolist()

    print(adjacent_indices)
    return adjacent_indices


def get_voltages_array(electrodes_file, voltages_unpress_file, voltages_press_file, noise_factor=0.0001):
    # Read data
    df = pd.read_csv(electrodes_file)
    adjacent_indices = find_adjacent_indices(electrodes_file)
    
    # Load voltage files
    df_voltage_press = pd.read_csv(voltages_press_file)
    df_voltage_unpress = pd.read_csv(voltages_unpress_file)
    
    # Calculate voltage differences
    voltages_press = df_voltage_press.iloc[:, adjacent_indices]
    voltages_unpress = df_voltage_unpress.iloc[:, adjacent_indices]
    voltage_diffs = np.array(((voltages_press - voltages_unpress)).values.tolist())
    
    # Add noise (same shape as original data)
    noise = np.random.normal(loc=0, scale=np.random.uniform(0, noise_factor), size=voltage_diffs.shape)
    noisy_voltages = voltage_diffs + noise
    
    # concatenate
    voltages_combined = np.concatenate((voltage_diffs, noisy_voltages), axis=0)
    
    # Save and return
    np.savetxt("6_point_voltages_rms.csv", voltages_combined, delimiter=",", fmt="%.6f")
    return voltages_combined

if __name__ == "__main__":

    output = get_output_array("patterns.csv")
    print(output.shape)
    print(output)

    arr = get_voltages_array("electrodes.csv", "unpressed_rms.csv", "pressed_rms.csv")
    print(arr)
    print(arr.shape)
    
    combos = find_adjacent_indices("electrodes.csv")
    print(combos)