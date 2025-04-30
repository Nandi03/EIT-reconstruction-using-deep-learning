import numpy as np
import pandas as pd

class DataLoader:
    """
    DataLoader for processing single-touch and multi-touch tactile sensing data.

    This class reads CSV files containing voltage measurements and touch coordinates 
    from our custom datasets created with EIDORS simulation for EIT-based skin. 
    It builds input-output datasets suitable for deep learning tasks such as 
    classification or regression.

    Attributes:
        pairs_electrodes (int): Number of electrode pairs used in voltage measurements.
        precision (int, optional): Number of decimal places to round coordinates to (default is 2).
        coordinate_to_index (dict): Maps rounded coordinate tuples to unique class indices.
        index_to_coordinate (dict): Maps class indices back to coordinate tuples.
        voltage_array (list): List of input feature arrays (voltage differences).
        output_array (list): List of output label arrays (one-hot or weighted vectors).
        baseline (np.ndarray or None): Baseline voltage readings.
        num_classes (int): Number of unique touch positions (classes) detected.
    """
    def __init__(self, pairs_electrodes, precision=5):
        self.pairs_electrodes = pairs_electrodes
        self.precision = precision
        self.coordinate_to_index = {}
        self.index_to_coordinate = {}
        self.voltage_array = []
        self.output_array = []
        self.baseline = None
        self.num_classes = 0

    def load_data(self, single_touch_file, multi_touch_file, three_d=False, force=False, randforce=False):
        """
        Load and process single-touch and multi-touch tactile data.

        Args:
            single_touch_file (str): ath to single-touch CSV file.
            multi_touch_file (str): Path to multi-touch CSV file.
            three_d (bool, optional): Whether data includes 3D coordinates. Defaults to False.
            force (bool, optional): Whether to weight labels by force/area. Defaults to False
            randforce (bool, optional): Whether to use individual force values. Defaults to False
        """
        single_touch_data = pd.read_csv(single_touch_file)
        self._initialise_baseline(single_touch_data, three_d)
        self._process_single_touch_data(single_touch_data, three_d, force)
        multi_touch_data = pd.read_csv(multi_touch_file)
        self._process_multi_touch_data(multi_touch_data, three_d, force, randforce)

        self.voltage_array = np.array(self.voltage_array, dtype=np.float32)
        self.output_array = np.array(self.output_array, dtype=np.float32)

    def _initialise_baseline(self, data, three_d):
        """Set the baseline from the first sample."""
        if not three_d:
            self.baseline = data.iloc[0, 2: 2 + self.pairs_electrodes] # baseline for 2D coords
        else:
            self.baseline = data.iloc[0, 3: 3 + self.pairs_electrodes] # baseline for 3D coords

    def _process_single_touch_data(self, data, three_d, force):
        """Process all single touch samples."""
        self._map_coordinates(data, three_d)
        self._build_single_touch_arrays(data, three_d, force)

    def _map_coordinates(self, data, three_d):
        """Map each coordinate to an index for classification.""" 
        index = 0
        for i, row in data.iterrows():
            if i == 0:
                continue  # Skip baseline
            coord = self._get_coord(row, three_d)
            if coord not in self.coordinate_to_index:
                self.coordinate_to_index[coord] = index
                self.index_to_coordinate[index] = coord
                index += 1
        self.num_classes = len(self.coordinate_to_index) # set class count

    def _build_single_touch_arrays(self, data, three_d, force):
        """Create input-output pairs from single touch data."""
        for i, row in data[1:].iterrows():
            voltage_readings = self._extract_voltage_readings(row, three_d) - self.baseline
            coord_index = self.coordinate_to_index[self._get_coord(row, three_d)]
            one_hot_vector = np.zeros(self.num_classes)
            if force:
                area = self._extract_area(row, three_d)
                one_hot_vector[coord_index] = area 
            else:
                one_hot_vector[coord_index] = 1
            self.voltage_array.append(voltage_readings)
            self.output_array.append(one_hot_vector)

    def _process_multi_touch_data(self, data, three_d, force, randforce):
        """Process multi-touch samples and append them."""
        for _, row in data.iterrows():
            voltages, output = self._extract_multi_touch_sample(row, three_d, force, randforce)
            self.voltage_array.append(voltages)
            self.output_array.append(output)

    def _extract_multi_touch_sample(self, row, three_d, force, randforce):
        """Extract one multi-touch sample."""
        output = np.zeros(self.num_classes)
        area = 1
        # Read the area column, for the force applied if this is a regression task
        if not three_d and not randforce:
            area = float(row[self.num_classes * 2 + self.pairs_electrodes])
        elif three_d and not randforce:
            area = float(row[(self.num_classes - 1) * 3 + self.pairs_electrodes])

        i = 1
        # process 3D coordinates -> classes
        if three_d:
            while i < self.num_classes:
                x = row.get(f'X_Coord_{i}')
                y = row.get(f'Y_Coord_{i}')
                z = row.get(f'Z_Coord_{i}')
                if pd.isna(x) or pd.isna(y) or pd.isna(z):
                    break
                coord = (round(float(x), self.precision), round(float(y), self.precision), round(float(z), self.precision))
                index = self.coordinate_to_index.get(coord)
                if index is not None:
                    output[index] = self._determine_force_value(row, i, force, randforce, three_d, area)
                i += 1
            # Find the change in voltage signals, from baseline
            voltages = row[(self.num_classes - 1) * 3: (self.num_classes - 1) * 3 + self.pairs_electrodes].to_numpy() - self.baseline
        
        # process 2D coordinates -> classes
        else:
            while i < self.num_classes + 1:
                x = row.get(f'X_Coord_{i}')
                y = row.get(f'Y_Coord_{i}')
                if pd.isna(x) or pd.isna(y):
                    break
                coord = (round(float(x), self.precision), round(float(y), self.precision))
                index = self.coordinate_to_index.get(coord)
                if index is not None:
                    output[index] = self._determine_force_value(row, i, force, randforce, three_d, area)
                i += 1
            # Find the change in voltage signals, from baseline
            voltages = row[self.num_classes * 2: self.num_classes * 2 + self.pairs_electrodes].to_numpy() - self.baseline

        return voltages, output

    def _determine_force_value(self, row, i, force, randforce, three_d, area):
        """Determine what force value to assign in the output vector."""
        if randforce:
            key = f'Area_{i}' if three_d else f'Force_{i}'
            return row.get(key, 1)
        return area if force else 1

    def _get_coord(self, row, three_d):
        """Extract rounded coordinate from a row."""
        if not three_d:
            return (round(float(row['X_Coord']), self.precision), round(float(row['Y_Coord']), self.precision))
        else:
            return (round(float(row['X_Coord']), self.precision), round(float(row['Y_Coord']), self.precision), round(float(row['Z_Coord']), self.precision))

    def _extract_voltage_readings(self, row, three_d):
        """Extract voltage readings from a row."""
        if not three_d:
            return row[2:2 + self.pairs_electrodes].to_numpy()
        else:
            return row[3:3 + self.pairs_electrodes].to_numpy()

    def _extract_area(self, row, three_d):
        """Extract force/area information from a row."""
        if not three_d:
            return float(row[2 + self.pairs_electrodes])
        else:
            return float(row[3 + self.pairs_electrodes])
