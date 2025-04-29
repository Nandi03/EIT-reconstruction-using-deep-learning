% This script uses EIDORS to simulate multi-touch on a 2D circular skin
% with 8 electrodes, with six particularly selected coordinates, made to
% replicate the experimental dataset. Boundary voltages are measured using
% EIDORS forward solver and stored in a CSV file. 

% EIDORS initialisation (adjust the path as necessary)
run('C:/path/to/eidors/eidors-v3.11/eidors/eidors_startup.m');

rng('shuffle');  % Ensure different random points every time you run the code

R = 1; % radius of the skin
n_electrodes = 8;  % Number of electrodes
grid_size = 0.5; % size of the squares (angular resolution)
imdl = mk_common_model('c2c', n_electrodes);  % Create a circular 2D model
imdl.fwd_model.nodes_per_elem = 4;
fmdl = imdl.fwd_model;  % Extract forward model

% Display the FEM model with electrode labels
show_fem(fmdl);

% Generate grid points in polar coordinates and convert to Cartesian
theta_vals = linspace(0, 2*pi, 7);  % Angle values, split into 6 slices


angle_shift = deg2rad(20);  % Convert 20 degrees to radians
elec_pos = linspace(0, 2*pi, n_electrodes + 1); % Original positions
elec_pos(end) = [];  % Remove duplicate at 2*pi

for i = 1:n_electrodes
    imdl.fwd_model.electrode(i).z_contact = 100;
end
fmdl_2d = imdl.fwd_model;  % Extract forward model
figure;


% Prepare storage for results
coordinates = [];
voltages = [];
area = [];

% coordinates of touch points
touch_points = [
    -0.050000, 0.400000; 
    -0.346410, 0.246497;
    -0.296410, -0.253503;
    0.050000, -0.400000;
    0.346410, -0.246497;
    0.296410, 0.253503
];


% Number of available touch points
num_touch_points = size(touch_points, 1);

% Loop for different impacted radii
for m = 0.1:0.1:0.4
    % Simulate for a random number of touch points between 1 and 6
    for sim_idx = 1:3000  % Run simulations
        % Randomly choose a number of touch points between 1 and 6
        n_touch_points = randi([1, num_touch_points]);
        
        % Randomly select those touch points from the possible coordinates
        random_indices = randperm(num_touch_points, n_touch_points);
        selected_points = touch_points(random_indices, :);
        
        % Simulate touch by modifying conductivity at selected points
        sim_img = mk_image(fmdl, 1);      % Homogeneous conductivity 
        
        % Modify conductivity for each selected point
        for touch_idx = 1:n_touch_points
            point = selected_points(touch_idx, :);
            sim_img.elem_data(find_elements_within_radius(fmdl, point, m)) = 0.1; % Decrease conductivity
        end
        
        % Solve for voltage measurements for this simulation
        voltage_data = fwd_solve(sim_img);   % This solves the forward problem
        
        % Store the coordinates of the touch points and voltage measurements
        % Store all touch point coordinates in a flat format 
        coordinates_row = nan(1, 2 * num_touch_points);  
        coordinates_row(1:2 * n_touch_points) = reshape(selected_points', 1, []); 
        coordinates = [coordinates; coordinates_row];  
        
        % Store the voltage data
        voltages = [voltages; voltage_data.meas'];  % Append voltage readings
        area = [area; m];
    end
end

% Display the size of coordinates and voltage results
disp(size(coordinates));
disp(size(voltages));

% Separate x and y coordinates into different matrices
x_coords = coordinates(:, 1:2:end); % Take all odd columns for X_Coord
y_coords = coordinates(:, 2:2:end); % Take all even columns for Y_Coord

% Combine x_coords, y_coords, and voltages into one matrix for the table
data_matrix = [x_coords, y_coords, voltages, area];

% Create variable names for the table
x_var_names = arrayfun(@(i) sprintf('X_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
y_var_names = arrayfun(@(i) sprintf('Y_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
voltage_var_names = arrayfun(@(i) sprintf('Voltage_%d', i), 1:size(voltages, 2), 'UniformOutput', false);
% 'Area' will be a single column name
area_var_name = {'Area'};

% Combine all variable names into one cell array
column_names = [x_var_names, y_var_names, voltage_var_names, area_var_name];


% Create the table with proper column names
result_table = array2table(data_matrix, 'VariableNames', column_names);

% Save results to CSV
writetable(result_table, 'multi_touch_simulation_8_6p_new.csv');

disp('Multi-touch simulation complete.');

function element_indices = find_elements_within_radius(fwd_model, coords, radius)
    % Find the indices of elements within a given radius of the touch coordinates
    %
    % fwd_model: EIDORS forward model containing mesh and nodes data
    % coords: 2D coordinates [x, y] where touch occurs (normalized)
    % radius: Radius within which elements will be affected
    
    % Extract element node indices and their coordinates
    nodes = fwd_model.nodes;  % Coordinates of all nodes in the model
    elems = fwd_model.elems;  % Indices of nodes for each element (triangles)
    
    % Calculate centroids of all elements
    centroids = zeros(size(elems, 1), 2);
    for i = 1:size(elems, 1)
        % Find the nodes of the current element
        node_indices = elems(i, :);  % Node indices for the current element
        
        % Get the coordinates of these nodes
        node_coords = nodes(node_indices, :);
        
        % Calculate the centroid of the element (average of the node coordinates)
        centroids(i, :) = mean(node_coords, 1);
    end
    
    % Calculate the distance from each centroid to the given coordinates
    distances = sqrt((centroids(:, 1) - coords(1)).^2 + (centroids(:, 2) - coords(2)).^2);
    
    % Find the indices of elements within the specified radius
    element_indices = find(distances <= radius);
end
