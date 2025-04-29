% This script uses EIDORS to simulate multi-touch on a 2D circular skin
% with n_electrodes electrodes, with touch points sampled on a polar grid
% of a specified resolutions, grid_size. We simulate with varying impacted 
% area across selected touch points.
% Boundary voltages are measured using EIDORS forward solver and stored 
% in a CSV file.

% EIDORS initialisation (adjust the path as necessary)
run('C:/path/to/eidors/eidors-v3.11/eidors/eidors_startup.m');

rng('shuffle');  % Ensure different random points every time you run the code

R = 1; % radius of the skin
n_electrodes = 24;  % Number of electrodes
grid_size = 0.4; % size of the squares (angular resolution)
imdl = mk_common_model('c2c', n_electrodes);  % Create a circular 2D model
imdl.fwd_model.nodes_per_elem = 4;
fmdl = imdl.fwd_model;  % Extract forward model

% Generate grid points in polar coordinates and convert to Cartesian
theta_vals = 0:grid_size:(2*pi);         % Angular resolution of the grid
radius_vals = grid_size:grid_size:R;    % Radial resolution of the grid

% Prepare storage for results
coordinates = [];
voltages = [];
forces = [];

% Loop through grid points to generate possible touch points
touch_points = [0, 0];
for r = radius_vals
    for theta = theta_vals
        % Convert polar coordinates to Cartesian
        x = r * cos(theta);
        y = r * sin(theta);
        
        % Ensure the point is within the circular area
        if sqrt(x^2 + y^2) <= R
            % Store the generated points
            touch_points = [touch_points; x, y];
        end
    end
end

% Number of available touch points
num_touch_points = size(touch_points, 1);
conductivies = [0.1, 0.2, 0.3, 0.4];

% Loop for different impacted radii
for m = 0.1:0.1:0.4
    % Simulate for a random number of touch points between 1 and 32
    for sim_idx = 1:5000  % Run simulations
        % Randomly choose a number of touch points between 1 and 32
        n_touch_points = randi([1, num_touch_points]);
        
        % Randomly select those touch points from the possible coordinates
        random_indices = randperm(num_touch_points, n_touch_points);
        selected_points = touch_points(random_indices, :);
        
        % Simulate touch by modifying conductivity at selected points
        sim_img = mk_image(fmdl, 1);      % Homogeneous conductivity (for simplicity)
        
        % Initialize array to store the forces applied at each point
        forces_row = nan(1, n_touch_points);  % Store forces for this simulation

        % Modify conductivity for each selected point
        for touch_idx = 1:n_touch_points
            point = selected_points(touch_idx, :);
            
            % Randomize conductivity decrease: Choose 0.1, 0.3, or 0.5
            i = randi(4);
            conductivity_decrease = conductivies(i);
            
            % Store the force applied (proportional to conductivity decrease)
            forces_row(touch_idx) = conductivity_decrease;  % Store the force
            
            % Modify the elements' conductivity within the touch radius
            sim_img.elem_data(find_elements_within_radius(fmdl, point, m)) = conductivity_decrease;
        end
        
        % Solve for voltage measurements for this simulation
        voltage_data = fwd_solve(sim_img);   % This solves the forward problem
        
        % Store the coordinates of the touch points and voltage measurements
        coordinates_row = nan(1, 2 * num_touch_points);  
        coordinates_row(1:2 * n_touch_points) = reshape(selected_points', 1, []);  
        coordinates = [coordinates; coordinates_row];  
        
        % Store the forces applied at each touch point
        forces_row_extended = nan(1, num_touch_points);  
        forces_row_extended(1:n_touch_points) = forces_row;  
        forces = [forces; forces_row_extended];  
        
        % Store the voltage data
        voltages = [voltages; voltage_data.meas'];  
    end
end

% Display the size of coordinates and voltage results
disp(size(coordinates));
disp(size(voltages));
disp(size(forces));

% Separate x and y coordinates into different matrices
x_coords = coordinates(:, 1:2:end); 
y_coords = coordinates(:, 2:2:end); 

% Combine x_coords, y_coords, voltages, forces, and area into one matrix for the table
data_matrix = [x_coords, y_coords, voltages, forces];

% Create variable names for the table
x_var_names = arrayfun(@(i) sprintf('X_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
y_var_names = arrayfun(@(i) sprintf('Y_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
voltage_var_names = arrayfun(@(i) sprintf('Voltage_%d', i), 1:size(voltages, 2), 'UniformOutput', false);
force_var_names = arrayfun(@(i) sprintf('Force_%d', i), 1:num_touch_points, 'UniformOutput', false);

% Combine all variable names into one cell array
column_names = [x_var_names, y_var_names, voltage_var_names, force_var_names];

% Create the table with proper column names
result_table = array2table(data_matrix, 'VariableNames', column_names);

% Save results to CSV
writetable(result_table, 'circ_random_forces_24_4.csv');

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
