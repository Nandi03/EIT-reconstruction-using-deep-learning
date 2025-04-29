% This script uses EIDORS software to simulate multi-touch 
% on a cylindrical 3D EIT-based skin. Using the forward solver
% we measure boundary voltages, when multiple points on the skin
% are pressed, with varying forces. The data is stored in a csv file.

% EIDORS initialisation (adjust the path as necessary)
run('C:/path/to/eidors/eidors-v3.11/eidors/eidors_startup.m');

rng('shuffle');  % Ensure different random points every time you run the code

% create skin model
nelec = 16; 
ring_vert_pos = [0.1]; 
R = 0.2;
fmdl = ng_mk_cyl_models([1, R, 0.05], [nelec, ring_vert_pos], [0.05, 2, 0.05]);

stim = mk_stim_patterns(nelec, 1, [0,1], [0,1], {'meas_current'}, 1);
fmdl.stimulation = stim;

conduct = 1;
img = mk_image(fmdl, conduct); 

% Prepare storage for results
coordinates = [];
voltages = [];
areas = [];  % Store individual areas for each touch point

grid_size = 0.4;
% Generate grid points in polar coordinates and convert to Cartesian
theta_vals = 0:grid_size:(2*pi); 
Z_vals = 0.0:0.2:1;

% Generate all possible touch points
touch_points = [];
for z = Z_vals
    for theta = theta_vals
        % Convert polar coordinates to Cartesian
        x = R * cos(theta);
        y = R * sin(theta);
        
        % Ensure the point is within the circular area
        if sqrt(x^2 + y^2) <= R
            % Store the generated points
            touch_points = [touch_points; x, y, z];
        end
    end
end
% Number of available touch points
num_touch_points = size(touch_points, 1);

% Define a constant force (conductivity)
constant_conductivity = 0.1;  % This remains the same for all points

% Run multiple simulations
for sim_idx = 1:15000  
    % Randomly choose a number of touch points (at least 2)
    n_touch_points = randi([2, num_touch_points]);
    
    % Randomly select those touch points
    random_indices = randperm(num_touch_points, n_touch_points);
    selected_points = touch_points(random_indices, :);
    
    % Create simulation image with homogeneous conductivity
    sim_img = mk_image(fmdl, 1);  
    
    touch_areas = nan(1, num_touch_points);  % Store areas per touch point

    % Modify conductivity for each selected point
    for touch_idx = 1:n_touch_points
        point = selected_points(touch_idx, :);
        
        % Randomly select a touch area between 0.1 and 0.4
        m = 0.1 + (0.4 - 0.1) * rand();
        touch_areas(touch_idx) = m;

        % Apply constant conductivity to elements within radius
        sim_img.elem_data(find_elements_within_radius(fmdl, point, m)) = constant_conductivity;
    end

    % Solve for voltage measurements
    voltage_data = fwd_solve(sim_img);
    
    % Store the coordinates of the touch points
    coordinates_row = nan(1, 3 * num_touch_points);  
    coordinates_row(1:3 * n_touch_points) = reshape(selected_points', 1, []);  
    coordinates = [coordinates; coordinates_row];  

    % Store the voltage data
    voltages = [voltages; voltage_data.meas'];  
    
    % Store areas (each row corresponds to a simulation)
    areas = [areas; touch_areas];  
end

% Display the size of coordinates and voltage results
disp(size(coordinates));
disp(size(voltages));

% Separate x, y, z coordinates into different matrices
x_coords = coordinates(:, 1:3:end);
y_coords = coordinates(:, 2:3:end);
z_coords = coordinates(:, 3:3:end);

% Combine data into one matrix for the table
data_matrix = [x_coords, y_coords, z_coords, voltages, areas];

% Create variable names for the table
x_var_names = arrayfun(@(i) sprintf('X_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
y_var_names = arrayfun(@(i) sprintf('Y_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
z_var_names = arrayfun(@(i) sprintf('Z_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
voltage_var_names = arrayfun(@(i) sprintf('Voltage_%d', i), 1:size(voltages, 2), 'UniformOutput', false);
area_var_names = arrayfun(@(i) sprintf('Area_%d', i), 1:num_touch_points, 'UniformOutput', false);

% Combine all variable names into one cell array
column_names = [x_var_names, y_var_names, z_var_names, voltage_var_names, area_var_names];

% Create the table with proper column names
result_table = array2table(data_matrix, 'VariableNames', column_names);

% Save results to CSV
writetable(result_table, 'cylinder_randforces_16_2.csv');

disp('Multi-touch simulation complete.');

function element_indices = find_elements_within_radius(fwd_model, coords, radius)
    % Find the indices of mesh elements whose centroids are within a given radius
    nodes = fwd_model.nodes;  % N x 3 matrix of node coordinates
    elems = fwd_model.elems;  % E x 4 matrix of element connectivity
    
    % Ensure coords is a row vector
    coords = coords(:)';  % Make sure it's 1x3
    
    % Calculate centroids of each tetrahedral element
    % Sum the coordinates of all 4 nodes and divide by 4
    centroids = zeros(size(elems,1), 3);
    for i = 1:size(elems,1)
        centroids(i,:) = mean(nodes(elems(i,:),:), 1);
    end
    
    % Compute distances from centroids to target coordinate
    distances = sqrt(sum((centroids - coords).^2, 2));
    
    % Select elements within radius
    element_indices = find(distances <= radius);
end
