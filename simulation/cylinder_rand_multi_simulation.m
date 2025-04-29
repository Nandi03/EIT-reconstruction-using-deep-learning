% This script using EIDORS software to simulate multi-touch 
% on a cylindrical 3D EIT-based skin. Using the forward solver
% we measure boundary voltages, when multiple touch points on the skin
% are pressed. We measure voltages across different forces by adjusting 
% the impacted area. The data is stored in a csv file.

% EIDORS initialization (adjust the path as necessary)
run('C:/path/to/eidors/eidors-v3.11/eidors/eidors_startup.m');

rng('shuffle');  % Ensure different random points every time you run the code

% Create skin model
nelec= 16; 
ring_vert_pos = [0.1]; 
R = 0.2;
fmdl= ng_mk_cyl_models([1,R,0.05],[nelec,ring_vert_pos],[0.05,2,0.05]);

stim = mk_stim_patterns(nelec,1,[0,1],[0,1],{'meas_current'},1);
fmdl.stimulation = stim;

conduct = 1;
img = mk_image( fmdl, conduct ); 

% Prepare storage for results
coordinates = [];
voltages = [];
area = [];

grid_size = 0.2;

% Generate grid points in polar coordinates and convert to Cartesian
theta_vals = 0:grid_size:(2*pi); 
Z_vals = 0.0:0.1:1;

% Loop through grid points to generate distinct pairs of touch points
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

% Loop for different touch radii
for m = 0.1:0.1:0.2
    % Simulate for a random number of touch points between 1 and 
    for sim_idx = 1:4000  % Run simulations
        % Randomly choose a number of touch points between 1 and 
        n_touch_points = randi([2,  num_touch_points]);
        
        % Randomly select those touch points from the possible coordinates
        random_indices = randperm(num_touch_points, n_touch_points);
        selected_points = touch_points(random_indices, :);
        
        % Simulate touch by modifying conductivity at selected points
        sim_img = mk_image(fmdl, 1);     
        
        % Modify conductivity for each selected point
        for touch_idx = 1:n_touch_points
            point = selected_points(touch_idx, :);
            sim_img.elem_data(find_elements_within_radius(fmdl, point, m)) = 0.1; % Decrease conductivity
        end
        
        % Solve for voltage measurements for this simulation
        voltage_data = fwd_solve(sim_img);   
        
        % Store the coordinates of the touch points and voltage measurements
        % Store all touch point coordinates in a flat format
        coordinates_row = nan(1, 3 * num_touch_points);  
        coordinates_row(1:3 * n_touch_points) = reshape(selected_points', 1, []);  
        coordinates = [coordinates; coordinates_row];  % Append to coordinates
        
        % Store the voltage data
        voltages = [voltages; voltage_data.meas'];  % Append voltage readings
        area = [area;m];
    end
end

% Display the size of coordinates and voltage results
disp(size(coordinates));
disp(size(voltages));

% Separate x, y, z coordinates into different matrices
x_coords = coordinates(:, 1:3:end); % Take all first columns for X_Coord
y_coords = coordinates(:, 2:3:end); % Take all second columns for Y_Coord
z_coords = coordinates(:, 3:3:end); % Take all third columns for Z_Coord

% Combine x_coords, y_coords, z_coords, and voltages into one matrix for the table
data_matrix = [x_coords, y_coords, z_coords, voltages, area];

% Create variable names for the table
x_var_names = arrayfun(@(i) sprintf('X_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
y_var_names = arrayfun(@(i) sprintf('Y_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
z_var_names = arrayfun(@(i) sprintf('Z_Coord_%d', i), 1:num_touch_points, 'UniformOutput', false);
voltage_var_names = arrayfun(@(i) sprintf('Voltage_%d', i), 1:size(voltages, 2), 'UniformOutput', false);
area_name = {'Area'};

% Combine all variable names into one cell array
column_names = [x_var_names, y_var_names, z_var_names, voltage_var_names, area_name];

% Create the table with proper column names
result_table = array2table(data_matrix, 'VariableNames', column_names);

% Save results to CSV
writetable(result_table, 'cylinder_multi_touch_16_2_new_2.csv');

disp('Multi-touch simulation complete.');

function element_indices = find_elements_within_radius(fwd_model, coords, radius)
    % Find the indices of mesh elements whose centroids are within a given radius
    nodes = fwd_model.nodes;  % N x 3 matrix of node coordinates
    elems = fwd_model.elems;  % E x 4 matrix of element connectivity
    
    % Ensure coords is a row vector
    coords = coords(:)';  % Make sure it's 1x3
    
    % Calculate centroids of each element
    centroids = zeros(size(elems,1), 3);
    for i = 1:size(elems,1)
        centroids(i,:) = mean(nodes(elems(i,:),:), 1);
    end
    
    % Compute distances from centroids to target coordinate
    distances = sqrt(sum((centroids - coords).^2, 2));
    
    % Select elements within radius
    element_indices = find(distances <= radius);
end