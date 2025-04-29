% This script uses EIDORS to simulate single-touch on a 2D circular skin
% with n_electrodes electrodes, with touch points sampled on a polar grid
% of a specified resolutions, grid_size. 
% Boundary voltages are measured using EIDORS forward solver and stored 
% in a CSV file.

% EIDORS initialisation
run('C:/path/to/eidors/eidors-v3.11/eidors/eidors_startup.m');

R = 1; % radius of the skin
n_electrodes = 16;  % Number of electrodes
grid_size = 0.5; % grid resolution
imdl = mk_common_model('c2c', n_electrodes);  % Create a circular 2D model
imdl.fwd_model.nodes_per_elem = 4;
fmdl_2d = imdl.fwd_model;  % Extract forward model
% Generate grid points in polar coordinates and convert to Cartesian
theta_vals = 0:grid_size:(2*pi);         % Angular resolution of the grid
radius_vals = grid_size:grid_size:R;    % Radial resolution of the grid

show_fem(fmdl_2d)
% Prepare storage for results
coordinates = [];
voltages = [];
area = [];

% baseline voltages measured with homogeneuous conductivity
coordinates = [coordinates; "base", "base"];
sim_img = mk_image(fmdl_2d, 1); 
voltage_data = fwd_solve(sim_img);
voltages = [voltages; voltage_data.meas'];
area = [area; 0];
multiplier = -999.0;

% loop through different impacted radii
for m = 0.1:0.1:0.5
    % Loop through grid points
    center_coord = [0, 0]; 
    % Simulate the touch at the center
    sim_img = mk_image(fmdl_2d, 1);  % Create a homogeneous image
    sim_img.elem_data(find_elements_within_radius(fmdl_2d, center_coord, m)) = 0.1;  % Modify conductivity at the center
    % Solve for voltage data for the center point
    voltage_data = fwd_solve(sim_img);
    
    % Store results for the center point
    coordinates = [coordinates; center_coord];  
    voltages = [voltages; voltage_data.meas'];  
    area = [area; m];
    for r = radius_vals
        for theta = theta_vals
            % Convert polar coordinates to Cartesian
            x = r * cos(theta);
            y = r * sin(theta);
            
            % Ensure the point is within the circular area
            if sqrt(x^2 + y^2) <= R
                % Simulate touch at (x, y)
                sim_img = mk_image(fmdl_2d, 1);  % Homogeneous conductivity
               
                % Find elements within the radius
         
                sim_img.elem_data(find_elements_within_radius(fmdl_2d, [x, y], m)) = 0.1 ; % Decrease conductivity at the point
                show_fem(sim_img)
                % Solve for voltage data for this touch point
                voltage_data = fwd_solve(sim_img);  
                
                % Store results
                    
                coordinates = [coordinates; x, y];    % Store (x, y) coordinates
                voltages = [voltages; voltage_data.meas'];  % Store voltage readings
                area = [area; m];
                
            end
        end
    end
end

disp(size(coordinates));
disp(size(voltages));
% Create a table with the results
result_table = table(coordinates(:,1), coordinates(:,2), voltages, area, ...
    'VariableNames', {'X_Coord', 'Y_Coord', 'Voltage', 'Area'});

% Save to CSV
writetable(result_table, 'touch_simulation_results_8.csv');

disp('Simulation complete.');


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

