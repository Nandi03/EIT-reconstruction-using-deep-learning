% This script uses EIDORS to simulate single-touch on a 2D circular skin
% with 8 electrodes, with six particularly selected coordinates, made to
% replicate the experimental dataset. Boundary voltages are measured using
% EIDORS forward solver and stored in a CSV file.

% Initialise EIDORS
run('C:/path/to/eidors/eidors-v3.11/eidors/eidors_startup.m');
 
R = 1; % radius of the skin
n_electrodes = 8;  % Number of electrodes
grid_size = 0.5; % size of the squares
imdl = mk_common_model('c2c', n_electrodes);  % Create a circular 2D model
imdl.fwd_model.nodes_per_elem = 4;

% Number of electrodes and radius
n_electrodes = 8;
R = 1;

angle_shift = deg2rad(20);  % Convert 20 degrees to radians
elec_pos = linspace(0, 2*pi, n_electrodes + 1); % Original positions
elec_pos(end) = [];  % Remove duplicate at 2*pi

% Apply angle shift
elec_pos = elec_pos + angle_shift;

for i = 1:n_electrodes
    imdl.fwd_model.electrode(i).pos = R * [cos(elec_pos(i)), sin(elec_pos(i))];
    imdl.fwd_model.electrode(i).z_contact = 0.1;
end
fmdl_2d = imdl.fwd_model;  % Extract forward model


theta_vals = linspace(0, 2*pi, 7);  % Angle values, split into 6 slices

% Prepare storage for results
coordinates = [];
voltages = [];
area = [];

% baseline voltages measured with homogeneous conductivity
coordinates = [coordinates; "base", "base"];
sim_img = mk_image(fmdl_2d, 1);
voltage_data = fwd_solve(sim_img);
voltages = [voltages; voltage_data.meas'];
area = [area; 0];

% six touch points defined
touch_points = [
    -0.050000, 0.400000; 
    -0.346410, 0.246497;
    -0.296410, -0.253503;
    0.050000, -0.400000;
    0.346410, -0.246497;
    0.296410, 0.253503
];

for m = 0.1:0.1:0.5
    % Loop through grid points
    for i = 1:size(touch_points)
         x = touch_points(i, 1);  % Extract the x-coordinate
         y = touch_points(i, 2);  % Extract the y-coordinate
            
        % Ensure the point is within the circular area
        if sqrt(x^2 + y^2) <= R
            % Simulate touch at (x, y) 
            sim_img = mk_image(fmdl_2d, 1);      % Homogeneous conductivity
      
           
            sim_img.elem_data(find_elements_within_radius(fmdl_2d, [x, y], m)) = 0.1 ; % Decrease conductivity at the point
            % Solve for voltage data for this touch point
            voltage_data = fwd_solve(sim_img);   % This solves the forward problem
            show_fem(sim_img)
            % Store results
                
            coordinates = [coordinates; x, y];    % Store (x, y) coordinates
            voltages = [voltages; voltage_data.meas'];  % Store voltage readings
            area = [area; m];
            
        end
    end
end

disp(size(coordinates));
disp(size(voltages));
% Create a table with the results
result_table = table(coordinates(:,1), coordinates(:,2), voltages, area, ...
    'VariableNames', {'X_Coord', 'Y_Coord', 'Voltage', 'Area'});

% Save to CSV
writetable(result_table, 'touch_simulation_results_8_6p_new.csv');

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

