run('C:/path/to/eidors/eidors-v3.11/eidors/eidors_startup.m');

% This script using EIDORS software to simulate single-touch 
% on a cylindrical 3D EIT-based skin. Using the forward solver
% we measure boundary voltages, when single touch points on the skin
% are pressed.We measure voltages with different forces, by adjusting
% the impacted area. The data is stored in a csv file.

% Create cylindrical skin
nelec= 16; % num of electrodes, 8, 16..
ring_vert_pos = [0.1]; 
R = 0.2;
fmdl= ng_mk_cyl_models([1,R,0.05],[nelec,ring_vert_pos],[0.05,2,0.05]);

stim = mk_stim_patterns(nelec,1,[0,1],[0,1],{'meas_current'},1);
fmdl.stimulation = stim;

conduct = 1;
img = mk_image( fmdl, conduct ); 

show_fem(fmdl);

% Prepare storage for results
coordinates = [];
voltages = [];
area = [];

% measure baseline voltages with homogeneous conductivity
coordinates = [coordinates; "base", "base", "base"];
sim_img = mk_image(fmdl, 1);  
voltage_data = fwd_solve(sim_img);
voltages = [voltages; voltage_data.meas'];
area = [area; 0];

grid_size = 0.2; % grid resolution
% Generate grid points in polar coordinates and convert to Cartesian
theta_vals = 0:grid_size:(2*pi);  % Angular resolution of the grid


for m = 0.1:0.1:0.2

    % Loop through grid points
    center_coord = [0, 0, 0];

    % Simulate the touch at the center
    sim_img = mk_image(fmdl, 1);  % Create a homogeneous image

    % Modify conductivity at the center
    sim_img.elem_data(find_elements_within_radius(fmdl, center_coord, m)) = 0.1;  
    
    % Solve for voltage data for the center point
    voltage_data = fwd_solve(sim_img);
    
    % Store results for the center point
    coordinates = [coordinates; center_coord];  % Add the center coordinates
    voltages = [voltages; voltage_data.meas'];  % Add voltage readings
    area = [area; m];
    for theta = theta_vals
        for  z = 0:0.1:1
            % Convert polar coordinates to Cartesian
            x = R * cos(theta);
            y = R * sin(theta);
            
            % Ensure the point is within the circular area
            if sqrt(x^2 + y^2) <= R
                sim_img = mk_image(fmdl, 1);  
                % Simulate touch at (x, y)
                % Find elements within the impacted radius
                elements = find_elements_within_radius(fmdl, [x, y, z], m);
                
                % Decrease conductivity at the impacted radius
                sim_img.elem_data(elements) = 0.1 ; 

                % Solve the forward problem for voltage data 
                voltage_data = fwd_solve(sim_img);   
                
                % Store results
                coordinates = [coordinates; x, y, z];    % Store (x, y, z) coordinates
                voltages = [voltages; voltage_data.meas'];  
                area = [area; m];
            end
        end
    end
end     

disp(size(coordinates));
disp(size(voltages));
% Create a table with the results
result_table = table(coordinates(:,1), coordinates(:,2), coordinates(:, 3), voltages, area, ...
    'VariableNames', {'X_Coord', 'Y_Coord', 'Z_Coord', 'Voltage', 'Area'});

% Save to CSV
writetable(result_table, 'cylinder_sim_single_16_2_new.csv');

disp('Simulation complete.');

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



