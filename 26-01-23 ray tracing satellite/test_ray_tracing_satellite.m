clear all; close all; clc;

% Test script for Ray Tracing Satellite Model
% Tests the basic functionality of the RayTracingSatellite class

% Add current directory to path to ensure class is found
addpath(pwd);

% Create a standard satellite according to specification
satellite = RayTracingSatellite.create_standard_satellite();

% Print satellite information
fprintf('=== Satellite Information ===\n');
satellite.print_info();

% Test geometry methods
fprintf('\n=== Geometry Information ===\n');
vertices = satellite.get_vertices();
fprintf('Vertices (%d points):\n', size(vertices, 1));
disp(vertices);

faces = satellite.get_faces();
fprintf('Faces (%d faces):\n', size(faces, 1));
for i = 1:size(faces, 1)
    fprintf('Face %d: vertices ', i);
    fprintf('%d ', faces(i, :));
    fprintf('\n');
end

normals = satellite.get_face_normals();
fprintf('Face normals:\n');
disp(normals);

% Test propagation methods (should return unchanged for simplified model)
fprintf('\n=== Propagation Test ===\n');
[pos_new, vel_new] = satellite.propagate_orbit(1.0);
fprintf('Position after 1s: [%.2f, %.2f, %.2f] m\n', pos_new(1), pos_new(2), pos_new(3));
fprintf('Velocity after 1s: [%.2f, %.2f, %.2f] m/s\n', vel_new(1), vel_new(2), vel_new(3));

[att_new, omega_new] = satellite.propagate_attitude(1.0);
fprintf('Attitude after 1s: [%.4f, %.4f, %.4f, %.4f]\n', att_new(1), att_new(2), att_new(3), att_new(4));
fprintf('Angular velocity after 1s: [%.4f, %.4f, %.4f] rad/s\n', omega_new(1), omega_new(2), omega_new(3));

% Visualize the satellite
fprintf('\n=== Satellite Visualization ===\n');
satellite.plot_satellite();

fprintf('\n=== Test Completed Successfully ===\n');
