% Ray Tracing Satellite Model
% Implements a modular satellite model constructed from cuboid components
% Based on specification: modular design for easy component addition

classdef RayTracingSatellite < handle
    % Modular satellite model for ray tracing applications
    %
    % Properties:
    %   components - Cell array of CuboidComponent objects
    %   mass - Total satellite mass (computed from components)
    %   position - Orbital position vector [x, y, z] in meters
    %   velocity - Orbital velocity vector [vx, vy, vz] in meters/second
    %   attitude - Attitude quaternion [q0, q1, q2, q3]
    %   angular_velocity - Angular velocity vector [wx, wy, wz] in rad/s
    %   body_frame_origin - Origin of body frame in bus coordinates
    %   name - Satellite name

    properties
        components        % Cell array of CuboidComponent objects
        mass             % Total satellite mass (computed)
        position         % Orbital position vector (m)
        velocity         % Orbital velocity vector (m/s)
        attitude         % Attitude quaternion
        angular_velocity % Angular velocity vector (rad/s)
        body_frame_origin % Body frame origin in bus coordinates
        name             % Satellite name
    end
    
    methods
        function obj = RayTracingSatellite(components, position, velocity, attitude, angular_velocity, name)
            % Constructor for modular ray tracing satellite model
            if nargin < 6
                name = 'RayTracingSat';
            end
            if nargin < 5
                angular_velocity = [0; 0; 0];
            end
            if nargin < 4
                attitude = [1; 0; 0; 0];  % Identity quaternion
            end
            if nargin < 3
                velocity = [0; 7669.2; 0];  % Approximate LEO velocity
            end
            if nargin < 2
                position = [6778137; 0; 0];  % LEO orbit (400km altitude)
            end
            if nargin < 1 || isempty(components)
                % Create default satellite with bus and panels
                components = obj.create_default_components();
            end

            % Set properties
            obj.components = components;
            obj.position = position;
            obj.velocity = velocity;
            obj.attitude = attitude;
            obj.angular_velocity = angular_velocity;
            obj.body_frame_origin = [0, 0, 0.0];  % On Nadir deck surface
            obj.name = name;

            % Compute total mass
            obj.update_mass();
        end

        function update_mass(obj)
            % Update total mass from components
            % For now, assign fixed masses to components
            obj.mass = 0;
            for i = 1:length(obj.components)
                comp = obj.components{i};
                % Assign mass based on volume (simplified)
                volume = prod(comp.dimensions);
                density = 1000;  % kg/m³ (simplified)
                comp_mass = volume * density;
                obj.mass = obj.mass + comp_mass;
            end
        end

        function add_component(obj, component)
            % Add a component to the satellite
            obj.components{end+1} = component;
            obj.update_mass();
        end

        function remove_component(obj, component_name)
            % Remove a component by name
            for i = length(obj.components):-1:1
                if strcmp(obj.components{i}.name, component_name)
                    obj.components(i) = [];
                    break;
                end
            end
            obj.update_mass();
        end

        function component = get_component(obj, component_name)
            % Get a component by name
            component = [];
            for i = 1:length(obj.components)
                if strcmp(obj.components{i}.name, component_name)
                    component = obj.components{i};
                    break;
                end
            end
        end

        function components = create_default_components()
            % Create default components: bus, two solar panels, and rotatable solar panel
            components = {};

            % Main bus: 2m x 1m x 1m, Nadir deck on X-Y plane, extending towards -Z
            % Bus extends from Z=0 (Nadir surface) to Z=-1 (towards Earth)
            bus = CuboidComponent([2.0, 1.0, 1.0], [0, 0, -0.5], 'Bus');
            components{end+1} = bus;

            % Solar panels: positioned at Y = ±0.5m on Nadir surface (Z=0)
            panel1 = CuboidComponent([2.0, 0.025, 1.0], [0,  0.5125, 0], 'SolarPanel+Y');
            panel2 = CuboidComponent([2.0, 0.025, 1.0], [0, -0.5125, 0], 'SolarPanel-Y');
            panel3 = CuboidComponent([2.0, 16.0, 0.01], [1, 0, -1.4], 'SolarPanelLarge');
            components{end+1} = panel1;
            components{end+1} = panel2;
            components{end+1} = panel3;

            % Large solar panel: at -X end, Z=0, 16m long along Y
            % Dimensions: 0.01m (X) x 16m (Y) x 2m (Z)
            % large_panel = CuboidComponent([2.0, 16.0, 0.01], [1.1, 0, 0], 'LargeSolarPanel');
            % components{end+1} = large_panel;

            % Rotatable solar panel: at +X end, Z=0, rotated 30 deg around Y
            % Dimensions: 0.01m (X) x 16m (Y) x 2m (Z)
            % rotatable_panel = CuboidComponent([0.01, 16.0, 2.0], [1.0, 0, 0], 'RotatableSolarPanel');
            % theta = 30 * pi / 180;  % 30 degrees
            % rotatable_panel.rotation_matrix = [
            %     cos(theta), 0, sin(theta);
            %     0, 1, 0;
            %     -sin(theta), 0, cos(theta)
            % ];
            % components{end+1} = rotatable_panel;
        end

        function print_info(obj)
            % Print satellite information
            fprintf('Ray Tracing Satellite: %s\n', obj.name);
            fprintf('Components: %d\n', length(obj.components));
            fprintf('Total Mass: %.2f kg\n', obj.mass);
            fprintf('Position: [%.2f, %.2f, %.2f] km\n', obj.position(1)/1000, obj.position(2)/1000, obj.position(3)/1000);
            fprintf('Velocity: [%.2f, %.2f, %.2f] m/s\n', obj.velocity(1), obj.velocity(2), obj.velocity(3));
            fprintf('Attitude quaternion: [%.4f, %.4f, %.4f, %.4f]\n', ...
                obj.attitude(1), obj.attitude(2), obj.attitude(3), obj.attitude(4));
            fprintf('Angular velocity: [%.4f, %.4f, %.4f] rad/s\n', ...
                obj.angular_velocity(1), obj.angular_velocity(2), obj.angular_velocity(3));
            fprintf('Body frame: Origin at middle of Nadir deck surface\n');
            fprintf('Body-X: Along long axis\n');
            fprintf('Body-Z: Out of Nadir deck\n');

            fprintf('\nComponents:\n');
            for i = 1:length(obj.components)
                fprintf('  %d. ', i);
                comp = obj.components{i};
                fprintf('Name: %s, Dimensions: [%.3f, %.3f, %.3f] m, Position: [%.3f, %.3f, %.3f] m\n', ...
                    comp.name, comp.dimensions(1), comp.dimensions(2), comp.dimensions(3), ...
                    comp.position(1), comp.position(2), comp.position(3));
            end
        end
        
        function vertices = get_vertices(obj)
            % Get all vertices from all components in body frame
            % Returns matrix of all vertices [x, y, z]

            if isempty(obj.components)
                vertices = [];
                return;
            end

            all_vertices = [];
            for i = 1:length(obj.components)
                comp_vertices = obj.components{i}.get_vertices();
                all_vertices = [all_vertices; comp_vertices];
            end
            vertices = all_vertices;
        end
        
        function faces = get_faces(obj)
            % Get all faces from all components
            % Returns matrix of face vertex indices (1-based)
            % Vertex indices are global across all components

            if isempty(obj.components)
                faces = [];
                return;
            end

            all_faces = [];
            vertex_offset = 0;
            for i = 1:length(obj.components)
                comp_faces = obj.components{i}.get_faces();
                % Adjust face indices by adding vertex offset
                adjusted_faces = comp_faces + vertex_offset;
                all_faces = [all_faces; adjusted_faces];
                vertex_offset = vertex_offset + 8;  % Each component has 8 vertices
            end
            faces = all_faces;
        end
        
        function normals = get_face_normals(obj)
            % Get outward normal vectors for each face from all components

            if isempty(obj.components)
                normals = [];
                return;
            end

            all_normals = [];
            for i = 1:length(obj.components)
                comp_normals = obj.components{i}.get_face_normals();
                all_normals = [all_normals; comp_normals];
            end
            normals = all_normals;
        end
        
        function [pos, vel] = propagate_orbit(obj, dt)
            % Simplified orbital propagation
            % For ray tracing, orbital motion may not be needed
            
            pos = obj.position;
            vel = obj.velocity;
        end
        
        function [attitude, omega] = propagate_attitude(obj, dt)
            % Simplified attitude propagation
            % For ray tracing, attitude dynamics may not be needed

            attitude = obj.attitude;
            omega = obj.angular_velocity;
        end

        function [area, cop] = compute_projected_area_and_cop(obj, direction)
            % Compute projected surface area and center of pressure from a given direction
            % Uses ray tracing to properly handle partial occlusion
            %
            % Input:
            %   direction - Vector [x, y, z] indicating the direction from which
            %               the satellite is viewed (e.g., sun direction)
            %
            % Output:
            %   area - Total projected surface area (m²) - accounting for occlusion
            %   cop  - Center of pressure position [x, y, z] in body frame (m)

            % Normalize direction vector
            direction = direction(:) / norm(direction);

            all_vertices = [];
            all_faces = [];
            all_normals = [];
            vertex_offset = 0;

            % Collect all vertices, faces, and normals from components
            for i = 1:length(obj.components)
                comp = obj.components{i};
                comp_vertices = comp.get_vertices();
                comp_faces = comp.get_faces();
                comp_normals = comp.get_face_normals();

                all_vertices = [all_vertices; comp_vertices];
                all_faces = [all_faces; comp_faces + vertex_offset];
                all_normals = [all_normals; comp_normals];

                vertex_offset = vertex_offset + 8;  % Each component has 8 vertices
            end

            % Process all front-facing faces
            visible_faces = [];
            for i = 1:size(all_faces, 1)
                face_normal = all_normals(i, :)';
                dot_prod = dot(face_normal, direction);

                % Only consider front-facing faces (dot product > 0)
                if dot_prod > 0
                    % Get vertices of this face
                    face_indices = all_faces(i, :);
                    face_vertices = all_vertices(face_indices, :);

                    % Calculate face area
                    v1 = face_vertices(2, :) - face_vertices(1, :);
                    v2 = face_vertices(4, :) - face_vertices(1, :);
                    cross_prod = cross(v1, v2);
                    face_area = 0.5 * norm(cross_prod);

                    % For quad, also calculate second triangle
                    v1 = face_vertices(3, :) - face_vertices(1, :);
                    v2 = face_vertices(4, :) - face_vertices(1, :);
                    cross_prod = cross(v1, v2);
                    face_area = face_area + 0.5 * norm(cross_prod);

                    % Project face area onto the view direction plane
                    projected_area = face_area * abs(dot_prod);

                    % Calculate face center
                    face_center = mean(face_vertices, 1)';

                    % Calculate visibility fraction using ray tracing from multiple samples
                    visibility_fraction = obj.calculate_visibility_fraction(face_center, ...
                                                                            face_vertices, ...
                                                                            all_faces, ...
                                                                            all_vertices, ...
                                                                            all_normals, ...
                                                                            direction, i);

                    % Calculate visible area
                    visible_area = projected_area * visibility_fraction;

                    % Debug output for face visibility analysis
                    fprintf('Face %d: proj_area=%.3f m²', i, projected_area);
                    fprintf(', visibility=%.4f, visible_area=%.3f m²\n', visibility_fraction, visible_area);

                    if visible_area > 1e-9  % Only include if some area is visible
                        visible_faces = [visible_faces; struct('area', visible_area, ...
                                                                'center', face_center)];
                    end
                end
            end

            % Calculate total projected area and center of pressure
            area = 0;
            cop = [0; 0; 0];

            for i = 1:length(visible_faces)
                face = visible_faces(i);
                area = area + face.area;
                cop = cop + face.area * face.center;
            end

            % Normalize center of pressure by total area
            if area > 0
                cop = cop / area;
            else
                cop = [0; 0; 0];
            end
        end

        function visibility_fraction = calculate_visibility_fraction(obj, face_center, face_vertices, ...
                                                                      all_faces, all_vertices, all_normals, ...
                                                                      view_direction, face_index)
            % Calculate visibility fraction by shooting rays from multiple points on the face
            % Uses Monte Carlo sampling across the face surface
            
            num_samples = 10000;  % 10000 rays per face (~100x100 grid) for higher accuracy
            visible_rays = 0;
            
            % Generate sample points across face (quad vertices 1-4)
            for i = 1:sqrt(num_samples)
                for j = 1:sqrt(num_samples)
                    % Normalized coordinates in [0, 1]
                    u = (i - 0.5) / sqrt(num_samples);
                    v = (j - 0.5) / sqrt(num_samples);
                    
                    % Bilinear interpolation on quad
                    % Sample point = (1-u)(1-v)*v0 + u(1-v)*v1 + u*v*v2 + (1-u)*v*v3
                    sample_point = (1-u)*(1-v)*face_vertices(1,:) + ...
                                   u*(1-v)*face_vertices(2,:) + ...
                                   u*v*face_vertices(3,:) + ...
                                   (1-u)*v*face_vertices(4,:);
                    
                    % Cast ray from the sample point on the face, looking backward
                    % This is the most direct approach: start at the surface and check for occlusion
                    ray_origin = sample_point;
                    ray_direction = -view_direction;  % Rays travel backward away from view direction
                    
                    % Find the closest intersection with any other front-facing face
                    % If an intersection exists, the ray is occluded
                    closest_distance = inf;
                    
                    for k = 1:size(all_faces, 1)
                        if k == face_index
                            continue;  % Skip self
                        end
                        
                        face_normal = all_normals(k, :)';
                        dot_prod = dot(face_normal, view_direction);
                        
                        % Only check against front-facing faces
                        if dot_prod > 0
                            face_verts = all_vertices(all_faces(k, :), :);
                            
                            % Test ray intersection with quad (two triangles)
                            [hit, dist] = obj.ray_triangle_intersection(ray_origin, ray_direction, ...
                                                                         face_verts(1, :), face_verts(2, :), face_verts(3, :), ...
                                                                         face_normal);
                            if hit && dist < closest_distance
                                closest_distance = dist;
                            end
                            
                            [hit, dist] = obj.ray_triangle_intersection(ray_origin, ray_direction, ...
                                                                         face_verts(1, :), face_verts(3, :), face_verts(4, :), ...
                                                                         face_normal);
                            if hit && dist < closest_distance
                                closest_distance = dist;
                            end
                        end
                    end
                    
                    % Ray is visible if no occlusion was found (no intersection)
                    if isinf(closest_distance)
                        visible_rays = visible_rays + 1;
                    end
                end
            end
            
            % Visibility fraction is the percentage of rays that didn't hit anything
            visibility_fraction = visible_rays / num_samples;
        end

        function [hit, distance] = ray_triangle_intersection(~, ray_origin, ray_direction, v0, v1, v2, face_normal)
            % Möller-Trumbore ray-triangle intersection algorithm
            % Also checks that the intersection is on the front side of the triangle
            
            epsilon = 1e-8;
            
            % Convert to column vectors if needed
            ray_origin = ray_origin(:);
            ray_direction = ray_direction(:) / norm(ray_direction);
            v0 = v0(:);
            v1 = v1(:);
            v2 = v2(:);
            
            edge1 = v1 - v0;
            edge2 = v2 - v0;
            
            h = cross(ray_direction, edge2);
            a = dot(edge1, h);
            
            if abs(a) < epsilon
                hit = false;
                distance = inf;
                return;
            end
            
            f = 1.0 / a;
            s = ray_origin - v0;
            u = f * dot(s, h);
            
            if u < 0.0 || u > 1.0
                hit = false;
                distance = inf;
                return;
            end
            
            q = cross(s, edge1);
            v = f * dot(ray_direction, q);
            
            if v < 0.0 || u + v > 1.0
                hit = false;
                distance = inf;
                return;
            end
            
            distance = f * dot(edge2, q);
            
            % Check that we're hitting the front of the triangle
            if distance > epsilon
                % Compute triangle normal
                tri_normal = cross(edge1, edge2);
                tri_normal = tri_normal / norm(tri_normal);
                
                % Only count as hit if ray is hitting from the correct side
                % (ray approaching the triangle: dot(ray_direction, tri_normal) < 0)
                if dot(ray_direction, tri_normal) < 0
                    hit = true;
                else
                    hit = false;
                    distance = inf;
                end
            else
                hit = false;
                distance = inf;
            end
        end

        function plot_satellite(obj, varargin)
            % Visualize the satellite as a 3D rectangular prism
            % Optional input: projection_direction [x, y, z]
            %
            % Usage:
            %   plot_satellite()                          % No projection
            %   plot_satellite([4.5, 0, 1])              % Show projection direction

            % Parse optional direction input
            projection_direction = [];
            if nargin > 1
                projection_direction = varargin{1}(:);
                projection_direction = projection_direction / norm(projection_direction);
            end

            % Create figure if none exists
            figure;
            hold on;

            % Define colors for different components
            colors = {'red', 'green', 'blue', 'yellow', 'cyan', 'magenta'};

            for i = 1:length(obj.components)
                comp = obj.components{i};
                comp_vertices = comp.get_vertices();
                comp_faces = comp.get_faces();

                % Plot each component with different color
                color_idx = mod(i-1, length(colors)) + 1;
                patch('Vertices', comp_vertices, 'Faces', comp_faces, ...
                      'FaceColor', colors{color_idx}, 'EdgeColor', 'black', 'FaceAlpha', 0.8);

                % Plot vertices as points for debugging
                plot3(comp_vertices(:,1), comp_vertices(:,2), comp_vertices(:,3), ...
                      'o', 'Color', colors{color_idx}, 'MarkerSize', 2, 'MarkerFaceColor', colors{color_idx});
            end

            % Plot projection direction if provided
            if ~isempty(projection_direction)
                % Calculate scale for visualization
                all_vertices = obj.get_vertices();
                max_dim = max(max(abs(all_vertices)));
                arrow_length = 2 * max_dim;

                % Draw arrow from center in projection direction
                quiver3(0, 0, 0, projection_direction(1)*arrow_length, ...
                        projection_direction(2)*arrow_length, projection_direction(3)*arrow_length, ...
                        'Color', 'magenta', 'LineWidth', 2.5, 'MaxHeadSize', 0.8, ...
                        'AlignVertexCenters', 'off');

                % Calculate and plot center of pressure
                [area, cop] = obj.compute_projected_area_and_cop(projection_direction);
                plot3(cop(1), cop(2), cop(3), 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm');
                text(cop(1), cop(2), cop(3), sprintf('  CoP\n  Area: %.3f m²', area), ...
                     'Color', 'magenta', 'FontSize', 9, 'FontWeight', 'bold');

                % Draw line from origin to CoP
                plot3([0, cop(1)], [0, cop(2)], [0, cop(3)], 'm--', 'LineWidth', 1.5);
            end

            % Add labels and formatting
            xlabel('X (m) - Body Frame');
            ylabel('Y (m) - Body Frame');
            zlabel('Z (m) - Body Frame');
            title(['Ray Tracing Satellite: ', obj.name]);
            grid on;
            axis equal;

            % Add coordinate frame arrows
            quiver3(0, 0, 0, 1.5, 0, 0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);  % X-axis (red)
            quiver3(0, 0, 0, 0, 1.5, 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);  % Y-axis (green)
            quiver3(0, 0, 0, 0, 0, 1.5, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);  % Z-axis (blue)

            % Add text labels for axes
            text(1.5, 0, 0, 'Body-X', 'Color', 'r', 'FontSize', 10);
            text(0, 0.75, 0, 'Body-Y', 'Color', 'g', 'FontSize', 10);
            text(0, 0, 0.75, 'Body-Z', 'Color', 'b', 'FontSize', 10);

            % Mark body frame origin
            plot3(0, 0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
            text(0.1, 0.1, 0.1, 'Body Frame Origin', 'FontSize', 8);

            hold off;

            % Set view for good visualization
            view([1,-1,1]);
            set(gca, 'ZDir', 'reverse');
        end
    end
    
    % Static methods for creating standard configurations
    methods (Static)
        function satellite = create_standard_satellite()
            % Create satellite according to specification
            % Bus: 2m x 1m x 1m with solar panels
            % Nadir deck on X-Y plane, extending towards negative Z
            % Body frame at middle of Nadir deck surface
            % Body-X along long axis (2m)
            % Body-Z out of Nadir deck (positive Z)

            % Create components
            components = {};

            % Main bus: 2m x 1m x 1m, Nadir deck on X-Y plane, extending towards -Z
            bus = CuboidComponent([2.0, 1.0, 1.0], [0, 0, -0.5], 'Bus');
            components{end+1} = bus;

            % Solar panels: positioned at Y = ±0.5m on Nadir surface (Z=0)
            panel1 = CuboidComponent([2.0, 1.0, 0.01], [0,  1.1, 0], 'SolarPanel+Y');
            panel2 = CuboidComponent([2.0, 1.0, 0.01], [0, -1.1, 0], 'SolarPanel-Y');
            panel3 = CuboidComponent([2.0, 16.0, 0.01], [1, 0, -1.4], 'SolarPanelLarge');
            components{end+1} = panel1;
            components{end+1} = panel2;
            components{end+1} = panel3;

            % Orbital parameters
            position = [6778137; 0; 0];  % LEO
            velocity = [0; 7669.2; 0];
            attitude = [1; 0; 0; 0];
            angular_velocity = [0; 0; 0];

            satellite = RayTracingSatellite(components, position, velocity, attitude, angular_velocity, 'SpecSatellite');
        end
    end
end
