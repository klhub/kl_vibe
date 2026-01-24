% CuboidComponent Class
% Simple cuboid component for ray tracing satellite
%
% Properties:
%   dimensions - [length, width, height] in meters
%   position - Position relative to satellite center [x, y, z] (m)
%   rotation_matrix - 3x3 rotation matrix for component orientation
%   name - Component name

classdef CuboidComponent
    properties
        dimensions  % [length, width, height] in meters
        position    % Position relative to satellite center [x, y, z] (m)
        rotation_matrix  % 3x3 rotation matrix
        name        % Component name
    end

    methods
        function obj = CuboidComponent(dimensions, position, name)
            % Constructor for cuboid component
            if nargin < 3
                name = 'Component';
            end
            if nargin < 2
                position = [0, 0, 0];
            end
            if nargin < 1
                dimensions = [1, 1, 1];
            end

            obj.dimensions = dimensions;
            obj.position = position;
            obj.rotation_matrix = eye(3);  % Identity matrix, no rotation
            obj.name = name;
        end

        function vertices = get_vertices(obj)
            % Get 8 vertices of the cuboid in body frame
            % Returns vertices relative to component center

            L = obj.dimensions(1) / 2;
            W = obj.dimensions(2) / 2;
            H = obj.dimensions(3) / 2;

            % Base vertices at component center
            base_vertices = [
                -L, -W, -H;
                 L, -W, -H;
                 L,  W, -H;
                -L,  W, -H;
                -L, -W,  H;
                 L, -W,  H;
                 L,  W,  H;
                -L,  W,  H;
            ];

            % Apply rotation
            rotated_vertices = (obj.rotation_matrix * base_vertices')';

            % Translate to component position
            vertices = rotated_vertices + repmat(obj.position, 8, 1);
        end

        function faces = get_faces(obj)
            % Get 6 faces of the cuboid
            % Each face is a quad defined by 4 vertex indices

            faces = [
                1, 2, 3, 4;  % Bottom face
                5, 6, 7, 8;  % Top face
                1, 2, 6, 5;  % Front face
                2, 3, 7, 6;  % Right face
                3, 4, 8, 7;  % Back face
                4, 1, 5, 8;  % Left face
            ];
        end

        function normals = get_face_normals(obj)
            % Get outward normal vectors for each face

            base_normals = [
                0,  0, -1;  % Bottom
                0,  0,  1;  % Top
                0, -1,  0;  % Front
                1,  0,  0;  % Right
                0,  1,  0;  % Back
               -1,  0,  0;  % Left
            ];

            normals = (obj.rotation_matrix * base_normals')';
        end
    end
end
