## Programming
- Use Matlab.
- Clean and consise code.
- Good comment.

## Spacecraft
- Bus dimension is 1m X 1m X 2m.
- Body frame is at the middle of the surface of the Nadir deck.
- Body-X is along the long axis.
- Body-Z is out of the Nadir deck.


New Features Added
1. compute_projected_area_and_cop() Method
Accepts a direction vector (e.g., [4.5, 0, 1])
Calculates the total projected surface area visible from that direction
Computes the center of pressure (CoP) - the weighted center of all visible surfaces
Only considers faces that are facing towards the direction (dot product > 0)
2. Enhanced plot_satellite() Method
Now accepts an optional projection direction parameter
Visualizes:
Magenta arrow: Shows the projection direction from the satellite origin
Magenta dot: Marks the center of pressure location
Dashed magenta line: Connects origin to the center of pressure
Text label: Displays the projected area value at the CoP
3. Updated Test Script
Demonstrates the new functionality with direction [4.5, 0, 1]
Outputs the normalized direction vector, projected area, and CoP coordinates
Creates two visualizations: one basic and one with the projection direction overlay
How It Works
The algorithm:

Normalizes the projection direction
Iterates through all faces of all components
Checks if each face is visible from that direction (using dot product with face normal)
Projects visible face areas onto the view plane
Calculates the weighted center of pressure based on projected areas
You can now call it directly in MATLAB:

[area, cop] = satellite.compute_projected_area_and_cop([4.5, 0, 1]);
satellite.plot_satellite([4.5, 0, 1]);

