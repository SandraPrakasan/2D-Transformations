import numpy as np
import matplotlib.pyplot as plt

def get_triangle():
    """Function to take user input for triangle vertices"""
    print("Enter coordinates of the triangle vertices (x, y):")
    points = []
    for i in range(3):
        x, y = map(float, input(f"Vertex {i+1}: ").split())
        points.append([x, y, 1])  # Convert to homogeneous coordinates
    return np.array(points).T  # Return as a 3x3 matrix

def get_transformation():
    """Function to take transformation parameters from user"""
    Tx, Ty = map(float, input("Enter translation (Tx, Ty): ").split())
    Sx, Sy = map(float, input("Enter scaling factors (Sx, Sy): ").split())
    theta = float(input("Enter rotation angle (in degrees): "))
    return Tx, Ty, Sx, Sy, np.radians(theta)  # Convert degrees to radians

def transformation_matrix(Tx, Ty, Sx, Sy, theta):
    """Generate composite transformation matrix"""
    # Translation Matrix
    T = np.array([
        [1, 0, Tx],
        [0, 1, Ty],
        [0, 0,  1]
    ])

    # Scaling Matrix
    S = np.array([
        [Sx, 0,  0],
        [0, Sy,  0],
        [0,  0,  1]
    ])

    # Rotation Matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Composite Transformation: T * R * S
    return T @ R @ S  # Matrix multiplication (T * R * S)

def plot_triangle(original, transformed):
    """Function to plot the original and transformed triangle"""
    plt.figure(figsize=(8, 6))

    # Extract x and y coordinates
    orig_x, orig_y = original[0], original[1]
    trans_x, trans_y = transformed[0], transformed[1]

    # Close the triangle (connect the first and last points)
    orig_x = np.append(orig_x, orig_x[0])
    orig_y = np.append(orig_y, orig_y[0])
    trans_x = np.append(trans_x, trans_x[0])
    trans_y = np.append(trans_y, trans_y[0])

    # Plot original triangle
    plt.plot(orig_x, orig_y, 'bo-', label="Original Triangle")

    # Plot transformed triangle
    plt.plot(trans_x, trans_y, 'ro-', label="Transformed Triangle")

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title("2D Composite Transformation")
    plt.show()

# Main Execution
triangle = get_triangle()  # Get original triangle
Tx, Ty, Sx, Sy, theta = get_transformation()  # Get transformation parameters
T_matrix = transformation_matrix(Tx, Ty, Sx, Sy, theta)  # Compute transformation matrix
transformed_triangle = T_matrix @ triangle  # Apply transformation

# Display results
print("Original Triangle Points:\n", triangle.T)
print("Transformed Triangle Points:\n", transformed_triangle.T)

# Plot the result
plot_triangle(triangle, transformed_triangle)

