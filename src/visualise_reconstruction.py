from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def create_cylinder_surface_mesh(index_to_coordinate, sample_array, colormap='plasma'):
    """
    Create a triangular mesh on the surface of a cylinder defined by the input points,
    with smooth interpolation of the colors across the mesh and complete closure.
    
    Args:
        index_to_coordinate (dict): A dictionary with indices as keys and 3D coordinates (x, y, z) as values.
        sample_array (list or np.ndarray): Values corresponding to the conductivity at each point.
        colormap (str): Colormap to use for visualization.
    """
    # Extract points and corresponding values from the dictionary
    indices = sorted(index_to_coordinate.keys())  # Ensure consistent ordering of indices
    points = np.array([index_to_coordinate[i] for i in indices])  # 3D coordinates (x, y, z)
    values = np.array([sample_array[i] for i in indices])  # Corresponding predicted values

    # Convert points to cylindrical coordinates (theta, z)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    theta = np.arctan2(y, x)  # Angular coordinate

    # Wrap points at -π and +π to close the cylinder
    extended_points = np.vstack([points, points])  # Duplicate the points
    extended_theta = np.concatenate([theta, theta + 2 * np.pi])  # Wrap around theta
    extended_z = np.concatenate([z, z])  # Duplicate z
    extended_values = np.concatenate([values, values])  # Duplicate values

    cylindrical_coords = np.column_stack((extended_theta, extended_z))  # 2D unwrapped coordinates

    # Perform 2D Delaunay triangulation on the unwrapped surface
    tri = Delaunay(cylindrical_coords)

    # Create a colormap instance
    cmap = plt.cm.get_cmap(colormap)
    norm = plt.Normalize(values.min(), values.max())  # Normalize values for color mapping

    # Create a 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the triangles with smooth interpolation
    for simplex in tri.simplices:
        triangle = extended_points[simplex]  # Vertices of the triangle
        # Interpolate the colors based on the conductivity values at the vertices
        verts = [list(zip(triangle[:, 0], triangle[:, 1], triangle[:, 2]))]  # 3D triangle vertices
        face_color = cmap(norm(extended_values[simplex].mean()))
        poly = Poly3DCollection(verts, facecolor=face_color, edgecolor='k', alpha=0.8)
        ax.add_collection3d(poly)

    # Scatter plot of original points with colors determined by predicted values
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap=colormap, edgecolor='k', s=50)
    fig.colorbar(sc, ax=ax, label='Conductivity')

    # Set plot details
    ax.set_title('Triangular Mesh on Cylinder Surface with Interpolated Colors')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set x-axis limits to -0.3 to 0.3
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])

    plt.show()



def create_triangular_mesh(sample_array, width, height, index_to_coordinate, colormap='plasma'):
    """
    Create a triangular mesh and interpolate the conductivity distribution
    between points to visualize it as a heatmap.
    
    Args:
        array (list or numpy array): Values corresponding to the conductivity at each point.
        index_to_coord (dict): A dictionary mapping indices to coordinates (x, y).
        width (int): Width of the output image.
        height (int): Height of the output image.
        colormap (str): Colormap to use for visualization.
    """
    # Extract coordinates and values
    points = np.array(list(index_to_coordinate.values()))  # Coordinates (x, y) of the points
    values = np.array(sample_array)  # Conductivity values at each point

    # Perform Delaunay triangulation
    tri = Delaunay(points)

    # Create a grid for interpolation
    x = np.linspace(points[:, 0].min(), points[:, 0].max(), width)
    y = np.linspace(points[:, 1].min(), points[:, 1].max(), height)
    grid_x, grid_y = np.meshgrid(x, y)

    # Interpolate the values using griddata
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

    # Plot the triangulated mesh with the interpolated heatmap
    plt.figure(figsize=(8, 6))
    plt.tricontourf(points[:, 0], points[:, 1], tri.simplices, values, cmap=colormap, alpha=0.7)
    plt.imshow(grid_z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap=colormap, alpha=0.9)
    plt.colorbar(label='Conductivity')
    plt.scatter(points[:, 0], points[:, 1], c='red', edgecolor='black', label='Sampling Points')  # Original points
    plt.legend()
    plt.title('Triangular Mesh with Interpolated Conductivity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
