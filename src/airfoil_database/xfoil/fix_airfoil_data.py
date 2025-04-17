import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def normalize_data(data):
    # Convert the list of points into a numpy array for easier manipulation
    data = np.array(data)
    
    # Separate the x and y coordinates
    x = data[:, 0]
    y = data[:, 1]
    
    # Find where x crosses 0 and make necessary adjustments
    min_x = np.min(x)
    max_x = np.max(x)
    
    # Normalize x values so that they start at 0 and end at 1
    x_normalized = (x - min_x) / (max_x - min_x)
    
    # Now we want to adjust the y-values, but retain the airfoil shape
    y_normalized = y  # If needed, we could apply a transformation here, but it's not always necessary
    
    # Create a new data array
    fixed_data = np.column_stack((x_normalized, y_normalized))
    
    return fixed_data

def normalize_pointcloud(pointcloud):
    """
    Normalizes a point cloud based on the distance between the leading edge (LE)
    and trailing edge (TE), scaling all other points accordingly.
    """
    points = np.array([[float(p[0]), float(p[1])] for p in pointcloud])
    if len(points) < 2:  # Need at least two points for LE and TE
        return np.array([])

    le_index = np.argmin(points[:, 0])
    te_index = np.argmax(points[:, 0])

    le_point = points[le_index]
    te_point = points[te_index]

    le_te_distance = np.linalg.norm(te_point - le_point)

    if le_te_distance == 0:
        return np.array([])  # Avoid division by zero

    # Translate to LE as origin
    translated_points = points - le_point

    # Scale based on LE-TE distance
    normalized_points = translated_points / le_te_distance

    return normalized_points

def reorder_airfoil_data_bad(data):
    # Convert the data into a numpy array for easier manipulation
    data = np.array(data)
    x = data[:,0]
    y = data[:,1]
    
    # Find the leading edge (LE) and trailing edge (TE) by their x-values
    LE_index = np.argmin(x)  # Index of the LE (smallest x)
    TE_index = np.argmax(x)  # Index of the TE (largest x)
    
    # Separate the data into two parts: suction surface (top) and bottom surface
    # top_surface = data[LE_index:TE_index + 1]  # Points from LE to TE
    # bottom_surface = data[TE_index:][::-1]  # Points from TE to LE, reversed
    flip = None
    for idx, val in enumerate(x):
        if idx+3 > len(x):
            continue
        if np.abs(val - x[idx+1]) > np.abs(val - x[idx+2]):
            flip = idx
            break

    if flip is not None:
        data_split_x = [x[0:flip+1], x[flip+1:]]
        ordered_x = np.append(data_split_x[0], np.flip(data_split_x[1]))
        data_split_y = [y[0:flip+1], y[flip+1:]]
        ordered_y = np.append(data_split_y[0], np.flip(data_split_y[1]))
    else:
        ordered_x = x
        ordered_y = y
    # Combine the surfaces to form the correct order
    #ordered_data = np.vstack((top_surface, bottom_surface))
    #ordered_data = np.roll(data, LE_index-1)
    ordered_data = np.stack((ordered_x,ordered_y), axis=1)
    
    return ordered_data

def reorder_airfoil_data(data):
    """Reorders airfoil data with precise trailing edge handling."""
    data = np.array(data)
    if len(data) < 3:
        return data

    LE_index = np.argmin(data[:, 0])
    LE = data[LE_index]
    TE_index = np.argmax(data[:, 0])
    TE = data[TE_index]

    # Find transition point
    transition_index = -1
    for i in range(TE_index, len(data) - 1):
        if data[i + 1, 0] < data[i, 0]:
            transition_index = i
            break

    if transition_index == -1:
        transition_index = TE_index

    # Split into surfaces
    upper = data[LE_index:transition_index + 1]
    lower = np.concatenate((data[transition_index + 1:], data[:LE_index]))

    # Sort by angle
    def sort_by_angle(surface):
        angles = np.arctan2(surface[:, 1] - LE[1], surface[:, 0] - LE[0])
        sorted_indices = np.argsort(angles)
        return surface[sorted_indices]

    upper = sort_by_angle(upper)
    lower = sort_by_angle(lower)

    # Reverse lower surface
    lower = np.flip(lower, axis=0)

    # Concatenate without overlap
    ordered_data = np.concatenate((upper, lower), axis=0)
    return ordered_data

def close_airfoil_data(data):
    """Ensures that the airfoil data is self-closing by adding the first point to the end."""
    data = np.array(data)
    if len(data) < 3:
        return data  # Not enough points to close

    if not np.allclose(data[0], data[-1]):
        return np.vstack((data, data[0]))  # Add the first point to the end
    return data  # Already closed

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv(r'C:\Users\fluff\AppData\Local\Temp\tmpxekbmff6.dat', sep=' ')
    data = np.array([df.iloc[:,0], df.iloc[:,1]])
    data = np.transpose(data)
    # Plot to verify
    plt.plot(data[:, 0], data[:, 1])
    plt.title("Fixed Airfoil Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-1, 1)
    plt.xlim(-0.1, 1.1)
    plt.grid(True)
    plt.show()

    fixed_data = reorder_airfoil_data(data)
    np.savetxt(r'D:\Mitchell\School\2025 Winter\DASC500\retry_airfoil.txt', fixed_data, delimiter=',')

    # Plot to verify
    plt.plot(fixed_data[:, 0], fixed_data[:, 1])
    plt.title("Fixed Airfoil Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-1, 1)
    plt.xlim(-0.1, 1.1)
    plt.grid(True)
    plt.show()