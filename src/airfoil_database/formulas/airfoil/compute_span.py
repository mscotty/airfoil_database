import math

# Calculate the total length of the airfoil
def calculate_span(points):
    total_length = 0.0
    for i in range(1, len(points)):
        dist = math.sqrt((points[i][0] - points[i - 1][0]) ** 2 + (points[i][1] - points[i - 1][1]) ** 2)
        total_length += dist
    return total_length