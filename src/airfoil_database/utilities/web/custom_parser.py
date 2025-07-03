import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_sections(file_content):
    """
    Parses a string containing file content and separates it into sections
    based on empty lines.
    
    Args:
        file_content: A string containing the content of the file.
    
    Returns:
        A list of strings, where each string represents a section of the file.
    """
    sections = []
    current_section = []
    
    for line in file_content.strip().split('\n'):
        if line.strip() == "":
            if current_section:  # Only append non-empty sections
                sections.append('\n'.join(current_section))
                current_section = []
        else:
            current_section.append(line)
            
    # Don't forget the last section if the file doesn't end with an empty line
    if current_section:
        sections.append('\n'.join(current_section))
        
    return sections

def parse_coordinates(text):
    """
    Parse coordinates from text, handling various formats.
    
    Args:
        text: String containing coordinate data
        
    Returns:
        numpy.ndarray of parsed coordinates
    """
    coordinates = []
    
    for line in text.splitlines():
        values = load_line(line)
        if values:
            coordinates.append(values)
    
    return np.array(coordinates) if coordinates else np.empty((0, 2))

def load_line(line):
    """
    Parse a line of text into X,Y coordinates, with validation.
    
    Args:
        line: String containing a single line of coordinate data
        
    Returns:
        List of [x, y] coordinates or None if invalid
    """
    split_line = line.split()
    
    # Skip lines that don't have exactly 2 values
    if len(split_line) != 2:
        return None
    
    try:
        # Clean and convert values
        values = []
        for value in split_line:
            # Remove parentheses if present
            cleaned_value = value.replace('(', '').replace(')', '')
            val = float(cleaned_value)
            
            # Validate coordinate is in expected range
            if abs(val) >= 2:
                return None
                
            values.append(val)
        
        return values
    except ValueError:
        # Skip lines with non-numeric data
        return None

def parse_file(filename):
    """
    Parse an airfoil data file into a pointcloud array.
    
    Args:
        filename: Path to the airfoil data file
        
    Returns:
        numpy.ndarray of pointcloud coordinates
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        logger.info(f'Loading: {filename}')
        sections = parse_sections(content)
        
        if len(sections) == 1:
            # Single section format - title line followed by coordinates
            lines = sections[0].splitlines()
            return parse_coordinates('\n'.join(lines[1:]))
            
        elif len(sections) == 2:
            # Two section format - typically description and coordinates
            description = sections[0].splitlines()[0]  # Save the description
            
            # Parse coordinates from both sections (some files have coordinates in both)
            coords1 = parse_coordinates('\n'.join(sections[0].splitlines()[1:]))
            coords2 = parse_coordinates(sections[1])
            
            # Combine if both have coordinates, otherwise use whichever has data
            if coords1.size > 0 and coords2.size > 0:
                return np.vstack((coords1, coords2))
            elif coords1.size > 0:
                return coords1
            else:
                return coords2
                
        elif len(sections) == 3:
            # Three section format - description and two coordinate sections
            description = sections[0]  # First section is description
            
            # Parse coordinates from the second and third sections
            coords1 = parse_coordinates(sections[1])
            coords2 = parse_coordinates(sections[2])
            
            # Combine coordinates, typically upper and lower surfaces
            if coords1.size > 0 and coords2.size > 0:
                return np.vstack((coords1, np.flipud(coords2)))
            elif coords1.size > 0:
                return coords1
            else:
                return coords2
        else:
            logger.error(f'Unexpected file format with {len(sections)} sections: {filename}')
            return np.empty((0, 2))
            
    except Exception as e:
        logger.error(f'Error loading {filename}: {str(e)}')
        return np.empty((0, 2))

def validate_pointcloud(pointcloud, min_points=4):
    """
    Validates a pointcloud array to ensure it has sufficient data.
    
    Args:
        pointcloud: numpy.ndarray of pointcloud coordinates
        min_points: Minimum number of points required for validity
        
    Returns:
        bool: True if valid, False otherwise
    """
    if pointcloud is None or not isinstance(pointcloud, np.ndarray):
        return False
        
    if pointcloud.size == 0 or pointcloud.shape[0] < min_points:
        return False
        
    return True

if __name__ == '__main__':
    folder = r"D:\Mitchell\School\2025 Winter\airfoil_database\airfoil_dat_files"
    airfoil = 'naca1'
    file = os.path.join(folder, airfoil + '.dat')
    
    pointcloud = parse_file(file)
    
    if validate_pointcloud(pointcloud):
        logger.info(f"Successfully parsed {airfoil} with {len(pointcloud)} points")
        print(pointcloud)
    else:
        logger.warning(f"Pointcloud validation failed for {airfoil}")