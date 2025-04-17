import numpy as np
import io

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
    lines = file_content.strip().split('\n')
    for line in lines:
        if line.strip() == "":
            if current_section:
                sections.append("\n".join(current_section))
                current_section = []
        else:
            current_section.append(line)
    if current_section:
        sections.append("\n".join(current_section))
    return sections

def parse_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    sections = parse_sections(content)
    if len(sections) > 1:
        for idx, section in enumerate(sections):
            if idx == 1:
                pointcloud = np.loadtxt(io.StringIO(section))
            if idx == 2:
                pointcloud = np.vstack((pointcloud, np.flip(np.loadtxt(io.StringIO(section)))))
        print(pointcloud)
    return pointcloud


if __name__ == '__main__':
    import os
    folder = r"D:\Mitchell\School\2025 Winter\airfoil_database\airfoil_dat_files"
    airfoil = 'fx67k150'
    file = os.path.join(folder, airfoil+'.dat')
    pointcloud = parse_file(file)
