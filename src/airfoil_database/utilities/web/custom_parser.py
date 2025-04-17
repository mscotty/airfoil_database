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
    
    print(f'LOADING: {filename}')
    sections = parse_sections(content)
    if len(sections) == 1:
        lines = sections[0].splitlines()
        pointcloud = []
        for line in lines[1::]:
            val = load_line(line)
            if val:
                pointcloud.append(val)
        pointcloud = np.array(pointcloud)
    elif len(sections) == 2:
        pointcloud = []
        for idx, section in enumerate(sections):
            lines = section.splitlines()
            if idx == 0:
                description = lines[0]
                for line in lines[1::]:
                    val = load_line(line)
                    if val:
                        pointcloud.append(val)
            elif idx == 1:
                lines = section.splitlines()
                pointcloud = []
                for line in lines:
                    val = load_line(line)
                    if val:
                        pointcloud.append(val)
        pointcloud = np.array(pointcloud)
    elif len(sections) == 3:
        for idx, section in enumerate(sections):
            if idx == 0:
                description = section
            elif idx == 1:
                lines = section.splitlines()
                pointcloud = []
                for line in lines:
                    val = load_line(line)
                    if val:
                        pointcloud.append(val)
            elif idx == 2:
                pointcloud2 = []
                for line in lines:
                    val = load_line(line)
                    if val:
                        pointcloud2.append(val)
                pointcloud = np.vstack((np.array(pointcloud), np.flipud(np.array(pointcloud2))))
    else:
        print(f'ERROR LOADING: {filename}')
    return pointcloud


def load_line(line):
    split_line = line.split()
    value = []
    if len(split_line) == 2:
        for split in split_line:
            try:
                split = split.replace('(', '').replace(')', '')
                val = float(split)
                if np.abs(val) < 2:
                    value.append(float(split))
                else:
                    return None
            except ValueError:
                return None
    
    return value
                


if __name__ == '__main__':
    import os
    folder = r"D:\Mitchell\School\2025 Winter\airfoil_database\airfoil_dat_files"
    #airfoil = 'fx72150b'
    airfoil = 'naca1'
    file = os.path.join(folder, airfoil+'.dat')
    pointcloud = parse_file(file)
    print(pointcloud)
