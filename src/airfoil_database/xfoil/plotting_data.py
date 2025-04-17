import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_xfoil_polar(filepath):
    """Parses an XFOIL .pol file and extracts alpha, Cl, Cd, Cdp, and Cm."""
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        start_idx = None
        for i, line in enumerate(lines):
            if 'alpha' in line and 'CL' in line:  # Header row
                start_idx = i + 2  # Data starts two lines below
                break
        
        if start_idx is None:
            raise ValueError(f"No valid data found in {filepath}")
        
        for line in lines[start_idx:]:
            values = line.split()
            if len(values) >= 5:
                data.append([float(v) for v in values[:5]])
    
    return np.array(data)  # (alpha, Cl, Cd, Cdp, Cm)

def extract_reynolds(filename):
    """Extracts the Reynolds number from the filename (format: re*_n9_inviscid.pol)."""
    match = re.search(r're(\d+)', filename)
    return int(match.group(1)) if match else None

def plot_polar_data(folder, save_folder):
    """Loads all .pol files in a folder and saves Cl, Cd, Cdp, and Cm vs. alpha plots."""
    pol_files = [f for f in os.listdir(folder) if f.endswith('.pol')]
    
    data_by_re = {}
    for file in pol_files:
        re_num = extract_reynolds(file)
        if re_num:
            data_by_re[re_num] = parse_xfoil_polar(os.path.join(folder, file))
    
    if not data_by_re:
        print("No valid .pol files found.")
        return
    
    labels = {"Cl": 1, "Cd": 2, "Cdp": 3, "Cm": 4}
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Different colors for each Re
    
    os.makedirs(save_folder, exist_ok=True)
    
    for param, idx in labels.items():
        plt.figure(figsize=(8, 6))
        for i, (re_num, data) in enumerate(sorted(data_by_re.items())):
            alpha, Cl, Cd, Cdp, Cm = data.T
            color = colors[i % len(colors)]
            plt.plot(alpha, data[:, idx], color, label=f'Re={re_num}')
        
        plt.xlabel("Angle of Attack (alpha)", fontsize=18)
        plt.ylabel(param, fontsize=18)
        plt.legend(fontsize=18)
        plt.grid()
        plt.title(f"{param} vs Alpha", fontsize=20)

        plt.xlim(-20, 20)
        if idx == 1:
            plt.ylim(-1.5,2)
        
        save_path = os.path.join(save_folder, f"alpha_vs_{param.lower()}.png")
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    folder = r'D:\Mitchell\School\airfoils\fx63137'
    plot_polar_data(folder, folder)
