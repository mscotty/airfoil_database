import os
import matplotlib.pyplot as plt

from DASC500.classes.AirfoilDatabase import AirfoilDatabase
from DASC500.classes.XFoilRunner import XFoilRunner
from DASC500.classes.DataAnalysis import DataAnalysis
from DASC500.utilities.get_top_level_module import get_top_level_module_path

output_folder = os.path.join(get_top_level_module_path(), '../../outputs/project')
def test():
    db = AirfoilDatabase(db_dir="my_airfoil_database")
    df = db.get_aero_coeffs('aquilasm')
    print(df)

def test_store_aero_coeffs():
    db = AirfoilDatabase(db_dir="my_airfoil_database")
    db.store_aero_coeffs("test_airfoil", 100000, 0.1, 0, 1.0, 0.01, 0.0, 0.1)
    results = db.get_aero_coeffs("test_airfoil")
    print(f"Test results: {results}")

def add_aero_data_to_database():
    db = AirfoilDatabase(db_dir="my_airfoil_database")
    xfoil = XFoilRunner("D:/Mitchell/software/CFD/xfoil.exe")

    alpha_start=0
    alpha_end=8
    alpha_increment=4
    mach_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Define Mach numbers
    reynolds_list = [1000, 5000, 10000, 25000, 50000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000, 1500000, 2000000]  # Define Reynolds numbers

    db.run_all_airfoils(xfoil, max_workers=1, mach_list=mach_list, reynolds_list=reynolds_list, alpha_start=alpha_start, alpha_end=alpha_end, alpha_increment=alpha_increment)
    db.close()

def plot_data():
    db = AirfoilDatabase(db_dir="my_airfoil_database")
    #db.plot_airfoil('goe229', output_dir=output_folder)
    db.plot_airfoil('goe229', output_dir=output_folder, output_name='goe229_bad.png')
    #db.plot_airfoil('hq1511', output_dir=output_folder)
    #db.plot_multiple_airfoils(['goe229', 'hq1511'], output_dir=output_folder)
    db.close()

def stat_analysis_on_airfoil():
    db = AirfoilDatabase(db_dir="my_airfoil_database")
    df = db.get_airfoil_dataframe()
    da = DataAnalysis(dataframe=df)
    da.calculate_stats()
    output_file = os.path.join(output_folder, 'airfoil_stats.txt')
    da.print_stats(file=output_file)
    da.plot_histograms_per_col(key_in='Num_Points', 
                               binning_method='Square Root',
                               title_name='Airfoil # of Points in Point Cloud',
                               y_axis_name='Frequency',
                               x_axis_name='# of Points in Point Cloud',
                               output_dir=output_folder)
    db.plot_airfoil_series_pie(output_dir=output_folder)

def perform_airfoil_validity_checks():
    db = AirfoilDatabase(db_dir="my_airfoil_database")
    db.check_airfoil_validity()

def stat_analysis_on_airfoil_geom():
    db = AirfoilDatabase(db_dir="my_airfoil_database")
    df = db.get_airfoil_geometry_dataframe()
    df.drop(columns=['leading_edge_radius', 'trailing_edge_angle'], inplace=True)
    da = DataAnalysis(dataframe=df)
    da.calculate_stats()
    output_file = os.path.join(output_folder, 'airfoil_geom_stats.txt')
    da.print_stats(file=output_file)
    da.plot_histograms_per_col(key_in='aspect_ratio', 
                               binning_method='Square Root', 
                               title_name='Histogram of Airfoil Calculated AR',
                               x_axis_name='Airfoil Calculated AR',
                               y_axis_name='# of Airfoils',
                               output_dir=output_folder)
    da.plot_histograms_per_col(key_in='thickness_to_chord_ratio', 
                               binning_method='Square Root', 
                               title_name='Histogram of Airfoil Calculated t/c',
                               x_axis_name='Airfoil Calculated t/c',
                               y_axis_name='# of Airfoils',
                               output_dir=output_folder)
    da.plot_histograms_per_col(key_in='max_camber', 
                               binning_method='Square Root', 
                               title_name='Histogram of Airfoil Calculated Max Camber',
                               x_axis_name='Airfoil Calculated Max Camber',
                               x_axis_range=[0, 0.3],
                               y_axis_name='# of Airfoils',
                               output_dir=output_folder)
    da.plot_histograms_per_col(key_in='max_thickness', 
                               binning_method='Square Root', 
                               title_name='Histogram of Airfoil Calculated Max Thickness',
                               x_axis_name='Airfoil Calculated Max Thickness',
                               y_axis_name='# of Airfoils',
                               output_dir=output_folder)
    da.plot_histograms_per_col(key_in='span', 
                               binning_method='Square Root', 
                               title_name='Histogram of Airfoil Calculated Span',
                               x_axis_name='Airfoil Calculated Span',
                               y_axis_name='# of Airfoils',
                               output_dir=output_folder)
    

if __name__ == "__main__":
    # plot_data()
    db = AirfoilDatabase(db_dir="my_airfoil_database")
    db.plot_airfoil_series_horizontal_bar(output_dir=output_folder, output_name='airfoil_series_horizontal_bar.png')
    #db.plot_airfoil('ag12')
    #data = db.get_airfoil_data('ag12')
    #print(data[1])
    #db.fix_all_airfoils()
    #db.update_airfoil_series()
    #db.compute_geometry_metrics()
    #data = db.get_airfoil_data('ag12')
    #print(data[1])
    #ax1 = db.plot_airfoil('ag12')
    #ax2 = db.plot_airfoil('whitcomb')
    #ax3 = db.plot_airfoil('wb140')
    #plt.show()
    # data = db.get_airfoil_data('e377')
    #db.plot_airfoil('naca4412', output_dir=output_folder)
    #db.plot_airfoil('n12', output_dir=output_folder)
    #db.plot_airfoil('fx63137', output_dir=output_folder)
    #plt.show()
    # print(data[0])
    stat_analysis_on_airfoil_geom()
    # print(data[1])
    #db.output_pointcloud_to_file('fx63137', r'D:\Mitchell\School\airfoils\fx63137\fx63137.txt')
