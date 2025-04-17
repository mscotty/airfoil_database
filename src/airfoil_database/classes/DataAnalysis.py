import os
from copy import deepcopy
import warnings

import pandas as pd
import numpy as np
import scipy.stats as stats


from airfoil_database.utilities.data_type.distinguish_data_types import distinguish_data_types
from airfoil_database.utilities.print.print_series_mode import print_series_mode
from airfoil_database.formulas.statistics.confidence_interval import calculate_confidence_interval
from airfoil_database.formulas.statistics.hypothesis_test import hypothesis_test

from airfoil_database.models.build_linear_regression_model import linear_regression_model
from airfoil_database.models.build_mult_linear_regression_model import multiple_linear_regression
from airfoil_database.models.build_parsimonious_regression_model import stepwise_parsimonious_regression

from airfoil_database.plotting.plot_histogram import plot_histogram
from airfoil_database.plotting.plot_stacked_bar_chart import plot_stacked_bar_chart_horizontal, plot_stacked_bar_chart_vertical
from airfoil_database.plotting.plot_clustered_bar_chart import plot_clustered_bar_chart_horizontal, plot_clustered_bar_chart_vertical
from airfoil_database.plotting.plot_individual_bar_charts import plot_individual_bar_charts
from airfoil_database.plotting.plot_line_chart import plot_line_chart
from airfoil_database.plotting.plot_heatmap import plot_heatmap
from airfoil_database.plotting.plot_radar_chart import plot_radar_chart
from airfoil_database.plotting.visualize_regression_models import visualize_regression_models


# -------------------------------------
# DataAnalysis Class Definition
# -------------------------------------
class DataAnalysis:
    def __init__(self, file=None, dataframe=None):
        """
        Initialize the DataAnalysis object by loading a CSV file 
        and calculating initial stats and numeric columns.
        """
        self.file = file
        if file is not None:
            self.df_original = pd.read_csv(file)
        elif file is None and dataframe is not None:
            self.df_original = dataframe
        else:
            raise ValueError('Need a valid file or dataframe.')
        self.df_test = None
        self.df = deepcopy(self.df_original)
        self.determine_numeric_col()
        self.calculate_stats()
    
    def downsample_dataframe(self, new_size=None, frac=None, random_state=None):
        """
        Downsample a DataFrame to a specified number of rows.

        Parameters:
            df (pd.DataFrame): The original DataFrame.
            new_size (int): The desired number of rows in the downsampled DataFrame.
            random_state (int, optional): Random seed for reproducibility.

        Returns:
            pd.DataFrame: The downsampled DataFrame.
        """
        if new_size is None and frac is None:
            raise ValueError("Must provide either a new_size or a frac to downsample data.")
        elif new_size is not None and frac is not None:
            raise ValueError("Cannot provide both new_size and frac, only one is supported.")
        if new_size is not None and new_size > len(self.df_original):
            warnings.warn(f"{new_size} provided is outside of the data's supported range: 1-{len(self.df_original)}")
            self.df = self.df_original.copy()  # Return original if the requested size is too large
        elif frac is not None and frac > 1:
            warnings.warn(f"INPUT:WARNING: fraction input was greater than supported, defaulting to 1")
            self.df = self.df_original.copy() # Return original if the requested size is too large
        
        if frac is None:
            self.df = self.df_original.sample(n=new_size, random_state=random_state)
        elif new_size is None:
            self.df = self.df_original.sample(frac=frac, random_state=random_state)
        
        self.df_test = self.df_original.drop(self.df.index)

    def determine_numeric_col(self):
        """
        Identify numeric columns in the DataFrame and store them.
        """
        self.col_types = distinguish_data_types(self.df)
        col_types = np.array(list(self.col_types.values()))
        headers = np.array(self.df.columns)
        num_headers = headers[col_types == 'Numeric']
        self.num_headers = {header: {} for header in num_headers}
    
    def calculate_stats(self):
        """
        Calculate and store statistics (mean, median, variance, etc.)
        for numeric columns in the DataFrame.
        """
        for key, value in self.num_headers.items():
            value['mean'] = self.df[key].mean()
            value['median'] = self.df[key].median()
            value['mode'] = self.df[key].mode(dropna=True)
            value['pop_variance'] = self.df[key].var(ddof=0)
            value['pop_std'] = self.df[key].std(ddof=0)
            value['sample_variance'] = self.df[key].var()
            value['sample_std'] = self.df[key].std()
            value['first_quartile'] = self.df[key].quantile(0.25)
            value['third_quartile'] = self.df[key].quantile(0.75)
    
    def print_stats(self, file=None):
        """!
        @brief Print or save statistics of numeric columns.
        Args:
        - file (str): File path to save stats. If None, prints to console.
        """
        for key, value in self.num_headers.items():
            # Build the string with all metrics
            stats_string = (
                f"Calculated metrics for {key}\n"
                f"Mean: {value['mean']}\n"
                f"Median: {value['median']}\n"
                f"Mode:\n{print_series_mode(value['mode'])}\n"
                f"Population Variance: {value['pop_variance']}\n"
                f"Population Standard Deviation: {value['pop_std']}\n"
                f"Sample Variance: {value['sample_variance']}\n"
                f"Sample Standard Deviation: {value['sample_std']}\n"
                f"First quartile: {value['first_quartile']}\n"
                f"Third quartile: {value['third_quartile']}\n"
            )

            # Print or write the string based on the `file` argument
            if file is None:
                print(stats_string)
            else:
                with open(file, 'a+') as f:
                    f.write(stats_string)
            
    
    def calculate_pearson_corr_coeff(self, 
                                     col1_name, 
                                     col2_name):
        """!
        @brief Calculate the Pearson correlation coefficient between two columns.
        Args:
        - col1_name (str): First column name.
        - col2_name (str): Second column name.
        """
        return self.df[col1_name].corr(self.df[col2_name])

    def confidence_intervals(self, confidence=0.95):
        """
        Compute confidence intervals for the mean and variance of each numerical column in a Pandas DataFrame.

        Parameters:
            confidence (float): Confidence level (default 0.95 for 95%).

        Returns:
            dict: Confidence intervals for mean and variance of each column.
        """
        conf_interval = {}

        for column in self.df.select_dtypes(include=[np.number]):  # Process only numerical columns
            result = calculate_confidence_interval(self.df[column], confidence=confidence)

            # Store results
            conf_interval[column] = result
        
        self.conf_interval = conf_interval
    
    def print_confidence_intervals(self, file=None, col_names=None):
        """!
        @brief Print or save confidence intervals of numeric columns.
        Args:
        - file (str): File path to save confidence intervals. If None, prints to console.
        """
        if col_names is None:
            col_names = self.df.select_dtypes(include=[np.number])

        for col, res in self.conf_interval.items():
            if col not in col_names:
                continue
            # Build the string with all metrics
            conf_string = (
                f"{col}:\n"
                f"Mean CI: {res['mean_CI']}\n"
                f"Variance CI: {res['variance_CI']}\n"
            )

            # Print or write the string based on the `file` argument
            if file is None:
                print(conf_string)
            else:
                with open(file, 'a+') as f:
                    f.write(conf_string)
    
    def hypothesis_test(self, data_col_name, **kwargs):
        return hypothesis_test(self.df[data_col_name], **kwargs)
    
    def build_linear_regression_model(self, *args):
        self.lin_reg_model = linear_regression_model(self.df, *args)
    
    def build_mult_linear_regression_model(self, *args, **kwargs):
        self.mult_lin_reg_model = multiple_linear_regression(self.df, *args, **kwargs)
    
    def build_stepwise_parsimonious_regression_model(self, *args, **kwargs):
        model, vars, vif = stepwise_parsimonious_regression(self.df, *args, **kwargs)
        self.parsimonious_model = {'final_model': model, 'used_vars': vars, 'vif': vif}

    def plot_histograms_per_col(self,
                                key_in=None, 
                                **kwargs):
        """!
        @brief Create and save histograms for numeric columns using the specified binning method.
        Args:
        - kwargs: Optional arguments for binning method, output directory, or bin width/count.
        """
        if key_in is None:
            key_in = self.num_headers.keys()
        if isinstance(key_in, str):
            key_in = [key_in]
            
        for key in key_in:
            data = self.df[key].dropna()
            plot_histogram(data,
                           **kwargs)
    
    def plot_stacked_bar_chart_horizontal(self, 
                                          **kwargs):
        plot_stacked_bar_chart_horizontal(self.df, 
                                          **kwargs)
    
    def plot_stacked_bar_chart_vertical(self, 
                                        **kwargs):
        plot_stacked_bar_chart_vertical(self.df, 
                                        **kwargs)
    
    def plot_clustered_bar_chart_horizontal(self, 
                                            **kwargs):
        plot_clustered_bar_chart_horizontal(self.df, 
                                            **kwargs)
    
    def plot_clustered_bar_chart_vertical(self, 
                                          **kwargs):
        plot_clustered_bar_chart_vertical(self.df, 
                                          **kwargs)
    
    def plot_individual_bar_charts(self, 
                               **kwargs):
        plot_individual_bar_charts(self.df, 
                                   **kwargs)
    
    def plot_line_chart(self, 
                        **kwargs):
        plot_line_chart(self.df, 
                        **kwargs)
    
    def plot_heatmap(self, 
                     **kwargs):
        plot_heatmap(self.df,
                     **kwargs)
    
    def plot_radar_chart(self,
                         **kwargs):
        plot_radar_chart(self.df,
                         **kwargs)
    
    def vis_reg_models(self, *args, **kwargs):
        visualize_regression_models(self.df, *args, **kwargs)
