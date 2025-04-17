# Determine binning method
import os

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from airfoil_database.plotting.auto_select_binning_method import auto_select_binning_method
from airfoil_database.formulas.histogram_bins.freedman_diaconis_rule import freedman_diaconis_rule
from airfoil_database.formulas.histogram_bins.square_root_rule import square_root_rule
from airfoil_database.formulas.histogram_bins.sturges_rule import sturges_rule


def plot_histogram(data,
                   binning_method=None, 
                   bin_width=None, 
                   bin_count=None, 
                   output_dir=None, 
                   output_name=None,
                   use_bin_width=True,
                   title_name=None,
                   title_font_size=28,
                   title_font_name='Times New Roman',
                   x_axis_range=None,
                   x_axis_name=None,
                   x_axis_units=None,
                   x_axis_font_size=24,
                   x_axis_font_name='Times New Roman',
                   y_axis_range=None,
                   y_axis_name=None,
                   y_axis_font_size=24,
                   y_axis_font_name='Times New Roman'):
    """!
    @brief Plot a histogram of the data using the specified binning method or custom bin configuration.
    @param[in] data Array-like, numeric data to be plotted.
    @param[in] binning_method (str) The binning method to use ("Freedman-Diaconis", "Square Root", "Sturges", or "Custom").
    @param[in] bin_width (float) Custom bin width, if specified.
    @param[in] bin_count (int) Custom bin count, if specified.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] use_bin_width (bool) Whether to use bin width instead of bin count for the histogram.
    @param[in] title_name (str) Custom title for the histogram.
    @param[in] title_font_size (int) Font size for the histogram title.
    @param[in] title_font_name (str) Font name for the histogram title.
    @param[in] x_axis_name (str) Custom label for the x-axis.
    @param[in] x_axis_units (str) Support label for the x-axis.
    @param[in] x_axis_font_size (int) Font size for the x-axis label.
    @param[in] x_axis_font_name (str) Font name for the x-axis label.
    @param[in] y_axis_name (str) Custom label for the y-axis.
    @param[in] y_axis_font_size (int) Font size for the y-axis label.
    @param[in] y_axis_font_name (str) Font name for the y-axis label.
    @exception ValueError Raised if an unsupported binning method is specified.
    """
    
    if binning_method is None and bin_width is None and bin_count is None:
        binning_meth = auto_select_binning_method(data)
    elif binning_method is None and bin_width is not None or bin_count is not None:
        binning_method = 'Custom'
    else:
        binning_meth = binning_method

    # Determine bin width or bin count
    if binning_meth == "Freedman-Diaconis":
        bin_width, bin_count = freedman_diaconis_rule(data)
    elif binning_meth == "Square Root":
        bin_width, bin_count = square_root_rule(data)
    elif binning_meth == "Sturges":
        bin_width, bin_count = sturges_rule(data)
    elif binning_meth == 'Custom':
        out_str = f'bin_width: {bin_width}' if use_bin_width else f'bin_count: {bin_count}'
        print(f'Using custom input {out_str}')
    else:
        raise ValueError(f"Unsupported binning method: {binning_method}")

    # Log the chosen binning method
    print(f"Using {binning_meth} rule:")
    print(f"  - Bin width: {bin_width:.2f}")
    print(f"  - Bin count: {bin_count}")

    # Configure bins based on user's choice
    if use_bin_width:
        bin_edges = np.arange(data.min(), data.max() + bin_width, bin_width)
        bins = dict(start=data.min(), end=data.max(), size=bin_width)
        bin_str = 'bin_width'
    else:
        bins = bin_count
        bin_str = 'bin_count'
    
    if x_axis_units is None:
        x_axis_units = ''
    else:
        x_axis_units = f" [{x_axis_units}]"

    # Create interactive histogram using Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data,
            xbins=bins if use_bin_width else None,  # Use bins dict only for bin width
            nbinsx=bins if not use_bin_width else None,  # Use bin count otherwise
            marker=dict(color='skyblue', line=dict(color='black', width=1)),
            opacity=0.7,
        )
    )

    # Update layout for clean appearance
    fig.update_layout(
        title=dict(
            text=f"Histogram of {data.name} ({binning_meth} Binning)" if title_name is None else title_name,
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5,  # Center title
        ),
        xaxis=dict(
            title=dict(text=data.name+x_axis_units if x_axis_name is None else x_axis_name, 
                    font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            gridcolor="lightgray",
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size-4),
        ),
        yaxis=dict(
            title=dict(text="Frequency" if y_axis_name is None else y_axis_name, 
                    font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            gridcolor="lightgray",
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size-4),
        ),
        font=dict(family="Times New Roman", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    if x_axis_range is not None:
        fig.update_layout(xaxis_range=x_axis_range)
    
    if y_axis_range is not None:
        fig.update_layout(yaxis_range=y_axis_range)


    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

    # Save or display the histogram
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, 
                                f"{data.name}_histogram_{binning_meth}_{bin_str}.png" if output_name is None else output_name)
        pio.write_image(fig, file_path, format="png", width=800, height=600)
    else:
        fig.show()


def plot_horizontal_histogram(data,
                               binning_method=None,
                               bin_width=None,
                               bin_count=None,
                               output_dir=None,
                               output_name=None,
                               use_bin_width=True,
                               title_name=None,
                               title_font_size=28,
                               title_font_name='Times New Roman',
                               y_axis_range=None,
                               y_axis_name=None,
                               y_axis_units=None,
                               y_axis_font_size=24,
                               y_axis_font_name='Times New Roman',
                               x_axis_range=None,
                               x_axis_name=None,
                               x_axis_font_size=24,
                               x_axis_font_name='Times New Roman'):
    """!
    @brief Plot a horizontal histogram of the data using the specified binning method or custom bin configuration.
    @param[in] data Array-like, numeric data to be plotted.
    @param[in] binning_method (str) The binning method to use ("Freedman-Diaconis", "Square Root", "Sturges", or "Custom").
    @param[in] bin_width (float) Custom bin width, if specified.
    @param[in] bin_count (int) Custom bin count, if specified.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] use_bin_width (bool) Whether to use bin width instead of bin count for the histogram.
    @param[in] title_name (str) Custom title for the histogram.
    @param[in] title_font_size (int) Font size for the histogram title.
    @param[in] title_font_name (str) Font name for the histogram title.
    @param[in] y_axis_name (str) Custom label for the y-axis.
    @param[in] y_axis_units (str) Support label for the y-axis.
    @param[in] y_axis_font_size (int) Font size for the y-axis label.
    @param[in] y_axis_font_name (str) Font name for the y-axis label.
    @param[in] x_axis_name (str) Custom label for the x-axis (frequency).
    @param[in] x_axis_font_size (int) Font size for the x-axis label.
    @param[in] x_axis_font_name (str) Font name for the x-axis label.
    @exception ValueError Raised if an unsupported binning method is specified.
    """

    if binning_method is None and bin_width is None and bin_count is None:
        binning_meth = auto_select_binning_method(data)
    elif binning_method is None and (bin_width is not None or bin_count is not None):
        binning_method = 'Custom'
    else:
        binning_meth = binning_method

    # Determine bin width or bin count
    if binning_meth == "Freedman-Diaconis":
        bin_width, bin_count = freedman_diaconis_rule(data)
    elif binning_meth == "Square Root":
        bin_width, bin_count = square_root_rule(data)
    elif binning_meth == "Sturges":
        bin_width, bin_count = sturges_rule(data)
    elif binning_meth == 'Custom':
        out_str = f'bin_width: {bin_width}' if use_bin_width else f'bin_count: {bin_count}'
        print(f'Using custom input {out_str}')
    else:
        raise ValueError(f"Unsupported binning method: {binning_method}")

    # Log the chosen binning method
    print(f"Using {binning_meth} rule:")
    print(f"  - Bin width: {bin_width:.2f}")
    print(f"  - Bin count: {bin_count}")

    # Configure bins based on user's choice
    if use_bin_width:
        bin_edges = np.arange(data.min(), data.max() + bin_width, bin_width)
        bins = dict(start=data.min(), end=data.max(), size=bin_width)
        bin_str = 'bin_width'
    else:
        bins = bin_count
        bin_str = 'bin_count'

    if y_axis_units is None:
        y_axis_units = ''
    else:
        y_axis_units = f" [{y_axis_units}]"

    # Create interactive horizontal histogram using Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            y=data,
            ybins=bins if use_bin_width else None,  # Use bins dict only for bin width
            nbinsy=bins if not use_bin_width else None,  # Use bin count otherwise
            marker=dict(color='skyblue', line=dict(color='black', width=1)),
            opacity=0.7,
        )
    )

    # Update layout for clean appearance
    fig.update_layout(
        title=dict(
            text=f"Horizontal Histogram of {data.name} ({binning_meth} Binning)" if title_name is None else title_name,
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5,  # Center title
        ),
        yaxis=dict(
            title=dict(text=data.name + y_axis_units if y_axis_name is None else y_axis_name,
                       font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            gridcolor="lightgray",
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size - 4),
        ),
        xaxis=dict(
            title=dict(text="Frequency" if x_axis_name is None else x_axis_name,
                       font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            gridcolor="lightgray",
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size - 4),
        ),
        font=dict(family="Times New Roman", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=50),
    )
    if y_axis_range is not None:
        fig.update_layout(yaxis_range=y_axis_range)

    if x_axis_range is not None:
        fig.update_layout(xaxis_range=x_axis_range)

    # Add grid lines
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

    # Save or display the histogram
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir,
                                 f"{data.name}_horizontal_histogram_{binning_meth}_{bin_str}.png" if output_name is None else output_name)
        pio.write_image(fig, file_path, format="png", width=800, height=600)
    else:
        fig.show()
