import os

import plotly.graph_objects as go
import plotly.io as pio


def plot_vertical_bar_chart(data,
                            labels,
                            output_dir=None,
                            output_name=None,
                            title_name=None,
                            title_font_size=28,
                            title_font_name='Times New Roman',
                            x_axis_name=None,
                            x_axis_font_size=24,
                            x_axis_font_name='Times New Roman',
                            y_axis_name=None,
                            y_axis_units=None,
                            y_axis_font_size=24,
                            y_axis_font_name='Times New Roman'):
    """!
    @brief Plot a vertical bar chart.
    @param[in] data Array-like, numeric data for the bar heights.
    @param[in] labels List-like, labels for each bar along the x-axis.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the bar chart.
    @param[in] title_font_size (int) Font size for the bar chart title.
    @param[in] title_font_name (str) Font name for the bar chart title.
    @param[in] x_axis_name (str) Custom label for the x-axis.
    @param[in] x_axis_font_size (int) Font size for the x-axis label.
    @param[in] x_axis_font_name (str) Font name for the x-axis label.
    @param[in] y_axis_name (str) Custom label for the y-axis.
    @param[in] y_axis_units (str) Support label for the y-axis.
    @param[in] y_axis_font_size (int) Font size for the y-axis label.
    @param[in] y_axis_font_name (str) Font name for the y-axis label.
    """
    if y_axis_units is None:
        y_axis_units = ''
    else:
        y_axis_units = f" [{y_axis_units}]"

    fig = go.Figure(data=[go.Bar(x=labels, y=data)])

    fig.update_layout(
        title=dict(
            text=title_name if title_name is not None else "Vertical Bar Chart",
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text=x_axis_name,
                       font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            gridcolor="lightgray",
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size - 4),
        ),
        yaxis=dict(
            title=dict(text=(y_axis_name + y_axis_units) if y_axis_name is not None else "Value",
                       font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            gridcolor="lightgray",
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size - 4),
        ),
        font=dict(family="Times New Roman", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir,
                                 f"vertical_bar_chart.png" if output_name is None else output_name)
        pio.write_image(fig, file_path, format="png", width=800, height=600)
    else:
        fig.show()


def plot_horizontal_bar_chart(data,
                              labels,
                              output_dir=None,
                              output_name=None,
                              title_name=None,
                              title_font_size=28,
                              title_font_name='Times New Roman',
                              y_axis_name=None,
                              y_axis_font_size=24,
                              y_axis_font_name='Times New Roman',
                              x_axis_name=None,
                              x_axis_units=None,
                              x_axis_font_size=24,
                              x_axis_font_name='Times New Roman'):
    """!
    @brief Plot a horizontal bar chart.
    @param[in] data Array-like, numeric data for the bar lengths.
    @param[in] labels List-like, labels for each bar along the y-axis.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the bar chart.
    @param[in] title_font_size (int) Font size for the bar chart title.
    @param[in] title_font_name (str) Font name for the bar chart title.
    @param[in] y_axis_name (str) Custom label for the y-axis.
    @param[in] y_axis_font_size (int) Font size for the y-axis label.
    @param[in] y_axis_font_name (str) Font name for the y-axis label.
    @param[in] x_axis_name (str) Custom label for the x-axis.
    @param[in] x_axis_units (str) Support label for the x-axis.
    @param[in] x_axis_font_size (int) Font size for the x-axis label.
    @param[in] x_axis_font_name (str) Font name for the x-axis label.
    """
    if x_axis_units is None:
        x_axis_units = ''
    else:
        x_axis_units = f" [{x_axis_units}]"

    fig = go.Figure(data=[go.Bar(y=labels, x=data, orientation='h')])

    fig.update_layout(
        title=dict(
            text=title_name if title_name is not None else "Horizontal Bar Chart",
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5,
        ),
        yaxis=dict(
            title=dict(text=y_axis_name,
                       font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            gridcolor="lightgray",
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size - 4),
        ),
        xaxis=dict(
            title=dict(text=(x_axis_name + x_axis_units) if x_axis_name is not None else "Value",
                       font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            gridcolor="lightgray",
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size - 4),
        ),
        font=dict(family="Times New Roman", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir,
                                 f"horizontal_bar_chart.png" if output_name is None else output_name)
        pio.write_image(fig, file_path, format="png", width=800, height=600)
    else:
        fig.show()


if __name__ == '__main__':
    # Example data
    bar_data = [20, 35, 30, 23, 48]
    bar_labels = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']

    # Plot a vertical bar chart
    plot_vertical_bar_chart(bar_data, bar_labels, title_name="Vertical Bar Chart Example",
                             x_axis_name="Categories", y_axis_name="Values")

    # Plot a horizontal bar chart
    plot_horizontal_bar_chart(bar_data, bar_labels, title_name="Horizontal Bar Chart Example",
                               y_axis_name="Categories", x_axis_name="Values")

    # Save the horizontal bar chart to a directory
    output_directory = "output_plots"
    plot_horizontal_bar_chart(bar_data, bar_labels, output_dir=output_directory,
                                output_name="horizontal_bar_chart_saved.png",
                                title_name="Saved Horizontal Bar Chart",
                                y_axis_name="Items", x_axis_name="Count", x_axis_units="Units")