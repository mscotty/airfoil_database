import os
import plotly.graph_objects as go
import plotly.io as pio

def plot_stacked_bar_chart_vertical(data, 
                           x_column, 
                           y_column=None, 
                           output_dir=None, 
                           output_name=None, 
                           title_name=None, 
                           title_font_size=28, 
                           title_font_name='Times New Roman', 
                           x_axis_name=None, 
                           x_axis_font_size=24, 
                           x_axis_font_name='Times New Roman', 
                           y_axis_name=None, 
                           y_axis_font_size=24, 
                           y_axis_font_name='Times New Roman'):
    """
    @brief Plot a stacked bar chart.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] x_column (str) Column name for the x-axis.
    @param[in] y_column (str) Column name for the y-axis.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the chart.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    @param[in] x_axis_name (str) Custom label for the x-axis.
    @param[in] x_axis_font_size (int) Font size for the x-axis label.
    @param[in] x_axis_font_name (str) Font name for the x-axis label.
    @param[in] y_axis_name (str) Custom label for the y-axis.
    @param[in] y_axis_font_size (int) Font size for the y-axis label.
    @param[in] y_axis_font_name (str) Font name for the y-axis label.
    """
    fig = go.Figure()

    for column in data.columns:
        if column not in [x_column, y_column]:
            fig.add_trace(
                go.Bar(
                    name=column,
                    x=data[x_column],
                    y=data[column]
                )
            )

    # Update layout for stacked bar chart
    fig.update_layout(
        barmode='stack',
        title=dict(
            text=title_name if title_name else f"Stacked Bar Chart of {y_column} by {x_column}",
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5  # Center title
        ),
        xaxis=dict(
            title=dict(text=x_axis_name if x_axis_name else x_column, 
                       font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size-4),
        ),
        yaxis=dict(
            title=dict(text=y_axis_name if y_axis_name else y_column, 
                       font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size-4),
        ),
        font=dict(family="Times New Roman", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else f"stacked_bar_chart.png")
        fig.write_image(file_path, format="png", width=800, height=600)
    else:
        fig.show()


def plot_stacked_bar_chart_horizontal(data, 
                           y_column, 
                           x_column=None, 
                           output_dir=None, 
                           output_name=None, 
                           title_name=None, 
                           title_font_size=28, 
                           title_font_name='Times New Roman', 
                           x_axis_name=None, 
                           x_axis_font_size=24, 
                           x_axis_font_name='Times New Roman', 
                           y_axis_name=None, 
                           y_axis_font_size=24, 
                           y_axis_font_name='Times New Roman'):
    """
    @brief Plot a stacked bar chart.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] y_column (str) Column name for the y-axis.
    @param[in] x_column (str) Optional column name for the x-axis. If None, all columns except y_column are used.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the chart.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    @param[in] x_axis_name (str) Custom label for the x-axis.
    @param[in] x_axis_font_size (int) Font size for the x-axis label.
    @param[in] x_axis_font_name (str) Font name for the x-axis label.
    @param[in] y_axis_name (str) Custom label for the y-axis.
    @param[in] y_axis_font_size (int) Font size for the y-axis label.
    @param[in] y_axis_font_name (str) Font name for the y-axis label.
    """
    if x_column is None:
        x_columns = [col for col in data.columns if col != y_column]
    else:
        x_columns = [x_column]

    fig = go.Figure()

    for column in x_columns:
        fig.add_trace(
            go.Bar(
                name=column,
                x=data[column],
                y=data[y_column],
                orientation='h'  # Horizontal orientation
            )
        )

    # Update layout for stacked bar chart
    fig.update_layout(
        barmode='stack',
        title=dict(
            text=title_name if title_name else f"Stacked Bar Chart of Values by {y_column}",
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5  # Center title
        ),
        xaxis=dict(
            title=dict(text=x_axis_name if x_axis_name else "Values", 
                       font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size-4),
        ),
        yaxis=dict(
            title=dict(text=y_axis_name if y_axis_name else y_column, 
                       font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size-4),
        ),
        font=dict(family="Times New Roman", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else f"stacked_bar_chart.png")
        fig.write_image(file_path, format="png", width=800, height=600)
    else:
        fig.show()
