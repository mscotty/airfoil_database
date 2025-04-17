import os
import plotly.graph_objects as go
import plotly.io as pio


def plot_line_chart(data, 
                       x_column, 
                       y_column, 
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
    @brief Plot a custom chart to provide insights into the data.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] x_column (str) Column name for the x-axis.
    @param[in] y_column (str) Column name for the y-axis.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the chart. If None, a generic title is used.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    @param[in] x_axis_name (str) Custom label for the x-axis. If None, the x_column name is used.
    @param[in] x_axis_font_size (int) Font size for the x-axis labels.
    @param[in] x_axis_font_name (str) Font name for the x-axis labels.
    @param[in] y_axis_name (str) Custom label for the y-axis. If None, the y_column name is used.
    @param[in] y_axis_font_size (int) Font size for the y-axis labels.
    @param[in] y_axis_font_name (str) Font name for the y-axis labels.
    """
    fig = go.Figure()

    # Add trace for the data
    fig.add_trace(
        go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode='markers+lines',
            marker=dict(size=10, color='blue')
        )
    )

    # Update layout for the insight chart
    fig.update_layout(
        title=dict(
            text=title_name if title_name else f"Insight Chart of {y_column} by {x_column}",
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text=x_axis_name if x_axis_name else x_column, font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            tickfont=dict(family=x_axis_font_name, size=x_axis_font_size-4),
        ),
        yaxis=dict(
            title=dict(text=y_axis_name if y_axis_name else y_column, font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            tickfont=dict(family=y_axis_font_name, size=y_axis_font_size-4),
        ),
        font=dict(family="Times New Roman", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else f"line_chart.png")
        fig.write_image(file_path, format="png", width=800, height=600)
    else:
        fig.show()