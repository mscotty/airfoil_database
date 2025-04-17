import os
import plotly.graph_objects as go


def plot_heatmap(data, 
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
                 y_axis_font_name='Times New Roman',
                 colorscale='Jet'):
    """
    @brief Plot a heatmap to show intensity of values.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the chart.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    """
    fig = go.Figure(data=go.Heatmap(
        z=data.iloc[:, 1:].values,
        x=data.columns[1:],
        y=data.iloc[:, 0],
        colorscale=colorscale))

    fig.update_layout(
        title=dict(
            text=title_name if title_name else "Heatmap of Values",
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text=x_axis_name if x_axis_name else 'Values', font=dict(family=x_axis_font_name, size=x_axis_font_size)),
            tickfont=dict(family=title_font_name, size=title_font_size-6)
        ),
        yaxis=dict(
            title=dict(text=y_axis_name if y_axis_name else 'Values', font=dict(family=y_axis_font_name, size=y_axis_font_size)),
            tickfont=dict(family=title_font_name, size=title_font_size-6)
        )
    )

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else "heatmap.png")
        fig.write_image(file_path, format="png", width=800, height=600)
    else:
        fig.show()