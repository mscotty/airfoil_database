import os
import plotly.graph_objects as go


def plot_radar_chart(data, 
                     category_column=None,
                     output_dir=None, 
                     output_name=None, 
                     title_name=None, 
                     title_font_size=28, 
                     title_font_name='Times New Roman'):
    """
    @brief Plot a radar chart to compare categories across cities.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] output_dir (str) Directory to save the plot as a PNG file. If None, the plot is displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the chart.
    @param[in] title_font_size (int) Font size for the chart title.
    @param[in] title_font_name (str) Font name for the chart title.
    """
    if category_column is None:
        categories = data.columns[1:]  # Exclude first column
    else:
        categories = data.loc[:, data.columns != category_column]

    fig = go.Figure()

    for i, row in data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].values,
            theta=categories,
            fill='toself',
            name=row[data.columns[0]]  # City name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        title=dict(
            text=title_name if title_name else "Radar Chart",
            font=dict(family=title_font_name, size=title_font_size),
            x=0.5
        )
    )

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, output_name if output_name else "radar_chart.png")
        fig.write_image(file_path, format="png", width=800, height=600)
    else:
        fig.show()