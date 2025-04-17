import os
import plotly.graph_objects as go
import plotly.io as pio


def plot_individual_bar_charts(data, 
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
    @brief Plot individual bar charts for each unique value in the x_column.
    @param[in] data DataFrame containing the data to be plotted.
    @param[in] x_column (str) Column name for the x-axis.
    @param[in] y_column (str) Column name for the y-axis.
    @param[in] output_dir (str) Directory to save the plots as PNG files. If None, the plots are displayed interactively.
    @param[in] output_name (str) Custom file name for the saved plot.
    @param[in] title_name (str) Custom title for the charts. If None, generic titles are used.
    @param[in] title_font_size (int) Font size for the chart titles.
    @param[in] title_font_name (str) Font name for the chart titles.
    @param[in] x_axis_name (str) Custom label for the x-axis. If None, the x_column name is used.
    @param[in] x_axis_font_size (int) Font size for the x-axis labels.
    @param[in] x_axis_font_name (str) Font name for the x-axis labels.
    @param[in] y_axis_name (str) Custom label for the y-axis. If None, the y_column name is used.
    @param[in] y_axis_font_size (int) Font size for the y-axis labels.
    @param[in] y_axis_font_name (str) Font name for the y-axis labels.
    """
    unique_values = data[x_column].unique()

    for value in unique_values:
        subset = data[data[x_column] == value]
        fig = go.Figure()

        for column in subset.columns:
            if column not in [x_column, y_column]:
                fig.add_trace(
                    go.Bar(
                        name=column,
                        x=[value],
                        y=[subset[column].iloc[0]]
                    )
                )

        fig.update_layout(
            title=dict(
                text=title_name if title_name else f"Bar Chart for {value}",
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
            file_name = output_name if output_name else f"bar_chart_{value}.png"
            file_path = os.path.join(output_dir, file_name)
            fig.write_image(file_path, format="png", width=800, height=600)
        else:
            fig.show()