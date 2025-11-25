# dashboard/airfoil_dashboard.py
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import json

from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.utilities.helpers import pointcloud_string_to_array
from airfoil_database.xfoil.processor import PointcloudProcessor
from airfoil_database.xfoil.fix_point_cloud_simple import AirfoilProcessor


class AirfoilDashboard:
    def __init__(self, database_path="airfoil_data.db", database_dir="."):
        """Initialize the dashboard with database connection."""
        self.db = AirfoilDatabase(database_path, database_dir)
        self.app = dash.Dash(__name__)
        self.current_airfoil_data = None
        self.current_points = None
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div(
            [
                html.H1(
                    "Airfoil Database Dashboard",
                    style={"textAlign": "center", "marginBottom": 30},
                ),
                # Control Panel
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Airfoil Selection"),
                                dcc.Dropdown(
                                    id="airfoil-dropdown",
                                    placeholder="Select an airfoil...",
                                    style={"marginBottom": 10},
                                ),
                                html.Button(
                                    "Load Airfoil",
                                    id="load-btn",
                                    className="button-primary",
                                    style={"marginRight": 10},
                                ),
                                html.Button(
                                    "Process/Fix Points",
                                    id="process-btn",
                                    className="button-secondary",
                                    style={"marginRight": 10},
                                ),
                                html.Button(
                                    "Save Changes",
                                    id="save-btn",
                                    className="button-success",
                                ),
                            ],
                            className="six columns",
                        ),
                        html.Div(
                            [
                                html.H3("Point Management"),
                                html.P(
                                    "Click points on the graph to select/deselect them"
                                ),
                                html.Button(
                                    "Move Selected Up",
                                    id="move-up-btn",
                                    style={"marginRight": 5},
                                ),
                                html.Button(
                                    "Move Selected Down",
                                    id="move-down-btn",
                                    style={"marginRight": 5},
                                ),
                                html.Button(
                                    "Delete Selected",
                                    id="delete-btn",
                                    style={
                                        "backgroundColor": "#ff4444",
                                        "color": "white",
                                    },
                                ),
                                html.Br(),
                                html.Button(
                                    "Reverse Order",
                                    id="reverse-btn",
                                    style={"marginTop": 10, "marginRight": 5},
                                ),
                                html.Button(
                                    "Reset Points",
                                    id="reset-btn",
                                    style={"marginTop": 10},
                                ),
                            ],
                            className="six columns",
                        ),
                    ],
                    className="row",
                    style={"marginBottom": 20},
                ),
                # Status Display
                html.Div(
                    id="status-display",
                    style={
                        "marginBottom": 20,
                        "padding": 10,
                        "backgroundColor": "#f0f0f0",
                        "borderRadius": 5,
                    },
                ),
                # Main Content Area
                html.Div(
                    [
                        # Graph Panel
                        html.Div(
                            [
                                dcc.Graph(
                                    id="airfoil-plot",
                                    style={"height": "600px"},
                                    config={"displayModeBar": True},
                                )
                            ],
                            className="eight columns",
                        ),
                        # Data Panel
                        html.Div(
                            [
                                html.H4("Airfoil Information"),
                                html.Div(id="airfoil-info"),
                                html.H4("Point Data", style={"marginTop": 20}),
                                html.Div(
                                    [
                                        dash_table.DataTable(
                                            id="points-table",
                                            columns=[
                                                {
                                                    "name": "Index",
                                                    "id": "index",
                                                    "type": "numeric",
                                                },
                                                {
                                                    "name": "X",
                                                    "id": "x",
                                                    "type": "numeric",
                                                    "format": {"specifier": ".6f"},
                                                },
                                                {
                                                    "name": "Y",
                                                    "id": "y",
                                                    "type": "numeric",
                                                    "format": {"specifier": ".6f"},
                                                },
                                            ],
                                            style_cell={"textAlign": "left"},
                                            style_data_conditional=[
                                                {
                                                    "if": {"state": "selected"},
                                                    "backgroundColor": "rgba(0, 116, 217, 0.3)",
                                                }
                                            ],
                                            row_selectable="multi",
                                            page_size=15,
                                            fixed_rows={"headers": True},
                                            style_table={
                                                "height": "400px",
                                                "overflowY": "auto",
                                            },
                                        )
                                    ]
                                ),
                            ],
                            className="four columns",
                        ),
                    ],
                    className="row",
                ),
                # Hidden divs to store data
                html.Div(id="current-airfoil-data", style={"display": "none"}),
                html.Div(id="original-points-data", style={"display": "none"}),
            ],
            style={"margin": "20px"},
        )

    def setup_callbacks(self):
        """Setup all dashboard callbacks."""

        @self.app.callback(
            Output("airfoil-dropdown", "options"), Input("airfoil-dropdown", "id")
        )
        def update_airfoil_dropdown(_):
            """Populate airfoil dropdown with database contents."""
            try:
                df = self.db.get_airfoil_dataframe()
                options = [
                    {"label": f"{name} ({series})", "value": name}
                    for name, series in zip(df["Name"], df["Series"])
                ]
                return options
            except Exception as e:
                return [{"label": f"Error loading airfoils: {str(e)}", "value": ""}]

        @self.app.callback(
            [
                Output("current-airfoil-data", "children"),
                Output("original-points-data", "children"),
                Output("status-display", "children"),
            ],
            [Input("load-btn", "n_clicks"), Input("process-btn", "n_clicks")],
            [
                State("airfoil-dropdown", "value"),
                State("current-airfoil-data", "children"),
            ],
        )
        def load_or_process_airfoil(
            load_clicks, process_clicks, selected_airfoil, current_data
        ):
            """Load airfoil data or process current points."""
            ctx = dash.callback_context

            if not ctx.triggered:
                return None, None, "Select an airfoil and click 'Load Airfoil'"

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "load-btn":
                if not selected_airfoil:
                    return None, None, "Please select an airfoil first"

                try:
                    data = self.db.get_airfoil_data(selected_airfoil)
                    if not data:
                        return None, None, f"Airfoil '{selected_airfoil}' not found"

                    description, pointcloud, series, source = data
                    airfoil_data = {
                        "name": selected_airfoil,
                        "description": description,
                        "pointcloud": pointcloud,
                        "series": series,
                        "source": source,
                    }

                    return (
                        json.dumps(airfoil_data),
                        pointcloud,  # Store original points
                        f"Loaded airfoil: {selected_airfoil}",
                    )

                except Exception as e:
                    return None, None, f"Error loading airfoil: {str(e)}"

            elif button_id == "process-btn":
                if not current_data:
                    return None, None, "No airfoil loaded"

                try:
                    airfoil_data = json.loads(current_data)
                    processor = AirfoilProcessor()
                    processed_pointcloud, info = processor.process(
                        airfoil_data["pointcloud"]
                    )

                    if info["status"] == "error":
                        return (
                            current_data,
                            None,
                            f"Processing failed: {'; '.join(info['messages'])}",
                        )

                    # Update the airfoil data with processed points
                    airfoil_data["pointcloud"] = processed_pointcloud

                    status_msg = f"Processed airfoil: {airfoil_data['name']}\n"
                    status_msg += "\n".join(
                        info["messages"][:5]
                    )  # Show first 5 messages

                    return json.dumps(airfoil_data), None, status_msg

                except Exception as e:
                    return current_data, None, f"Error processing airfoil: {str(e)}"

            return None, None, ""

        @self.app.callback(
            [
                Output("airfoil-plot", "figure"),
                Output("points-table", "data"),
                Output("airfoil-info", "children"),
            ],
            [
                Input("current-airfoil-data", "children"),
                Input("move-up-btn", "n_clicks"),
                Input("move-down-btn", "n_clicks"),
                Input("delete-btn", "n_clicks"),
                Input("reverse-btn", "n_clicks"),
                Input("reset-btn", "n_clicks"),
            ],
            [
                State("points-table", "selected_rows"),
                State("original-points-data", "children"),
            ],
        )
        def update_plot_and_table(
            airfoil_data,
            move_up_clicks,
            move_down_clicks,
            delete_clicks,
            reverse_clicks,
            reset_clicks,
            selected_rows,
            original_points,
        ):
            """Update the plot and table based on current data and user actions."""

            if not airfoil_data:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No airfoil loaded")
                return empty_fig, [], "No airfoil selected"

            try:
                data = json.loads(airfoil_data)
                pointcloud_str = data["pointcloud"]

                # Handle point manipulation actions
                ctx = dash.callback_context
                if ctx.triggered:
                    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
                    points = pointcloud_string_to_array(pointcloud_str)

                    if points is not None and len(points) > 0:
                        if button_id == "reset-btn" and original_points:
                            pointcloud_str = original_points
                            points = pointcloud_string_to_array(pointcloud_str)
                            # Update the stored data
                            data["pointcloud"] = pointcloud_str

                        elif button_id in [
                            "move-up-btn",
                            "move-down-btn",
                            "delete-btn",
                            "reverse-btn",
                        ]:
                            if button_id == "reverse-btn":
                                points = points[::-1]
                            elif selected_rows and button_id == "delete-btn":
                                # Delete selected points
                                points = np.delete(points, selected_rows, axis=0)
                            elif selected_rows and button_id in [
                                "move-up-btn",
                                "move-down-btn",
                            ]:
                                # Move selected points
                                direction = -1 if button_id == "move-up-btn" else 1
                                points = self._move_points(
                                    points, selected_rows, direction
                                )

                            # Convert back to string and update data
                            pointcloud_str = self._points_to_string(points)
                            data["pointcloud"] = pointcloud_str

                # Parse current points
                points = pointcloud_string_to_array(pointcloud_str)

                if points is None or len(points) == 0:
                    empty_fig = go.Figure()
                    empty_fig.update_layout(title="Invalid point data")
                    return empty_fig, [], "Invalid point data"

                # Create the plot
                fig = self._create_airfoil_plot(points, data["name"])

                # Create table data
                table_data = [
                    {"index": i, "x": float(points[i, 0]), "y": float(points[i, 1])}
                    for i in range(len(points))
                ]

                # Create info display
                info_div = html.Div(
                    [
                        html.P(f"Name: {data['name']}"),
                        html.P(f"Series: {data.get('series', 'Unknown')}"),
                        html.P(f"Points: {len(points)}"),
                        html.P(f"Source: {data.get('source', 'Unknown')}"),
                        html.Hr(),
                        html.P(
                            f"Description: {data.get('description', 'None')}",
                            style={"fontSize": "12px"},
                        ),
                    ]
                )

                return fig, table_data, info_div

            except Exception as e:
                empty_fig = go.Figure()
                empty_fig.update_layout(title=f"Error: {str(e)}")
                return empty_fig, [], f"Error: {str(e)}"

        @self.app.callback(
            Output("status-display", "children", allow_duplicate=True),
            Input("save-btn", "n_clicks"),
            State("current-airfoil-data", "children"),
            prevent_initial_call=True,
        )
        def save_airfoil_changes(save_clicks, airfoil_data):
            """Save changes back to the database."""
            if not save_clicks or not airfoil_data:
                return "No changes to save"

            try:
                data = json.loads(airfoil_data)

                # Update the airfoil in the database
                self.db.store_airfoil_data(
                    name=data["name"],
                    description=data.get("description", ""),
                    pointcloud=data["pointcloud"],
                    airfoil_series=data.get("series", "UNKNOWN"),
                    source=data.get("source", ""),
                    overwrite=True,
                )

                return f"Successfully saved changes to {data['name']}"

            except Exception as e:
                return f"Error saving changes: {str(e)}"

    def _create_airfoil_plot(self, points: np.ndarray, name: str) -> go.Figure:
        """Create plotly figure for airfoil visualization."""
        fig = go.Figure()

        # Add airfoil outline
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="lines+markers",
                name="Airfoil",
                line=dict(color="blue", width=2),
                marker=dict(size=6, color="red"),
                hovertemplate="Point %{pointNumber}<br>X: %{x:.6f}<br>Y: %{y:.6f}<extra></extra>",
            )
        )

        # Add point numbers as annotations
        for i, (x, y) in enumerate(
            points[:: max(1, len(points) // 20)]
        ):  # Show every nth point number
            fig.add_annotation(
                x=x,
                y=y,
                text=str(i * max(1, len(points) // 20)),
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="gray",
                font=dict(size=10, color="gray"),
            )

            # Configure layout
        fig.update_layout(
            title=f"Airfoil: {name}",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            showlegend=True,
            hovermode="closest",
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
            margin=dict(l=50, r=50, t=50, b=50),
        )

        return fig

    def _move_points(
        self, points: np.ndarray, selected_indices: List[int], direction: int
    ) -> np.ndarray:
        """Move selected points up or down in the order."""
        if not selected_indices or len(points) <= 1:
            return points

        points = points.copy()
        n = len(points)

        # Sort indices to handle multiple selections properly
        selected_indices = sorted(selected_indices)

        if direction == -1:  # Move up
            for idx in selected_indices:
                if idx > 0:
                    points[[idx - 1, idx]] = points[[idx, idx - 1]]
        else:  # Move down
            for idx in reversed(selected_indices):
                if idx < n - 1:
                    points[[idx, idx + 1]] = points[[idx + 1, idx]]

        return points

    def _points_to_string(self, points: np.ndarray) -> str:
        """Convert points array back to string format."""
        if points is None or len(points) == 0:
            return ""

        lines = []
        for x, y in points:
            lines.append(f"{x:.6f} {y:.6f}")
        return "\n".join(lines)

    def run(self, debug=True, port=8050):
        """Run the dashboard server."""
        self.app.run(debug=debug, port=port)


# Additional utility dashboard for database overview
class DatabaseOverviewDashboard:
    def __init__(self, database_path="airfoil_data.db", database_dir="."):
        """Initialize the overview dashboard."""
        self.db = AirfoilDatabase(database_path, database_dir)
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup the overview dashboard layout."""
        self.app.layout = html.Div(
            [
                html.H1(
                    "Airfoil Database Overview",
                    style={"textAlign": "center", "marginBottom": 30},
                ),
                # Summary Statistics
                html.Div(
                    [html.Div(id="database-stats", className="twelve columns")],
                    className="row",
                    style={"marginBottom": 20},
                ),
                # Control Panel
                html.Div(
                    [
                        html.Button(
                            "Refresh Data",
                            id="refresh-btn",
                            className="button-primary",
                            style={"marginRight": 10},
                        ),
                        html.Button(
                            "Process All Airfoils",
                            id="process-all-btn",
                            className="button-secondary",
                            style={"marginRight": 10},
                        ),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select CSV/JSON Files")]
                            ),
                            style={
                                "width": "300px",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px",
                            },
                            multiple=False,
                        ),
                    ],
                    style={"marginBottom": 20},
                ),
                # Status Display
                html.Div(
                    id="overview-status",
                    style={
                        "marginBottom": 20,
                        "padding": 10,
                        "backgroundColor": "#f0f0f0",
                        "borderRadius": 5,
                    },
                ),
                # Main Content Tabs
                dcc.Tabs(
                    id="overview-tabs",
                    value="airfoils-tab",
                    children=[
                        dcc.Tab(label="Airfoils", value="airfoils-tab"),
                        dcc.Tab(label="Geometry Data", value="geometry-tab"),
                        dcc.Tab(label="Series Analysis", value="series-tab"),
                    ],
                ),
                html.Div(id="tab-content"),
            ],
            style={"margin": "20px"},
        )

    def setup_callbacks(self):
        """Setup overview dashboard callbacks."""

        @self.app.callback(
            [
                Output("database-stats", "children"),
                Output("overview-status", "children"),
            ],
            [Input("refresh-btn", "n_clicks"), Input("process-all-btn", "n_clicks")],
        )
        def update_database_stats(refresh_clicks, process_clicks):
            """Update database statistics."""
            ctx = dash.callback_context

            try:
                df = self.db.get_airfoil_dataframe()
                geometry_df = self.db.get_airfoil_geometry_dataframe()

                total_airfoils = len(df)
                series_counts = df["Series"].value_counts().to_dict()
                avg_points = df["Num_Points"].mean() if not df.empty else 0

                # Handle process all button
                status_message = "Database loaded successfully"
                if (
                    ctx.triggered
                    and ctx.triggered[0]["prop_id"] == "process-all-btn.n_clicks"
                ):
                    status_message = self._process_all_airfoils()

                # Create statistics cards
                stats_cards = html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    str(total_airfoils),
                                    style={"color": "#1f77b4", "margin": 0},
                                ),
                                html.P("Total Airfoils", style={"margin": 0}),
                            ],
                            className="three columns",
                            style={
                                "backgroundColor": "white",
                                "padding": 20,
                                "borderRadius": 5,
                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(
                            [
                                html.H3(
                                    str(len(series_counts)),
                                    style={"color": "#ff7f0e", "margin": 0},
                                ),
                                html.P("Series Types", style={"margin": 0}),
                            ],
                            className="three columns",
                            style={
                                "backgroundColor": "white",
                                "padding": 20,
                                "borderRadius": 5,
                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(
                            [
                                html.H3(
                                    f"{avg_points:.0f}",
                                    style={"color": "#2ca02c", "margin": 0},
                                ),
                                html.P("Avg Points", style={"margin": 0}),
                            ],
                            className="three columns",
                            style={
                                "backgroundColor": "white",
                                "padding": 20,
                                "borderRadius": 5,
                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(
                            [
                                html.H3(
                                    str(len(geometry_df)),
                                    style={"color": "#d62728", "margin": 0},
                                ),
                                html.P("With Geometry", style={"margin": 0}),
                            ],
                            className="three columns",
                            style={
                                "backgroundColor": "white",
                                "padding": 20,
                                "borderRadius": 5,
                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                "textAlign": "center",
                            },
                        ),
                    ],
                    className="row",
                )

                return stats_cards, status_message

            except Exception as e:
                error_div = html.Div(
                    [html.H3("Error Loading Database", style={"color": "red"})]
                )
                return error_div, f"Error: {str(e)}"

        @self.app.callback(
            Output("tab-content", "children"), Input("overview-tabs", "value")
        )
        def render_tab_content(active_tab):
            """Render content based on selected tab."""
            if active_tab == "airfoils-tab":
                return self._render_airfoils_tab()
            elif active_tab == "geometry-tab":
                return self._render_geometry_tab()
            elif active_tab == "series-tab":
                return self._render_series_tab()
            return html.Div("Select a tab")

        @self.app.callback(
            Output("overview-status", "children", allow_duplicate=True),
            Input("upload-data", "contents"),
            State("upload-data", "filename"),
            prevent_initial_call=True,
        )
        def handle_file_upload(contents, filename):
            """Handle file upload for bulk data import."""
            if contents is not None:
                try:
                    import base64
                    import io

                    content_type, content_string = contents.split(",")
                    decoded = base64.b64decode(content_string)

                    if filename.endswith(".csv"):
                        # Save to temporary file and import
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".csv", delete=False
                        ) as tmp:
                            tmp.write(decoded.decode("utf-8"))
                            tmp_path = tmp.name

                        self.db.add_airfoils_from_csv(tmp_path, overwrite=False)
                        import os

                        os.unlink(tmp_path)
                        return f"Successfully imported CSV file: {filename}"

                    elif filename.endswith(".json"):
                        # Save to temporary file and import
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False
                        ) as tmp:
                            tmp.write(decoded.decode("utf-8"))
                            tmp_path = tmp.name

                        self.db.add_airfoils_from_json(tmp_path, overwrite=False)
                        import os

                        os.unlink(tmp_path)
                        return f"Successfully imported JSON file: {filename}"

                    else:
                        return f"Unsupported file type: {filename}"

                except Exception as e:
                    return f"Error importing file {filename}: {str(e)}"

            return "No file uploaded"

    def _render_airfoils_tab(self):
        """Render the airfoils overview tab."""
        try:
            df = self.db.get_airfoil_dataframe()

            if df.empty:
                return html.Div("No airfoils in database")

            # Create series distribution chart
            series_counts = df["Series"].value_counts()
            series_fig = px.pie(
                values=series_counts.values,
                names=series_counts.index,
                title="Airfoil Series Distribution",
            )

            # Create points distribution histogram
            points_fig = px.histogram(
                df, x="Num_Points", nbins=20, title="Distribution of Point Counts"
            )

            return html.Div(
                [
                    html.Div([dcc.Graph(figure=series_fig)], className="six columns"),
                    html.Div([dcc.Graph(figure=points_fig)], className="six columns"),
                    html.Div(
                        [
                            html.H4("Airfoils Data Table"),
                            dash_table.DataTable(
                                data=df.to_dict("records"),
                                columns=[{"name": i, "id": i} for i in df.columns],
                                page_size=20,
                                sort_action="native",
                                filter_action="native",
                                style_cell={"textAlign": "left"},
                                style_table={"overflowX": "auto"},
                            ),
                        ],
                        className="twelve columns",
                        style={"marginTop": 20},
                    ),
                ]
            )

        except Exception as e:
            return html.Div(f"Error loading airfoils data: {str(e)}")

    def _render_geometry_tab(self):
        """Render the geometry data tab."""
        try:
            df = self.db.get_airfoil_geometry_dataframe()

            if df.empty:
                return html.Div("No geometry data in database")

            # Create scatter plots for geometry parameters
            thickness_fig = px.scatter(
                df,
                x="max_thickness",
                y="max_camber",
                hover_data=["name"],
                title="Thickness vs Camber",
            )

            return html.Div(
                [
                    html.Div(
                        [dcc.Graph(figure=thickness_fig)], className="twelve columns"
                    ),
                    html.Div(
                        [
                            html.H4("Geometry Data Table"),
                            dash_table.DataTable(
                                data=df.to_dict("records"),
                                columns=[
                                    (
                                        {
                                            "name": i,
                                            "id": i,
                                            "type": "numeric",
                                            "format": {"specifier": ".4f"},
                                        }
                                        if i not in ["id", "name"]
                                        else {"name": i, "id": i}
                                    )
                                    for i in df.columns
                                ],
                                page_size=20,
                                sort_action="native",
                                filter_action="native",
                                style_cell={"textAlign": "left"},
                                style_table={"overflowX": "auto"},
                            ),
                        ],
                        className="twelve columns",
                        style={"marginTop": 20},
                    ),
                ]
            )

        except Exception as e:
            return html.Div(f"Error loading geometry data: {str(e)}")

    def _render_series_tab(self):
        """Render the series analysis tab."""
        try:
            df = self.db.get_airfoil_dataframe()

            if df.empty:
                return html.Div("No data available for series analysis")

            # Group by series and analyze
            series_analysis = (
                df.groupby("Series")
                .agg({"Name": "count", "Num_Points": ["mean", "std", "min", "max"]})
                .round(2)
            )

            series_analysis.columns = [
                "Count",
                "Avg_Points",
                "Std_Points",
                "Min_Points",
                "Max_Points",
            ]
            series_analysis = series_analysis.reset_index()

            # Create bar chart
            bar_fig = px.bar(
                series_analysis,
                x="Series",
                y="Count",
                title="Number of Airfoils by Series",
            )

            return html.Div(
                [
                    html.Div([dcc.Graph(figure=bar_fig)], className="twelve columns"),
                    html.Div(
                        [
                            html.H4("Series Analysis Table"),
                            dash_table.DataTable(
                                data=series_analysis.to_dict("records"),
                                columns=[
                                    (
                                        {
                                            "name": i,
                                            "id": i,
                                            "type": "numeric",
                                            "format": {"specifier": ".2f"},
                                        }
                                        if i != "Series"
                                        else {"name": i, "id": i}
                                    )
                                    for i in series_analysis.columns
                                ],
                                sort_action="native",
                                style_cell={"textAlign": "left"},
                                style_table={"overflowX": "auto"},
                            ),
                        ],
                        className="twelve columns",
                        style={"marginTop": 20},
                    ),
                ]
            )

        except Exception as e:
            return html.Div(f"Error loading series analysis: {str(e)}")

    def _process_all_airfoils(self):
        """Process all airfoils in the database using the point cloud processor."""
        try:
            from sqlmodel import Session, select
            from airfoil_database.core.models import Airfoil

            processed_count = 0
            error_count = 0

            with Session(self.db.engine) as session:
                statement = select(Airfoil)
                airfoils = session.exec(statement).all()

                processor = AirfoilProcessor()

                for airfoil in airfoils:
                    try:
                        if airfoil.pointcloud:
                            processed_pointcloud, info = processor.process(
                                airfoil.pointcloud
                            )

                            if info["status"] == "success":
                                airfoil.pointcloud = processed_pointcloud
                                session.add(airfoil)
                                processed_count += 1
                            else:
                                error_count += 1
                        else:
                            error_count += 1

                    except Exception as e:
                        error_count += 1
                        continue

                session.commit()

            return f"Processed {processed_count} airfoils successfully. {error_count} errors encountered."

        except Exception as e:
            return f"Error processing airfoils: {str(e)}"

    def run(self, debug=True, port=8051):
        """Run the overview dashboard server."""
        self.app.run(debug=debug, port=port)


# CSS styling for better appearance
def get_dashboard_styles():
    """Return CSS styles for the dashboard."""
    return """
    .button-primary {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .button-primary:hover {
        background-color: #0056b3;
    }
    
    .button-secondary {
        background-color: #6c757d;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .button-secondary:hover {
        background-color: #545b62;
    }
    
    .button-success {
        background-color: #28a745;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .button-success:hover {
        background-color: #1e7e34;
    }
    
    body {
        font-family: Arial, sans-serif;
        background-color: #f8f9fa;
    }
    
    .row {
        margin-bottom: 20px;
    }
    """


# Main application launcher
def create_combined_app(database_path="airfoil_data.db", database_dir="."):
    """Create a combined application with both dashboards."""
    import threading
    import webbrowser
    import time

    # Create both dashboards
    airfoil_dashboard = AirfoilDashboard(database_path, database_dir)
    overview_dashboard = DatabaseOverviewDashboard(database_path, database_dir)

    # Add CSS styling
    styles = get_dashboard_styles()
    airfoil_dashboard.app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                {styles}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """

    overview_dashboard.app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
                {styles}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """

    def run_overview_dashboard():
        overview_dashboard.run(debug=False, port=8051)

    def run_airfoil_dashboard():
        airfoil_dashboard.run(debug=False, port=8050)

    # Start both dashboards in separate threads
    overview_thread = threading.Thread(target=run_overview_dashboard, daemon=True)
    airfoil_thread = threading.Thread(target=run_airfoil_dashboard, daemon=True)

    overview_thread.start()
    airfoil_thread.start()

    # Wait a moment for servers to start, then open browsers
    time.sleep(2)

    print("Starting Airfoil Database Dashboards...")
    print("Airfoil Detail Dashboard: http://localhost:8050")
    print("Database Overview Dashboard: http://localhost:8051")
    print("Press Ctrl+C to stop both dashboards")

    # Open both dashboards in browser
    webbrowser.open("http://localhost:8050")
    time.sleep(1)
    webbrowser.open("http://localhost:8051")

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down dashboards...")


# Usage example and main execution
if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Launch Airfoil Database Dashboard")
    # parser.add_argument(
    #     "--db-path", default="airfoil_data.db", help="Database file name"
    # )
    # parser.add_argument("--db-dir", default=".", help="Database directory")
    # parser.add_argument(
    #     "--mode",
    #     choices=["combined", "airfoil", "overview"],
    #     default="combined",
    #     help="Dashboard mode to run",
    # )
    # parser.add_argument(
    #     "--port", type=int, default=8050, help="Port number for single dashboard mode"
    # )

    # args = parser.parse_args()

    # if args.mode == "combined":
    #     create_combined_app(args.db_path, args.db_dir)
    # elif args.mode == "airfoil":
    #     dashboard = AirfoilDashboard(args.db_path, args.db_dir)
    #     print(f"Starting Airfoil Detail Dashboard on http://localhost:{args.port}")
    #     dashboard.run(debug=True, port=args.port)
    # elif args.mode == "overview":
    #     dashboard = DatabaseOverviewDashboard(args.db_path, args.db_dir)
    #     print(f"Starting Database Overview Dashboard on http://localhost:{args.port}")
    #     dashboard.run(debug=True, port=args.port)
    mode = 'combined'  # 'combined', 'airfoil', or 'overview'
    db_path = 'D:/Mitchell/School/2025_Winter/github/airfoil_database/airfoil_database/airfoils.db'
    db_dir = 'D:/Mitchell/School/2025_Winter/github/airfoil_database/airfoil_database'
    if mode == "combined":
        create_combined_app(db_path, db_dir)
    elif mode == "airfoil":
        dashboard = AirfoilDashboard(db_path, db_dir)
        print(f"Starting Airfoil Detail Dashboard on http://localhost:")
        dashboard.run(debug=True)
    elif mode == "overview":
        dashboard = DatabaseOverviewDashboard(db_path, db_dir)
        print(f"Starting Database Overview Dashboard on http://localhost:")
        dashboard.run(debug=True)
