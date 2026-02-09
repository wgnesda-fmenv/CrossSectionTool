"""
Cross-Section Tool for Geological/Lithologic Data
==================================================

A flexible tool for creating geological cross-sections from borehole lithologic data
or point measurements. Supports both interactive line drawing and programmatic input.

Author: Based on xsec_tool.py with enhanced functionality
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import Union, List, Tuple, Optional, Dict

import warnings

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    warnings.warn("rasterio not available. DEM functionality will be limited.")


class CrossSectionTool:
    """
    A comprehensive tool for creating geological cross-sections.
    
    This class handles:
    - Interactive or programmatic cross-section line definition
    - Projection of borehole/point data onto cross-section lines
    - Visualization of lithologic intervals
    - DEM integration for ground surface representation
    - Export of cross-section data
    
    Attributes:
        data (pd.DataFrame): The input dataset
        xsec_line (LineString): The cross-section line geometry
        xsec_data (pd.DataFrame): Projected data along the cross-section
        search_distance (float): Maximum distance for including points
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 column_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the CrossSectionTool.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with spatial coordinates and lithologic/measurement information
        column_mapping : dict, optional
            Mapping of expected columns to actual column names in data.
            Default expects: {
                'station_id': 'StationID',
                'x_coord': 'X_Coord',
                'y_coord': 'Y_Coord',
                'top_depth': 'Top_Depth',  # For interval data
                'bottom_depth': 'Bottom_Depth',  # For interval data
                'depth': 'Depth',  # For point data
                'description': 'Description',  # Lithology or measurement name
                'value': 'Value'  # Optional numeric value for coloring
            }
        
        Example:
        --------
        >>> df = pd.read_csv('borehole_data.csv')
        >>> xsec = CrossSectionTool(df)
        >>> xsec.set_line_interactive()
        >>> xsec.plot_cross_section()
        """
        self.data = data.copy()
        #add a check that there are no NaN values
        
        # Set up column mapping
        default_mapping = {
            'station_id': 'StationID',
            'x_coord': 'X_Coord',
            'y_coord': 'Y_Coord',
            'top_depth': 'Top_Depth',
            'bottom_depth': 'Bottom_Depth',
            'depth': 'Depth',
            'description': 'Description',
            'value': 'Value'
        }
        
        if column_mapping:
            default_mapping.update(column_mapping)
        
        self.columns = default_mapping
        self.xsec_line = None
        self.xsec_data = None
        self.search_distance = None
        self.dem_path = None
        self.reference_elevation = 0.0
        
    def set_line_interactive(self, 
                            figsize: Tuple[float, float] = (10, 8),
                            background_data: Optional[pd.DataFrame] = None):
        """
        Interactively draw a cross-section line on a map.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size for the interactive plot
        background_data : pd.DataFrame, optional
            Additional data to plot in background (e.g., all wells)
        
        Returns:
        --------
        list
            Endpoint coordinates [[x1, y1], [x2, y2]]
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all data points
        plot_data = background_data if background_data is not None else self.data
        ax.scatter(plot_data[self.columns['x_coord']], 
                  plot_data[self.columns['y_coord']], 
                  c='blue', alpha=0.5, s=20, label='Data Points')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Click two points to define cross-section line')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        # Get two points from user
        print("Click two points on the map to define the cross-section line...")
        pts = plt.ginput(2, timeout=0)
        
        if len(pts) < 2:
            raise ValueError("Need two points to define a line")
        
        # Draw the line
        xs = [pts[0][0], pts[1][0]]
        ys = [pts[0][1], pts[1][1]]
        ax.plot(xs, ys, 'r--', linewidth=2, label='Cross-section')
        ax.scatter(xs, ys, c='red', s=100, zorder=5)
        ax.legend()
        plt.draw()
        plt.pause(1)
        plt.close()
        
        endpoints = [[xs[0], ys[0]], [xs[1], ys[1]]]
        self.xsec_line = LineString(endpoints)
        
        print(f"Cross-section line defined: {endpoints}")
        return endpoints #added
    
    def set_line_programmatic(self, endpoints: List[List[float]]):
        """
        Programmatically set the cross-section line.
        
        Parameters:
        -----------
        endpoints : list
            List of two coordinate pairs [[x1, y1], [x2, y2]]
        
        Example:
        --------
        >>> xsec.set_line_programmatic([[100, 200], [500, 600]])
        """
        if len(endpoints) != 2:
            raise ValueError("Endpoints must contain exactly 2 coordinate pairs")
        
        self.xsec_line = LineString(endpoints)
        print(f"Cross-section line set: {endpoints}")
        return endpoints
    
    def _point_to_line_distance(self, x: float, y: float) -> float:
        """
        Calculate perpendicular distance from a point to the cross-section line.
        
        Parameters:
        -----------
        x, y : float
            Point coordinates
            
        Returns:
        --------
        float
            Distance to line
        """
        if self.xsec_line is None:
            raise ValueError("Cross-section line not defined. Use set_line_interactive() or set_line_programmatic()")
        
        point = Point(x, y)
        return point.distance(self.xsec_line)
    
    def _project_point_to_line(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Project a point onto the cross-section line and calculate distance along line.
        
        Parameters:
        -----------
        x, y : float
            Point coordinates
            
        Returns:
        --------
        tuple
            (projected_x, projected_y, distance_along_line)
        """
        if self.xsec_line is None:
            raise ValueError("Cross-section line not defined")
        
        point = Point(x, y)
        
        # Get the projection distance along the line (0 to 1)
        line_length = self.xsec_line.length
        projected_distance = self.xsec_line.project(point)
        
        # Get the actual projected point
        projected_point = self.xsec_line.interpolate(projected_distance)
        
        return projected_point.x, projected_point.y, projected_distance
    
    def build_cross_section(self,
                           search_distance: float,
                           # endpoints: List[List[float]]= endpoints, #added
                           dem_path: Optional[str] = None,
                           sample_num: Optional[int] = None, #number of points to sample dem to build ground surface profile 
                           reference_elevation: float = 0.0,
                           use_elevation: bool = True) -> pd.DataFrame:
        """
        Build the cross-section by projecting data onto the line.
        
        Parameters:
        -----------
        search_distance : float
            Maximum perpendicular distance from line to include points
        dem_path : str, optional
            Path to DEM raster file for ground surface elevation
        reference_elevation : float
            Constant elevation to use if no DEM provided (default: 0.0)
        use_elevation : bool
            If True, convert depths to elevations (default: True)
            
        Returns:
        --------
        pd.DataFrame
            Cross-section data with projected coordinates
        """
        if self.xsec_line is None:
            raise ValueError("Cross-section line not defined. Use set_line_interactive() or set_line_programmatic()")
        
        self.search_distance = search_distance
        self.dem_path = dem_path
        self.reference_elevation = reference_elevation
        
        # Calculate distances from line
        distances = []
        for idx, row in self.data.iterrows():
            dist = self._point_to_line_distance(row[self.columns['x_coord']], 
                                               row[self.columns['y_coord']])
            distances.append(dist)
        
        self.data['dist_to_line'] = distances
        
        # Filter by search distance
        filtered = self.data[self.data['dist_to_line'] <= search_distance].copy()
        
        if len(filtered) == 0:
            warnings.warn(f"No points found within {search_distance} units of the line")
            self.xsec_data = pd.DataFrame()
            return self.xsec_data
        
        # Project points onto line
        projections = []
        for idx, row in filtered.iterrows():
            proj_x, proj_y, dist_along = self._project_point_to_line(
                row[self.columns['x_coord']], 
                row[self.columns['y_coord']]
            )
            projections.append({
                'proj_x': proj_x,
                'proj_y': proj_y,
                'dist_along_line': dist_along
            })
        
        proj_df = pd.DataFrame(projections)
        filtered = pd.concat([filtered.reset_index(drop=True), proj_df], axis=1)
        
        # Get ground surface elevations #to fix -- sample 100 pts along the line to plot
        if dem_path and RASTERIO_AVAILABLE:
            elevations = self._sample_dem(filtered[self.columns['x_coord']].values,
                                         filtered[self.columns['y_coord']].values)
            filtered['ground_elevation'] = elevations

            #get a sampled ============================================================================================
            # elevations_plot = self._sample_dem   #sampled 100 points along the cross-section line to sample the dem
            # elevations_dist_along_line = 
            # endpoints = [[xs[0], ys[0]], [xs[1], ys[1]]]
            end_coords = list(self.xsec_line.coords)
            try:
                xsample = np.linspace(end_coords[0][0], end_coords[1][0], num = sample_num)
                xsec_slope = (end_coords[1][1] - end_coords[0][1]) / (end_coords[1][0] - end_coords[0][0])
                y_intercept = end_coords[0][1] - (xsec_slope * xsample[0])
                ysample = (xsec_slope*xsample) + y_intercept
            
            except: #if a straight line
                print("using straight line exception...")
                xsample = np.ones(sample_num)*end_coords[0][0]
                ysample = np.linspace(end_coords[0][1], end_coords[1][1], num = sample_num)

            #sample elevation
            elevation_plot = self._sample_dem(xsample, ysample)
            coord_list = [(x, y) for x, y in zip(xsample, ysample)]
            elevation_dist_along_line = [math.hypot(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in zip(coord_list, coord_list[1:])]
            print(len(elevation_plot), len(elevation_dist_along_line)) #first needs to be zero
            
            self.elevation_plot = elevation_plot
            self.elevation_dist_along_line = np.cumsum([0.0, *elevation_dist_along_line])
            # print(self.elevation_plot, self.elevation_dist_along_line)
            #======================================================
            
        else:
            filtered['ground_elevation'] = reference_elevation
            elevation_plot = []
            elevation_dist_along_line = []
        
        # Calculate elevations for plotting
        if use_elevation:
            # Check if we have interval data or point data
            if self.columns['top_depth'] in filtered.columns and self.columns['bottom_depth'] in filtered.columns:
                # Interval data
                filtered['top_elevation'] = filtered['ground_elevation'] - filtered[self.columns['top_depth']]
                filtered['bottom_elevation'] = filtered['ground_elevation'] - filtered[self.columns['bottom_depth']]
            elif self.columns['depth'] in filtered.columns:
                # Point data
                filtered['elevation'] = filtered['ground_elevation'] - filtered[self.columns['depth']]
        
        self.xsec_data = filtered.sort_values('dist_along_line')
        
        print(f"Cross-section built: {len(self.xsec_data)} records from {self.xsec_data[self.columns['station_id']].nunique()} stations")
        
        return self.xsec_data, self.elevation_plot, self.elevation_dist_along_line # added
    
    def _sample_dem(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """
        Sample elevation values from a DEM raster.
        
        Parameters:
        -----------
        x_coords, y_coords : np.ndarray
            Coordinate arrays
            
        Returns:
        --------
        np.ndarray
            Elevation values
        """
        if not RASTERIO_AVAILABLE:
            warnings.warn("rasterio not available, using reference elevation")
            return np.full(len(x_coords), self.reference_elevation)
        
        try:
            with rasterio.open(self.dem_path) as src:
                coord_list = [(x, y) for x, y in zip(x_coords, y_coords)]
                elevations = [val[0] for val in src.sample(coord_list)]
                return np.array(elevations)
        except Exception as e:
            warnings.warn(f"Error reading DEM: {e}. Using reference elevation.")
            return np.full(len(x_coords), self.reference_elevation)
    
    def plot_cross_section(self,
                          color_scheme: Optional[Dict[str, str]] = None,
                          figsize: Tuple[float, float] = (14, 6),
                          vertical_exaggeration: float = 1.0,
                          bar_width: Optional[float] = None,
                          plot_ground_surface: bool = True,
                          ylabel: str = "Elevation",
                          title: Optional[str] = None,
                          savepath: Optional[str] = None,
                          dpi: int = 300) -> plt.Figure:
        """
        Plot the cross-section.
        
        Parameters:
        -----------
        color_scheme : dict, optional
            Dictionary mapping lithology descriptions to colors - allow for random entry
            Example: {'Sand': 'yellow', 'Clay': 'brown', 'Gravel': 'gray'}
        figsize : tuple
            Figure size (width, height)
        vertical_exaggeration : float
            Vertical exaggeration factor (default: 1.0)
        bar_width : float, optional
            Width of lithology bars. If None, calculated automatically
        plot_ground_surface : bool
            Whether to plot ground surface line
        ylabel : str
            Y-axis label
        title : str, optional
            Plot title
        savepath : str, optional
            Path to save the figure
        dpi : int
            Resolution for saved figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.xsec_data is None or len(self.xsec_data) == 0:
            raise ValueError("No cross-section data available. Run build_cross_section() first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine if we have interval or point data
        has_intervals = (self.columns['top_depth'] in self.xsec_data.columns and 
                        self.columns['bottom_depth'] in self.xsec_data.columns)
        
        # Calculate bar width if not provided
        if bar_width is None:
            unique_stations = self.xsec_data.groupby(self.columns['station_id'])['dist_along_line'].first().values
            if len(unique_stations) > 1:
                min_spacing = np.min(np.diff(np.sort(unique_stations)))
                bar_width = min_spacing * 0.8
            else:
                bar_width = self.xsec_line.length * 0.05
        
        # Set up default color scheme if not provided
        if color_scheme is None:
            unique_descriptions = self.xsec_data[self.columns['description']].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_descriptions)))
            color_scheme = {desc: colors[i] for i, desc in enumerate(unique_descriptions)}
            print(color_scheme) #debugging
        
        # Plot lithologic intervals or points
        if has_intervals:
            self._plot_intervals(ax, color_scheme, bar_width)
        else:
            self._plot_points(ax, color_scheme, bar_width)
        
        # Plot ground surface
        if plot_ground_surface:
            # print("why", self.elevation_plot)
            self._plot_ground_surface(ax)
        
        # Formatting
        ax.set_xlabel('Distance Along Section (units)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        if title is None:
            title = f'Cross-Section (Search Distance: {self.search_distance:.1f} units)'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Set limits
        ax.set_xlim(0, self.xsec_line.length)
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, facecolor=color_scheme.get(desc, 'gray'), edgecolor='black', linewidth=0.5) for desc in sorted(color_scheme.keys())]
        labels = sorted(color_scheme.keys())
        ax.legend(handles, labels, loc='best', ncol=min(3, len(labels)), fontsize=9, title='Lithology')
        
        plt.tight_layout()
        
        if savepath:
            fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to: {savepath}")
        
        return fig
    
    def _plot_intervals(self, ax, color_scheme, bar_width):
        """Plot lithologic intervals as colored rectangles."""
        for station_id in self.xsec_data[self.columns['station_id']].unique():
            station_data = self.xsec_data[self.xsec_data[self.columns['station_id']] == station_id]
            x_pos = station_data['dist_along_line'].iloc[0]
            
            for _, row in station_data.iterrows():
                desc = row[self.columns['description']]
                color = color_scheme.get(desc, 'lightgray')
                
                top = row['top_elevation']
                bottom = row['bottom_elevation']
                height = top - bottom
                
                rect = Rectangle((x_pos - bar_width/2, bottom), 
                               bar_width, height,
                               facecolor=color, 
                               edgecolor='black', 
                               linewidth=0.5)
                ax.add_patch(rect)
    
    def _plot_points(self, ax, color_scheme, bar_width):
        """Plot point data as markers or short bars."""
        for _, row in self.xsec_data.iterrows():
            desc = row[self.columns['description']]
            color = color_scheme.get(desc, 'lightgray')
            x_pos = row['dist_along_line']
            y_pos = row['elevation']
            
            # Plot as a small marker
            ax.scatter(x_pos, y_pos, c=[color], s=100, 
                      edgecolor='black', linewidth=0.5, zorder=3)
    
    def _plot_ground_surface(self, ax):
        """Plot the ground surface profile."""
        # Get unique stations and their ground elevations
        # station_data = self.xsec_data.groupby(self.columns['station_id']).agg({
        #     'dist_along_line': 'first',
        #     'ground_elevation': 'first'
        # }).sort_values('dist_along_line')
        
        # ax.plot(station_data['dist_along_line'], 
        #        station_data['ground_elevation'],
        #        color='saddlebrown', linewidth=2, 
        #        linestyle='-', alpha=0.7, label='Ground Surface')
        
        # print(self.elevation_plot, self.elevation_dist_along_line)
        #alternative plotting of sampled data along line=================================================
        ax.plot(self.elevation_dist_along_line, self.elevation_plot, color='saddlebrown', linewidth=2, linestyle='-', alpha=0.7, label='Ground Surface')
    
    def plot_map_view(self,
                     figsize: Tuple[float, float] = (8, 8),
                     show_all_data: bool = True,
                     background_data: Optional[pd.DataFrame] = None) -> plt.Figure:
        """
        Plot a map view showing the cross-section line and data points.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        show_all_data : bool
            Whether to show all data points or only those in cross-section
        background_data : pd.DataFrame, optional
            Additional background data to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.xsec_line is None:
            raise ValueError("Cross-section line not defined")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot background data
        if background_data is not None:
            ax.scatter(background_data[self.columns['x_coord']], 
                      background_data[self.columns['y_coord']],
                      c='lightgray', s=20, alpha=0.5, label='Background')
        
        # Plot all data or filtered data
        if show_all_data:
            ax.scatter(self.data[self.columns['x_coord']], 
                      self.data[self.columns['y_coord']],
                      c='blue', s=30, alpha=0.5, label='All Data')
        
        # Plot cross-section data
        if self.xsec_data is not None and len(self.xsec_data) > 0:
            unique_stations = self.xsec_data.groupby(self.columns['station_id']).first()
            ax.scatter(unique_stations[self.columns['x_coord']], 
                      unique_stations[self.columns['y_coord']],
                      c='red', s=50, edgecolor='black', 
                      linewidth=1, label='In Cross-Section', zorder=5)
        
        # Plot cross-section line
        coords = list(self.xsec_line.coords)
        ax.plot([coords[0][0], coords[1][0]], 
               [coords[0][1], coords[1][1]],
               'r--', linewidth=2, label='Section Line')
        ax.scatter([coords[0][0], coords[1][0]], 
                  [coords[0][1], coords[1][1]],
                  c='red', s=100, marker='^', 
                  edgecolor='black', linewidth=1, zorder=6)
        
        # Add labels at endpoints
        ax.annotate('A', xy=coords[0], xytext=(10, 10), 
                   textcoords='offset points', fontsize=14, 
                   fontweight='bold', color='red')
        ax.annotate("A'", xy=coords[1], xytext=(10, 10), 
                   textcoords='offset points', fontsize=14, 
                   fontweight='bold', color='red')
        
        ax.set_xlabel('X Coordinate', fontsize=11)
        ax.set_ylabel('Y Coordinate', fontsize=11)
        ax.set_title('Map View - Cross-Section Location', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        return fig
    
    def get_cross_section_data(self) -> pd.DataFrame:
        """
        Get the cross-section data as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Cross-section data with all calculated fields
        """
        if self.xsec_data is None:
            raise ValueError("No cross-section data available. Run build_cross_section() first")
        
        return self.xsec_data.copy()
    
    def export_cross_section(self, 
                            filepath: str,
                            format: str = 'csv') -> None:
        """
        Export cross-section data to file.
        
        Parameters:
        -----------
        filepath : str
            Output file path
        format : str
            Output format ('csv', 'excel', 'shapefile')
        """
        if self.xsec_data is None or len(self.xsec_data) == 0:
            raise ValueError("No cross-section data to export")
        
        if format == 'csv':
            self.xsec_data.to_csv(filepath, index=False)
        elif format == 'excel':
            self.xsec_data.to_excel(filepath, index=False)
        elif format == 'shapefile':
            # Create point geometries for shapefile
            gdf = gpd.GeoDataFrame(
                self.xsec_data,
                geometry=[Point(x, y) for x, y in zip(
                    self.xsec_data[self.columns['x_coord']], 
                    self.xsec_data[self.columns['y_coord']]
                )],
                crs='EPSG:4326'  # Adjust CRS as needed
            )
            gdf.to_file(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Cross-section data exported to: {filepath}")


# Utility functions for standalone use
def quick_cross_section(data: pd.DataFrame,
                       endpoints: List[List[float]],
                       search_distance: float,
                       **kwargs) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Quick function to create a cross-section with minimal setup.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    endpoints : list
        Cross-section line endpoints [[x1, y1], [x2, y2]]
    search_distance : float
        Search distance from line
    **kwargs : additional arguments passed to plot_cross_section()
    
    Returns:
    --------
    tuple
        (figure, cross_section_dataframe)
    
    Example:
    --------
    >>> df = pd.read_csv('borings.csv')
    >>> fig, xsec_data = quick_cross_section(
    ...     df, 
    ...     endpoints=[[100, 200], [500, 600]],
    ...     search_distance=50
    ... )
    >>> plt.show()
    """
    xsec = CrossSectionTool(data)
    xsec.set_line_programmatic(endpoints)
    xsec.build_cross_section(search_distance)
    fig = xsec.plot_cross_section(**kwargs)
    
    return fig, xsec.get_cross_section_data()


if __name__ == "__main__":
    # Example usage and testing
    print("Cross-Section Tool loaded successfully!")
    print("\nExample usage:")
    print("""
    # Load your data
    df = pd.read_csv('borehole_data.csv')
    
    # Create cross-section tool
    xsec = CrossSectionTool(df)
    
    # Option 1: Interactive line drawing
    xsec.set_line_interactive()
    
    # Option 2: Programmatic line definition
    # xsec.set_line_programmatic([[100, 200], [500, 600]])
    
    # Build cross-section
    xsec.build_cross_section(search_distance=100)
    
    # Plot
    fig = xsec.plot_cross_section()
    plt.show()
    
    # Get data
    xsec_data = xsec.get_cross_section_data()
    
    # Export
    xsec.export_cross_section('cross_section.csv')
    """)
