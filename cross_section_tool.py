"""
Cross-Section Tool for Geological/Lithologic Data
==================================================

A flexible tool for creating geological cross-sections from borehole lithologic data
or point measurements. Supports both interactive line drawing and programmatic input.

Author: Based on xsec_tool.py with enhanced functionality

# G. Schmeda added functionality to pull bathymetry nearest to the sample date of each core
# along the cross-section line and plot the sample data releative to the most relevant ground surface profile.
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
import os
import re
from datetime import datetime

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
        bathy_folder (str): Path to folder containing dated bathymetry rasters
        bathy_files (dict): Dictionary mapping years to bathymetry file paths
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, gpd.GeoDataFrame],
        column_mapping: Optional[Dict[str, str]] = None,
        bathy_folder: Optional[str] = None,
    ):
        """
        Initialize the CrossSectionTool.

        Parameters:
        -----------
        data : pd.DataFrame or gpd.GeoDataFrame
            Input data with spatial coordinates and lithologic/measurement information.
            If GeoDataFrame, will extract X/Y coordinates from geometry if not present.
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
                'value': 'Value',  # Optional numeric value for coloring
                'sample_date': 'SampleDate'  # Sample date for bathymetry matching
            }
        bathy_folder : str, optional
            Path to folder containing dated bathymetry rasters (.tif files).
            Files should have year in filename (e.g., 'bathy_2015.tif', '2020.tif')

        Example:
        --------
        >>> df = pd.read_csv('borehole_data.csv')
        >>> xsec = CrossSectionTool(df)
        >>> xsec.set_line_interactive()
        >>> xsec.plot_cross_section()

        >>> # Or with GeoDataFrame and bathymetry
        >>> gdf = gpd.read_file('cores.shp')
        >>> xsec = CrossSectionTool(gdf, bathy_folder='bathymetry_rasters/')
        """
        # Handle GeoDataFrame input
        if isinstance(data, gpd.GeoDataFrame):
            self.data = data.copy()
            # Extract X, Y from geometry if not present in columns
            if (
                "X_Coord" not in self.data.columns
                and "x_coord" not in self.data.columns
            ):
                self.data["X_Coord"] = self.data.geometry.x
            if (
                "Y_Coord" not in self.data.columns
                and "y_coord" not in self.data.columns
            ):
                self.data["Y_Coord"] = self.data.geometry.y
        else:
            self.data = data.copy()

        # Set up column mapping
        default_mapping = {
            "station_id": "StationID",
            "x_coord": "X_Coord",
            "y_coord": "Y_Coord",
            "top_depth": "Top_Depth",
            "bottom_depth": "Bottom_Depth",
            "depth": "Depth",
            "description": "Description",
            "value": "Value",
            "sample_date": "SampleDate",
        }

        if column_mapping:
            default_mapping.update(column_mapping)

        self.columns = default_mapping
        self.xsec_line = None
        self.xsec_data = None
        self.search_distance = None
        self.dem_path = None
        self.bathy_folder = bathy_folder
        self.bathy_files = {}  # Dictionary mapping year -> file path
        self.reference_elevation = 0.0
        # Ensure elevation arrays always exist to avoid AttributeError later when plotting multiple ground surfaces...
        self.elevation_plot = np.array([])
        self.elevation_dist_along_line = np.array([])
        self.ground_surfaces = (
            {}
        )  # Dictionary to store multiple ground surfaces by year

        # If bathymetry folder provided, discover files
        if self.bathy_folder:
            self._discover_bathy_files()

    def set_line_interactive(
        self,
        figsize: Tuple[float, float] = (10, 8),
        background_data: Optional[pd.DataFrame] = None,
    ):
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
        ax.scatter(
            plot_data[self.columns["x_coord"]],
            plot_data[self.columns["y_coord"]],
            c="blue",
            alpha=0.5,
            s=20,
            label="Data Points",
        )

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Click two points to define cross-section line")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

        # Get two points from user
        print("Click two points on the map to define the cross-section line...")
        pts = plt.ginput(2, timeout=0)

        if len(pts) < 2:
            raise ValueError("Need two points to define a line")

        # Draw the line
        xs = [pts[0][0], pts[1][0]]
        ys = [pts[0][1], pts[1][1]]
        ax.plot(xs, ys, "r--", linewidth=2, label="Cross-section")
        ax.scatter(xs, ys, c="red", s=100, zorder=5)
        ax.legend()
        plt.draw()
        plt.pause(1)
        plt.close()

        endpoints = [[xs[0], ys[0]], [xs[1], ys[1]]]
        self.xsec_line = LineString(endpoints)

        print(f"Cross-section line defined: {endpoints}")
        return endpoints  # added

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
            raise ValueError(
                "Cross-section line not defined. Use set_line_interactive() or set_line_programmatic()"
            )

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

    def _discover_bathy_files(self):
        """
        Discover and catalog bathymetry files in the provided folder.

        Expects files to have a 4-digit year in the filename
        (e.g., 'bathy_2015.tif', '2020_bathymetry.tif', 'dem_2018.tif')
        """
        if self.bathy_folder is None:
            return

        if not os.path.exists(self.bathy_folder):
            warnings.warn(f"Bathymetry folder not found: {self.bathy_folder}")
            return

        # Get all .tif files
        try:
            tif_files = [
                f for f in os.listdir(self.bathy_folder) if f.lower().endswith(".tif")
            ]
        except Exception as e:
            warnings.warn(f"Error reading bathymetry folder: {e}")
            return

        # Extract year from filename (4-digit pattern)
        year_pattern = r"(\d{4})"

        for filename in tif_files:
            match = re.search(year_pattern, filename)
            if match:
                year = int(match.group(1))
                # Validate year is reasonable (e.g., 1900-2100)
                if 1900 <= year <= 2100:
                    filepath = os.path.join(self.bathy_folder, filename)
                    # If multiple files have same year, keep the first one found
                    if year not in self.bathy_files:
                        self.bathy_files[year] = filepath

        if self.bathy_files:
            years = sorted(self.bathy_files.keys())
            print(f"Found {len(self.bathy_files)} bathymetry rasters: years {years}")
        else:
            warnings.warn(
                f"No bathymetry files with year patterns found in {self.bathy_folder}"
            )

    def _get_closest_bathy_year(self, sample_date) -> Optional[int]:
        """
        Find the closest bathymetry year for a given sample date.

        Parameters:
        -----------
        sample_date : datetime, str, int, or float
            Sample date/year. Can be:
            - Integer year (e.g., 2015)
            - Date string (e.g., '2015-06-20', '06/20/2015')
            - datetime object

        Returns:
        --------
        int or None
            Closest year with bathymetry data, or None if no bathy files
        """
        if not self.bathy_files:
            return None

        # Convert sample_date to year
        sample_year = None

        if pd.isna(sample_date):
            return None

        if isinstance(sample_date, (int, float)):
            sample_year = int(sample_date)
        elif isinstance(sample_date, str):
            # Try to parse date string
            try:
                sample_year = pd.to_datetime(sample_date).year
            except:
                # Try to extract year from string
                match = re.search(r"(\d{4})", sample_date)
                if match:
                    sample_year = int(match.group(1))
                else:
                    warnings.warn(f"Could not parse year from: {sample_date}")
                    return None
        else:
            # Assume it's a datetime-like object
            try:
                sample_year = pd.to_datetime(sample_date).year
            except:
                warnings.warn(f"Could not extract year from: {sample_date}")
                return None

        # Find closest year
        available_years = sorted(self.bathy_files.keys())
        closest_year = min(available_years, key=lambda x: abs(x - sample_year))

        return closest_year

    def build_cross_section(
        self,
        search_distance: float,
        # endpoints: List[List[float]]= endpoints, #added
        dem_path: Optional[str] = None,
        sample_num: int = 100,  # number of points to sample dem to build ground surface profile
        reference_elevation: float = 0.0,
        use_elevation: bool = True,
    ) -> pd.DataFrame:
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
            raise ValueError(
                "Cross-section line not defined. Use set_line_interactive() or set_line_programmatic()"
            )

        self.search_distance = search_distance
        self.dem_path = dem_path
        self.reference_elevation = reference_elevation

        # Calculate distances from line
        distances = []
        for idx, row in self.data.iterrows():
            dist = self._point_to_line_distance(
                row[self.columns["x_coord"]], row[self.columns["y_coord"]]
            )
            distances.append(dist)

        self.data["dist_to_line"] = distances

        # Filter by search distance
        filtered = self.data[self.data["dist_to_line"] <= search_distance].copy()

        if len(filtered) == 0:
            warnings.warn(f"No points found within {search_distance} units of the line")
            self.xsec_data = pd.DataFrame()
            return self.xsec_data

        # Project points onto line
        projections = []
        for idx, row in filtered.iterrows():
            proj_x, proj_y, dist_along = self._project_point_to_line(
                row[self.columns["x_coord"]], row[self.columns["y_coord"]]
            )
            projections.append(
                {"proj_x": proj_x, "proj_y": proj_y, "dist_along_line": dist_along}
            )

        proj_df = pd.DataFrame(projections)
        filtered = pd.concat([filtered.reset_index(drop=True), proj_df], axis=1)

        # Get ground surface elevations
        # If using dated bathymetry, match each station to nearest year
        if self.bathy_files and self.columns["sample_date"] in filtered.columns:
            print("Using dated bathymetry rasters based on sample dates...")
            elevations = []
            filtered["bathy_year"] = pd.Series(
                [None] * len(filtered), dtype="Int64"
            )  # Track which year was used

            for idx, row in filtered.iterrows():
                sample_date = row[self.columns["sample_date"]]
                closest_year = self._get_closest_bathy_year(sample_date)

                if closest_year is not None:
                    bathy_path = self.bathy_files[closest_year]
                    elev = self._sample_dem(
                        np.array([row[self.columns["x_coord"]]]),
                        np.array([row[self.columns["y_coord"]]]),
                        dem_path=bathy_path,
                    )[0]
                    filtered.loc[idx, "bathy_year"] = closest_year
                else:
                    elev = self.reference_elevation

                elevations.append(elev)

            filtered["ground_elevation"] = elevations

            # Report which years were used
            years_used = filtered["bathy_year"].dropna().unique()
            if len(years_used) > 0:
                print(f"Used bathymetry from years: {sorted(years_used)}")

        # Otherwise use single DEM if provided
        elif dem_path and RASTERIO_AVAILABLE:
            elevations = self._sample_dem(
                np.asarray(filtered[self.columns["x_coord"]].values),
                np.asarray(filtered[self.columns["y_coord"]].values),
                dem_path=dem_path,
            )
            filtered["ground_elevation"] = elevations

            # Sample elevation along the cross-section line for continuous ground surface
            end_coords = list(self.xsec_line.coords)
            try:
                xsample = np.linspace(
                    end_coords[0][0], end_coords[1][0], num=sample_num
                )
                xsec_slope = (end_coords[1][1] - end_coords[0][1]) / (
                    end_coords[1][0] - end_coords[0][0]
                )
                y_intercept = end_coords[0][1] - (xsec_slope * xsample[0])
                ysample = (xsec_slope * xsample) + y_intercept

            except:  # if a vertical line
                print("Using vertical line sampling...")
                xsample = np.ones(sample_num) * end_coords[0][0]
                ysample = np.linspace(
                    end_coords[0][1], end_coords[1][1], num=sample_num
                )

            coord_list = [(x, y) for x, y in zip(xsample, ysample)]
            elevation_dist_along_line = [
                math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                for p1, p2 in zip(coord_list, coord_list[1:])
            ]

            # If using dated bathymetry, create ground surfaces for each unique year
            if self.bathy_files and "bathy_year" in filtered.columns:
                unique_years = sorted(filtered["bathy_year"].dropna().unique())

                for year in unique_years:
                    year = int(year)
                    bathy_path = self.bathy_files[year]
                    elevation_plot_year = self._sample_dem(
                        xsample, ysample, dem_path=bathy_path
                    )
                    dist_along = np.cumsum([0.0, *elevation_dist_along_line])
                    self.ground_surfaces[year] = {
                        "elevation": elevation_plot_year,
                        "distance": dist_along,
                    }
                    print(f"Created ground surface profile for year {year}")

                # Set default to most recent year for backward compatibility
                if unique_years:
                    most_recent = max(unique_years)
                    self.elevation_plot = self.ground_surfaces[int(most_recent)][
                        "elevation"
                    ]
                    self.elevation_dist_along_line = self.ground_surfaces[
                        int(most_recent)
                    ]["distance"]
            else:
                # Single DEM sampling
                elevation_plot = self._sample_dem(xsample, ysample, dem_path=dem_path)
                self.elevation_plot = elevation_plot
                self.elevation_dist_along_line = np.cumsum(
                    [0.0, *elevation_dist_along_line]
                )

        else:
            filtered["ground_elevation"] = reference_elevation
            elevation_plot = []
            elevation_dist_along_line = []

        # Calculate elevations for plotting
        if use_elevation:
            # Check if we have interval data or point data
            if (
                self.columns["top_depth"] in filtered.columns
                and self.columns["bottom_depth"] in filtered.columns
            ):
                # Interval data
                filtered["top_elevation"] = (
                    filtered["ground_elevation"] - filtered[self.columns["top_depth"]]
                )
                filtered["bottom_elevation"] = (
                    filtered["ground_elevation"]
                    - filtered[self.columns["bottom_depth"]]
                )
            elif self.columns["depth"] in filtered.columns:
                # Point data
                filtered["elevation"] = (
                    filtered["ground_elevation"] - filtered[self.columns["depth"]]
                )

        self.xsec_data = filtered.sort_values("dist_along_line")

        print(
            f"Cross-section built: {len(self.xsec_data)} records from {self.xsec_data[self.columns['station_id']].nunique()} stations"
        )

        return self.xsec_data

    def _sample_dem(
        self, x_coords: np.ndarray, y_coords: np.ndarray, dem_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Sample elevation values from a DEM raster.

        Parameters:
        -----------
        x_coords, y_coords : np.ndarray
            Coordinate arrays
        dem_path : str, optional
            Path to specific DEM file. If None, uses self.dem_path

        Returns:
        --------
        np.ndarray
            Elevation values
        """
        if not RASTERIO_AVAILABLE:
            warnings.warn("rasterio not available, using reference elevation")
            return np.full(len(x_coords), self.reference_elevation)

        # Use provided path or fall back to instance path
        path_to_use = dem_path if dem_path is not None else self.dem_path

        if path_to_use is None:
            return np.full(len(x_coords), self.reference_elevation)

        try:
            with rasterio.open(path_to_use) as src:
                coord_list = [(x, y) for x, y in zip(x_coords, y_coords)]
                elevations = [val[0] for val in src.sample(coord_list)]
                elevations = np.array(elevations)

                # Get NoData value from raster metadata
                nodata_value = src.nodata

                # Replace NoData values with NaN
                if nodata_value is not None:
                    elevations = np.where(
                        elevations == nodata_value, np.nan, elevations
                    )

                # Also handle common NoData sentinels
                elevations = np.where(elevations <= -9999, np.nan, elevations)

                return elevations
        except Exception as e:
            warnings.warn(
                f"Error reading DEM {path_to_use}: {e}. Using reference elevation."
            )
            return np.full(len(x_coords), self.reference_elevation)

    def generate_ground_surfaces(
        self,
        sample_num: int = 200,
        years: Optional[List[int]] = None,
        use_all_files: bool = False,
    ):
        """
        Create continuous ground-surface profiles by sampling each bathymetry raster
        along the current cross-section line.

        Call this after build_cross_section() to populate self.ground_surfaces for
        use with plot_all_ground_surfaces=True.

        Parameters:
        -----------
        sample_num : int
            Number of points to sample along the cross-section line (default: 200)
        years : list of int, optional
            Specific years to create profiles for. If None, uses years from xsec_data
        use_all_files : bool
            If True, create profiles for all discovered bathymetry files.
            If False (default), only create profiles for years matched to samples

        Example:
        --------
        >>> xsec.build_cross_section(search_distance=200)
        >>> xsec.generate_ground_surfaces(sample_num=300, use_all_files=False)
        >>> xsec.plot_cross_section(plot_all_ground_surfaces=True)
        """
        if not RASTERIO_AVAILABLE:
            warnings.warn("rasterio not available; cannot sample bathymetry rasters.")
            return

        if not hasattr(self, "xsec_line") or self.xsec_line is None:
            raise RuntimeError(
                "Cross-section line not set. Call set_line_programmatic() or set_line_interactive() first."
            )

        # Ensure container exists
        if not hasattr(self, "ground_surfaces") or self.ground_surfaces is None:
            self.ground_surfaces = {}

        # Decide which years to create profiles for
        available_years = sorted(
            [int(y) for y in getattr(self, "bathy_files", {}).keys()]
        )

        if years is None:
            if use_all_files:
                years_to_build = available_years
            else:
                # Take years matched to samples (if present)
                years_col = getattr(self, "xsec_data", pd.DataFrame()).get("bathy_year")
                if years_col is None or years_col.dropna().empty:
                    years_to_build = []
                else:
                    years_to_build = sorted(int(y) for y in years_col.dropna().unique())
        else:
            years_to_build = [int(y) for y in years if int(y) in available_years]

        if not years_to_build:
            print("No bathymetry years to build profiles for.")
            return

        # Sample coordinates along the line
        length = self.xsec_line.length
        distances_along = np.linspace(0.0, length, sample_num)
        points = [self.xsec_line.interpolate(d) for d in distances_along]
        xs = np.array([p.x for p in points])
        ys = np.array([p.y for p in points])

        # Sample each requested bathy raster
        for year in sorted(years_to_build):
            bathy_path = self.bathy_files.get(int(year))
            if bathy_path is None:
                continue
            try:
                elev = self._sample_dem(xs, ys, dem_path=bathy_path)
                # Store profile keyed by year (int)
                self.ground_surfaces[int(year)] = {
                    "distance": distances_along,
                    "elevation": np.array(elev),
                }
                print(f"Created ground surface profile for year {year}")
            except Exception as e:
                warnings.warn(
                    f"Failed to sample bathymetry for year {year} ({bathy_path}): {e}"
                )

        # Set default plotting profile to most recent year (if any)
        if self.ground_surfaces:
            most_recent = max(self.ground_surfaces.keys())
            prof = self.ground_surfaces[most_recent]
            self.elevation_plot = prof["elevation"]
            self.elevation_dist_along_line = prof["distance"]
            print(f"Set default profile to year {most_recent}")

    def plot_cross_section(
        self,
        color_scheme: Optional[Dict[str, str]] = None,
        figsize: Tuple[float, float] = (14, 6),
        vertical_exaggeration: float = 1.0,
        bar_width: Optional[float] = None,
        plot_ground_surface: bool = True,
        plot_all_ground_surfaces: bool = False,
        ylabel: str = "Elevation",
        depth_ylabel: str = "Depth (units)",
        title: Optional[str] = None,
        savepath: Optional[str] = None,
        dpi: int = 300,
        color_by: Optional[str] = None,
        value_threshold: Optional[float] = None,
        value_bins: Optional[List[float]] = None,
        value_colors: Optional[List[str]] = None,
        show_station_ids: bool = False,
        show_sample_dates: bool = False,
        show_bathy_years: bool = False,
        annotation_fontsize: int = 7,
    ) -> plt.Figure:
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
            Whether to plot ground surface line (default: True)
        plot_all_ground_surfaces : bool
            If True and multiple dated bathymetry surfaces exist, plot all of them.
            If False, only plot the most recent one (default: False)
        ylabel : str
            Y-axis label for elevation plot
        depth_ylabel : str
            Y-axis label for depth subplot (default: "Depth (units)")
        title : str, optional
            Plot title
        savepath : str, optional
            Path to save the figure
        dpi : int
            Resolution for saved figure
        color_by : str, optional
            'description' (default) or 'value' - determines coloring mode
        value_threshold : float, optional
            Simple binary threshold for value-based coloring (default: 1000)
            Values > threshold = red, values <= threshold = green
        value_bins : list of float, optional
            List of ascending thresholds for multi-range value coloring
            Example: [500, 1000, 2000] creates ranges: (-inf,500], (500,1000], (1000,2000], (2000,inf)
        value_colors : list of str, optional
            Colors for each range (length must equal len(value_bins) + 1)
            Example: ['blue', 'green', 'orange', 'red'] for 3 bins
        show_station_ids : bool
            If True, display station IDs above each core/station (default: False)
        show_sample_dates : bool
            If True, display sample dates above each core/station (default: False)
        show_bathy_years : bool
            If True, display the matched bathymetry year for each station (default: False)
        annotation_fontsize : int
            Font size for station labels (default: 7)

        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.xsec_data is None or len(self.xsec_data) == 0:
            raise ValueError(
                "No cross-section data available. Run build_cross_section() first"
            )

        # Store value-based coloring parameters for helper methods
        self._color_by = color_by
        self._value_threshold = (
            value_threshold if value_threshold is not None else 1000.0
        )
        self._value_bins = sorted(value_bins) if value_bins else None
        self._value_colors = value_colors

        # Determine if we have interval or point data
        has_intervals = (
            self.columns["top_depth"] in self.xsec_data.columns
            and self.columns["bottom_depth"] in self.xsec_data.columns
        )

        # Create subplots: 2 rows if using value-based coloring with intervals, 1 otherwise
        if color_by == "value" and has_intervals:
            fig, (ax, ax_depth) = plt.subplots(
                2,
                1,
                figsize=(figsize[0], figsize[1] * 1.5),
                sharex=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax_depth = None

        # Calculate bar width if not provided
        if bar_width is None:
            unique_stations = (
                self.xsec_data.groupby(self.columns["station_id"])["dist_along_line"]
                .first()
                .values
            )
            if len(unique_stations) > 1:
                min_spacing = np.min(np.diff(np.sort(unique_stations)))
                bar_width = min_spacing * 0.8
            else:
                bar_width = self.xsec_line.length * 0.05

        # Set up default color scheme if not provided (skip if using pure value-based coloring)
        if color_scheme is None and color_by != "value":
            unique_descriptions = self.xsec_data[self.columns["description"]].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_descriptions)))
            color_scheme = {
                desc: colors[i] for i, desc in enumerate(unique_descriptions)
            }
            print(color_scheme)  # debugging

        # Align bars with ground surface line if profiles exist
        # Interpolate ground elevation from the profile at each sample's position
        if self.ground_surfaces and has_intervals:
            # Use the most recent year's profile for interpolation
            most_recent_year = max(self.ground_surfaces.keys())
            profile = self.ground_surfaces[most_recent_year]

            # Create a working copy to avoid modifying original data
            plot_data = self.xsec_data.copy()

            # Interpolate ground elevation at each sample's distance along line
            interpolated_elevations = np.interp(
                plot_data["dist_along_line"], profile["distance"], profile["elevation"]
            )

            # Recalculate elevations using interpolated ground surface
            plot_data["top_elevation"] = (
                interpolated_elevations - plot_data[self.columns["top_depth"]]
            )
            plot_data["bottom_elevation"] = (
                interpolated_elevations - plot_data[self.columns["bottom_depth"]]
            )

            # Temporarily replace xsec_data for plotting
            original_xsec_data = self.xsec_data
            self.xsec_data = plot_data

        # Plot lithologic intervals or points
        if has_intervals:
            self._plot_intervals(ax, color_scheme, bar_width)
        else:
            self._plot_points(ax, color_scheme, bar_width)

        # Plot ground surface(s)
        if plot_ground_surface:
            if plot_all_ground_surfaces and self.ground_surfaces:
                self._plot_all_ground_surfaces(ax)
            else:
                self._plot_ground_surface(ax)

        # Restore original xsec_data if we modified it for alignment
        if self.ground_surfaces and has_intervals and "original_xsec_data" in locals():
            self.xsec_data = original_xsec_data

        # Formatting
        if ax_depth is None:
            # Single plot - add xlabel to main axis
            ax.set_xlabel("Distance Along Section (units)", fontsize=11)
        # If dual plots, xlabel will be added to bottom plot
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        if title is None:
            title = f"Cross-Section (Search Distance: {self.search_distance:.1f} units)"
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Set limits
        ax.set_xlim(0, self.xsec_line.length)

        # Calculate and set y-axis limits based on data
        # matplotlib doesn't auto-scale for patches, so we calculate from data
        y_values = []

        if has_intervals:
            if (
                "top_elevation" in self.xsec_data.columns
                and "bottom_elevation" in self.xsec_data.columns
            ):
                y_values.extend(self.xsec_data["top_elevation"].values)
                y_values.extend(self.xsec_data["bottom_elevation"].values)
        else:
            if "elevation" in self.xsec_data.columns:
                y_values.extend(self.xsec_data["elevation"].values)

        # Include ground surface elevations if available
        if plot_ground_surface and len(self.elevation_plot) > 0:
            y_values.extend(self.elevation_plot)

        # Set y-limits with some padding
        if y_values:
            # Filter out NaN and invalid values
            y_values_clean = [v for v in y_values if not np.isnan(v) and v > -9999]
            if y_values_clean:
                y_min = np.min(y_values_clean)
                y_max = np.max(y_values_clean)
                y_range = y_max - y_min if y_max != y_min else 10
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

        # Plot depth subplot if using value-based coloring with intervals
        if ax_depth is not None:
            self._plot_intervals_depth(ax_depth, bar_width)
            ax_depth.set_xlabel("Distance Along Section (units)", fontsize=11)
            ax_depth.set_ylabel(depth_ylabel, fontsize=11)
            ax_depth.grid(True, alpha=0.3, axis="y")
            ax_depth.set_xlim(0, self.xsec_line.length)
            ax_depth.invert_yaxis()  # Depth increases downward

            # Set depth limits based on original depth values
            depth_values = []
            depth_values.extend(self.xsec_data[self.columns["top_depth"]].values)
            depth_values.extend(self.xsec_data[self.columns["bottom_depth"]].values)
            if depth_values:
                depth_min = np.min(depth_values)
                depth_max = np.max(depth_values)
                depth_range = depth_max - depth_min if depth_max != depth_min else 1
                ax_depth.set_ylim(
                    depth_max + 0.05 * depth_range, depth_min - 0.05 * depth_range
                )

        # Add station annotations if requested
        if show_station_ids or show_sample_dates or show_bathy_years:
            self._add_station_annotations(
                ax,
                show_station_ids,
                show_sample_dates,
                show_bathy_years,
                annotation_fontsize,
            )

        # Add legend
        if color_scheme is not None and len(color_scheme) > 0:
            handles = [
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=color_scheme.get(desc, "gray"),
                    edgecolor="black",
                    linewidth=0.5,
                )
                for desc in sorted(color_scheme.keys())
            ]
            labels = sorted(color_scheme.keys())

            # Add ground surface to legend if plotted
            if plot_ground_surface:
                from matplotlib.lines import Line2D

                if plot_all_ground_surfaces and self.ground_surfaces:
                    # Add all ground surface profiles
                    colors_gs = plt.cm.viridis(
                        np.linspace(0, 1, len(self.ground_surfaces))
                    )
                    for idx, year in enumerate(sorted(self.ground_surfaces.keys())):
                        handles.append(
                            Line2D(
                                [0], [0], color=colors_gs[idx], linewidth=2, alpha=0.7
                            )
                        )
                        labels.append(f"Ground Surface {year}")
                elif len(self.elevation_plot) > 0:
                    # Add single ground surface
                    handles.append(
                        Line2D([0], [0], color="saddlebrown", linewidth=2, alpha=0.7)
                    )
                    labels.append("Ground Surface")

            ax.legend(
                handles,
                labels,
                loc="best",
                ncol=min(3, len(labels)),
                fontsize=9,
                title="Lithology",
            )
        elif color_by == "value":
            # Create legend for value-based coloring
            if self._value_bins and self._value_colors:
                # Multi-range legend
                handles = []
                labels = []
                bins = self._value_bins
                colors = self._value_colors

                # First range: -inf to first bin
                handles.append(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=colors[0],
                        edgecolor="black",
                        linewidth=0.5,
                    )
                )
                labels.append(f"≤ {bins[0]}")

                # Middle ranges
                for i in range(len(bins) - 1):
                    handles.append(
                        plt.Rectangle(
                            (0, 0),
                            1,
                            1,
                            facecolor=colors[i + 1],
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    )
                    labels.append(f"{bins[i]} - {bins[i+1]}")

                # Last range: last bin to +inf
                handles.append(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=colors[-1],
                        edgecolor="black",
                        linewidth=0.5,
                    )
                )
                labels.append(f"> {bins[-1]}")

                # Add ground surface to legend if plotted
                if plot_ground_surface:
                    from matplotlib.lines import Line2D

                    if plot_all_ground_surfaces and self.ground_surfaces:
                        # Add all ground surface profiles
                        colors_gs = plt.cm.viridis(
                            np.linspace(0, 1, len(self.ground_surfaces))
                        )
                        for idx, year in enumerate(sorted(self.ground_surfaces.keys())):
                            handles.append(
                                Line2D(
                                    [0],
                                    [0],
                                    color=colors_gs[idx],
                                    linewidth=2,
                                    alpha=0.7,
                                )
                            )
                            labels.append(f"Ground Surface {year}")
                    elif len(self.elevation_plot) > 0:
                        # Add single ground surface
                        handles.append(
                            Line2D(
                                [0], [0], color="saddlebrown", linewidth=2, alpha=0.7
                            )
                        )
                        labels.append("Ground Surface")

                ax.legend(
                    handles, labels, loc="best", ncol=1, fontsize=9, title="Value Range"
                )
            else:
                # Binary threshold legend
                thresh = self._value_threshold
                handles = [
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor="green",
                        edgecolor="black",
                        linewidth=0.5,
                    ),
                    plt.Rectangle(
                        (0, 0), 1, 1, facecolor="red", edgecolor="black", linewidth=0.5
                    ),
                ]
                labels = [f"≤ {thresh}", f"> {thresh}"]

                # Add ground surface to legend if plotted
                if plot_ground_surface:
                    from matplotlib.lines import Line2D

                    if plot_all_ground_surfaces and self.ground_surfaces:
                        # Add all ground surface profiles
                        colors_gs = plt.cm.viridis(
                            np.linspace(0, 1, len(self.ground_surfaces))
                        )
                        for idx, year in enumerate(sorted(self.ground_surfaces.keys())):
                            handles.append(
                                Line2D(
                                    [0],
                                    [0],
                                    color=colors_gs[idx],
                                    linewidth=2,
                                    alpha=0.7,
                                )
                            )
                            labels.append(f"Ground Surface {year}")
                    elif len(self.elevation_plot) > 0:
                        # Add single ground surface
                        handles.append(
                            Line2D(
                                [0], [0], color="saddlebrown", linewidth=2, alpha=0.7
                            )
                        )
                        labels.append("Ground Surface")

                ax.legend(
                    handles, labels, loc="best", ncol=1, fontsize=9, title="Value Range"
                )

        plt.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
            print(f"Figure saved to: {savepath}")

        return fig

    def _get_color_for_value(self, val, default="lightgray"):
        """
        Return color string for numeric value based on bins/colors or threshold.

        Parameters:
        -----------
        val : float
            The numeric value to color
        default : str
            Default color for missing/NaN values

        Returns:
        --------
        str
            Color string
        """
        # Handle None/NaN values
        try:
            if val is None or pd.isna(val):
                return default
        except Exception:
            return default

        # Bins-based mapping (preferred if specified)
        if (
            getattr(self, "_value_bins", None) is not None
            and getattr(self, "_value_colors", None) is not None
        ):
            bins = self._value_bins
            colors = self._value_colors
            # Find index: number of bin thresholds less than value
            idx = sum(val > b for b in bins)
            # Safeguard: colors length should be len(bins)+1
            if idx < len(colors):
                return colors[idx]
            else:
                return colors[-1]

        # Fallback: simple binary threshold
        thresh = getattr(self, "_value_threshold", 1000.0)
        return "red" if val > thresh else "green"

    def _plot_intervals(self, ax, color_scheme, bar_width):
        """Plot lithologic intervals as colored rectangles."""
        for station_id in self.xsec_data[self.columns["station_id"]].unique():
            station_data = self.xsec_data[
                self.xsec_data[self.columns["station_id"]] == station_id
            ]
            x_pos = station_data["dist_along_line"].iloc[0]

            for _, row in station_data.iterrows():
                desc = row[self.columns["description"]]

                # Determine color: value-based or description-based
                if (
                    getattr(self, "_color_by", None) == "value"
                    and self.columns.get("value") in row.index
                ):
                    val = row.get(self.columns["value"], None)
                    color = self._get_color_for_value(val)
                else:
                    color = (
                        color_scheme.get(desc, "lightgray")
                        if color_scheme
                        else "lightgray"
                    )

                top = row["top_elevation"]
                bottom = row["bottom_elevation"]
                height = top - bottom

                rect = Rectangle(
                    (x_pos - bar_width / 2, bottom),
                    bar_width,
                    height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(rect)

    def _plot_intervals_depth(self, ax, bar_width):
        """Plot lithologic intervals using original depth values (not elevation-corrected)."""
        for station_id in self.xsec_data[self.columns["station_id"]].unique():
            station_data = self.xsec_data[
                self.xsec_data[self.columns["station_id"]] == station_id
            ]
            x_pos = station_data["dist_along_line"].iloc[0]

            for _, row in station_data.iterrows():
                # Determine color using value
                if self.columns.get("value") in row.index:
                    val = row.get(self.columns["value"], None)
                    color = self._get_color_for_value(val)
                else:
                    color = "lightgray"

                # Use original depth values (not elevation-corrected)
                top = row[self.columns["top_depth"]]
                bottom = row[self.columns["bottom_depth"]]
                height = bottom - top  # depth increases downward

                rect = Rectangle(
                    (x_pos - bar_width / 2, top),
                    bar_width,
                    height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(rect)

    def _plot_points(self, ax, color_scheme, bar_width):
        """Plot point data as markers or short bars."""
        for _, row in self.xsec_data.iterrows():
            desc = row[self.columns["description"]]

            # Determine color: value-based or description-based
            if (
                getattr(self, "_color_by", None) == "value"
                and self.columns.get("value") in row.index
            ):
                val = row.get(self.columns["value"], None)
                color = self._get_color_for_value(val)
            else:
                color = (
                    color_scheme.get(desc, "lightgray") if color_scheme else "lightgray"
                )

            x_pos = row["dist_along_line"]
            y_pos = row.get("elevation", row.get("top_elevation", None))

            # Plot as a small marker
            ax.scatter(
                x_pos,
                y_pos,
                c=[color],
                s=100,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

    def _plot_ground_surface(self, ax):
        """Plot the ground surface profile (single surface or most recent)."""
        if len(self.elevation_plot) > 0 and len(self.elevation_dist_along_line) > 0:
            ax.plot(
                self.elevation_dist_along_line,
                self.elevation_plot,
                color="saddlebrown",
                linewidth=2,
                linestyle="-",
                alpha=0.7,
                label="Ground Surface",
            )

    def _plot_all_ground_surfaces(self, ax):
        """Plot all dated ground surface profiles."""
        if not self.ground_surfaces:
            self._plot_ground_surface(ax)
            return

        # Color scheme for different years
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.ground_surfaces)))

        for idx, (year, surface_data) in enumerate(
            sorted(self.ground_surfaces.items())
        ):
            ax.plot(
                surface_data["distance"],
                surface_data["elevation"],
                color=colors[idx],
                linewidth=2,
                linestyle="-",
                alpha=0.7,
                label=f"Ground Surface {year}",
            )

        print(f"Plotted {len(self.ground_surfaces)} ground surface profiles")

    def _add_station_annotations(
        self,
        ax,
        show_station_ids: bool,
        show_sample_dates: bool,
        show_bathy_years: bool,
        fontsize: int,
    ):
        """Add text annotations for station IDs, sample dates, and/or bathymetry years."""
        # Group by station to avoid duplicate labels
        for station_id in self.xsec_data[self.columns["station_id"]].unique():
            station_data = self.xsec_data[
                self.xsec_data[self.columns["station_id"]] == station_id
            ]

            # Get the position along the line (use first occurrence)
            dist_along = station_data["dist_along_line"].iloc[0]

            # Get the highest elevation for this station to place label above it
            if "ground_elevation" in station_data.columns:
                y_position = station_data["ground_elevation"].iloc[0]
            elif "plot_elevation" in station_data.columns:
                y_position = station_data["plot_elevation"].max()
            else:
                # Fallback: use the minimum depth (highest point)
                if self.columns["top_depth"] in station_data.columns:
                    y_position = -station_data[self.columns["top_depth"]].min()
                else:
                    y_position = 0

            # Build annotation text
            label_parts = []

            if show_station_ids:
                label_parts.append(f"{station_id}")

            if (
                show_sample_dates
                and self.columns["sample_date"] in station_data.columns
            ):
                sample_date = station_data[self.columns["sample_date"]].iloc[0]
                if pd.notna(sample_date):
                    # Format date nicely if it's a datetime
                    if isinstance(sample_date, (pd.Timestamp, datetime)):
                        date_str = sample_date.strftime("%Y-%m-%d")
                    else:
                        date_str = str(sample_date)
                    label_parts.append(date_str)

            if show_bathy_years and "bathy_year" in station_data.columns:
                bathy_year = station_data["bathy_year"].iloc[0]
                if pd.notna(bathy_year):
                    label_parts.append(f"Bathy: {int(bathy_year)}")

            if label_parts:
                label_text = "\n".join(label_parts)
                ax.annotate(
                    label_text,
                    xy=(dist_along, y_position),
                    xytext=(0, 5),  # Offset 5 points above
                    textcoords="offset points",
                    fontsize=fontsize,
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="gray",
                        alpha=0.7,
                    ),
                    rotation=90,
                )

    def plot_map_view(
        self,
        figsize: Tuple[float, float] = (8, 8),
        show_all_data: bool = True,
        background_data: Optional[pd.DataFrame] = None,
    ) -> plt.Figure:
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
            ax.scatter(
                background_data[self.columns["x_coord"]],
                background_data[self.columns["y_coord"]],
                c="lightgray",
                s=20,
                alpha=0.5,
                label="Background",
            )

        # Plot all data or filtered data
        if show_all_data:
            ax.scatter(
                self.data[self.columns["x_coord"]],
                self.data[self.columns["y_coord"]],
                c="blue",
                s=30,
                alpha=0.5,
                label="All Data",
            )

        # Plot cross-section data
        if self.xsec_data is not None and len(self.xsec_data) > 0:
            unique_stations = self.xsec_data.groupby(self.columns["station_id"]).first()
            ax.scatter(
                unique_stations[self.columns["x_coord"]],
                unique_stations[self.columns["y_coord"]],
                c="red",
                s=50,
                edgecolor="black",
                linewidth=1,
                label="In Cross-Section",
                zorder=5,
            )

        # Plot cross-section line
        coords = list(self.xsec_line.coords)
        ax.plot(
            [coords[0][0], coords[1][0]],
            [coords[0][1], coords[1][1]],
            "r--",
            linewidth=2,
            label="Section Line",
        )
        ax.scatter(
            [coords[0][0], coords[1][0]],
            [coords[0][1], coords[1][1]],
            c="red",
            s=100,
            marker="^",
            edgecolor="black",
            linewidth=1,
            zorder=6,
        )

        # Add labels at endpoints
        ax.annotate(
            "A",
            xy=coords[0],
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            color="red",
        )
        ax.annotate(
            "A'",
            xy=coords[1],
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            color="red",
        )

        ax.set_xlabel("X Coordinate", fontsize=11)
        ax.set_ylabel("Y Coordinate", fontsize=11)
        ax.set_title(
            "Map View - Cross-Section Location", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_aspect("equal")

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
            raise ValueError(
                "No cross-section data available. Run build_cross_section() first"
            )

        return self.xsec_data.copy()

    def export_cross_section(self, filepath: str, format: str = "csv") -> None:
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

        if format == "csv":
            self.xsec_data.to_csv(filepath, index=False)
        elif format == "excel":
            self.xsec_data.to_excel(filepath, index=False)
        elif format == "shapefile":
            # Create point geometries for shapefile
            gdf = gpd.GeoDataFrame(
                self.xsec_data,
                geometry=[
                    Point(x, y)
                    for x, y in zip(
                        self.xsec_data[self.columns["x_coord"]],
                        self.xsec_data[self.columns["y_coord"]],
                    )
                ],
                crs="EPSG:4326",  # Adjust CRS as needed
            )
            gdf.to_file(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Cross-section data exported to: {filepath}")


# Utility functions for standalone use
def quick_cross_section(
    data: pd.DataFrame, endpoints: List[List[float]], search_distance: float, **kwargs
) -> Tuple[plt.Figure, pd.DataFrame]:
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
    print(
        """
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
    """
    )
