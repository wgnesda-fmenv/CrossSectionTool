"""
Example: Using CrossSectionTool with Dated Bathymetry Rasters
==============================================================

This example demonstrates:
1. Loading data as a GeoDataFrame or DataFrame
2. Using a folder of dated bathymetry rasters
3. Matching cores to nearest-year bathymetry
4. Plotting multiple ground surface profiles
"""

import pandas as pd
import geopandas as gpd
from cross_section_tool import CrossSectionTool
import matplotlib.pyplot as plt

# =============================================================================
# Example 1: Using DataFrame with dated bathymetry
# =============================================================================


def example_with_dataframe():
    """Example using regular DataFrame with sample dates and bathymetry folder."""

    # Load your data (CSV with coordinates and sample dates)
    # Expected columns: StationID, X_Coord, Y_Coord, Top_Depth, Bottom_Depth,
    #                  Description, SampleDate
    df = pd.read_csv("cross_section_data.csv")

    # Initialize tool with bathymetry folder
    # The folder should contain .tif files with years in filenames
    # e.g., 'bathy_2015.tif', 'bathymetry_2018.tif', '2020.tif'
    xsec = CrossSectionTool(
        data=df,
        bathy_folder="path/to/bathymetry_rasters/",  # Replace with your folder path
    )

    # Define cross-section line
    endpoints = [[100, 200], [500, 600]]  # Replace with your coordinates
    xsec.set_line_programmatic(endpoints)

    # Build cross-section
    # Each station will use bathymetry from the year closest to its sample date
    xsec.build_cross_section(
        search_distance=100,
        sample_num=100,  # Number of points to sample along line for ground surface
    )

    # Plot with single ground surface (most recent)
    fig1 = xsec.plot_cross_section(
        plot_ground_surface=True,
        plot_all_ground_surfaces=False,
        title="Cross-Section with Most Recent Bathymetry",
    )

    # Plot with all dated ground surfaces
    fig2 = xsec.plot_cross_section(
        plot_ground_surface=True,
        plot_all_ground_surfaces=True,  # Show all available bathymetry years
        title="Cross-Section with All Dated Bathymetry Surfaces",
    )

    plt.show()

    # Export results
    xsec.export_cross_section("cross_section_output.csv")

    return xsec


# =============================================================================
# Example 2: Using GeoDataFrame
# =============================================================================


def example_with_geodataframe():
    """Example using GeoDataFrame (e.g., from shapefile)."""

    # Load GeoDataFrame from shapefile, geopackage, etc.
    gdf = gpd.read_file("cores.shp")

    # The tool will automatically extract X, Y from geometry if needed
    # Make sure your GeoDataFrame has these columns:
    # - StationID (or specify in column_mapping)
    # - Top_Depth, Bottom_Depth (for interval data)
    # - Description (lithology)
    # - SampleDate (for bathymetry matching)

    # Initialize with custom column mapping if needed
    column_mapping = {
        "station_id": "CoreID",  # If your ID column is named 'CoreID'
        "sample_date": "Date",  # If your date column is named 'Date'
        "description": "Lithology",  # If your lithology column is named 'Lithology'
    }

    xsec = CrossSectionTool(
        data=gdf,
        column_mapping=column_mapping,
        bathy_folder="path/to/bathymetry_rasters/",
    )

    # Interactive line drawing
    endpoints = xsec.set_line_interactive()

    # Build and plot
    xsec.build_cross_section(search_distance=50, sample_num=100)
    fig = xsec.plot_cross_section(plot_all_ground_surfaces=True)
    plt.show()

    return xsec


# =============================================================================
# Example 3: Without dated bathymetry (single DEM)
# =============================================================================


def example_single_dem():
    """Example using single DEM file (backward compatible)."""

    df = pd.read_csv("cross_section_data.csv")

    # No bathy_folder specified - uses traditional single DEM approach
    xsec = CrossSectionTool(data=df)

    xsec.set_line_programmatic([[100, 200], [500, 600]])

    # Provide single DEM path
    xsec.build_cross_section(
        search_distance=100,
        dem_path="single_dem.tif",  # Single DEM file
        sample_num=100,
    )

    fig = xsec.plot_cross_section()
    plt.show()

    return xsec


# =============================================================================
# Example 4: Checking which bathymetry years were used
# =============================================================================


def example_check_bathy_usage():
    """Check which bathymetry years were matched to which cores."""

    df = pd.read_csv("cross_section_data.csv")
    xsec = CrossSectionTool(data=df, bathy_folder="path/to/bathymetry_rasters/")

    xsec.set_line_programmatic([[100, 200], [500, 600]])
    xsec.build_cross_section(search_distance=100, sample_num=100)

    # Get cross-section data
    xsec_data = xsec.get_cross_section_data()

    # Check which bathymetry year was used for each station
    if "bathy_year" in xsec_data.columns:
        summary = (
            xsec_data.groupby("StationID")
            .agg({"SampleDate": "first", "bathy_year": "first"})
            .reset_index()
        )

        print("\nBathymetry Year Matching:")
        print(summary)

        # See distribution of bathymetry years used
        print("\nBathymetry Year Distribution:")
        print(xsec_data["bathy_year"].value_counts().sort_index())

    return xsec


# =============================================================================
# Run examples
# =============================================================================

if __name__ == "__main__":
    print("CrossSectionTool - Dated Bathymetry Examples")
    print("=" * 60)

    # Uncomment the example you want to run:

    # xsec = example_with_dataframe()
    # xsec = example_with_geodataframe()
    # xsec = example_single_dem()
    # xsec = example_check_bathy_usage()

    print("\nExample usage guide:")
    print("1. Prepare bathymetry folder with .tif files containing year in filename")
    print("   e.g., 'bathy_2015.tif', 'bathymetry_2018.tif', 'dem_2020.tif'")
    print("2. Ensure your data has SampleDate column (or specify in column_mapping)")
    print("3. Tool will automatically match each core to nearest-year bathymetry")
    print("4. Use plot_all_ground_surfaces=True to visualize all dated surfaces")
