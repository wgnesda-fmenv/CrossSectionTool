"""
Example Usage of CrossSectionTool
==================================

This script demonstrates how to use the CrossSectionTool with both
lithologic interval data and point data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cross_section_tool import CrossSectionTool, quick_cross_section


def create_sample_lithologic_data():
    """
    Create sample borehole lithologic data for demonstration.
    
    Returns a DataFrame with interval data (top/bottom depths).
    """
    np.random.seed(42)
    
    # Create 10 boreholes along a transect
    n_borings = 10
    x_coords = np.linspace(100, 1000, n_borings)
    y_coords = np.linspace(200, 800, n_borings) + np.random.normal(0, 50, n_borings)
    
    lithologies = ['Sand', 'Clay', 'Gravel', 'Silt', 'Sandy Clay']
    
    data = []
    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        station_id = f'BH-{i+1:03d}'
        
        # Create 3-5 lithologic layers per boring
        n_layers = np.random.randint(3, 6)
        top_depth = 0
        
        for layer in range(n_layers):
            thickness = np.random.uniform(5, 20)
            bottom_depth = top_depth + thickness
            
            data.append({
                'StationID': station_id,
                'X_Coord': x,
                'Y_Coord': y,
                'Top_Depth': top_depth,
                'Bottom_Depth': bottom_depth,
                'Description': np.random.choice(lithologies)
            })
            
            top_depth = bottom_depth
    
    return pd.DataFrame(data)


def create_sample_point_data():
    """
    Create sample point measurement data (e.g., water levels).
    
    Returns a DataFrame with point measurements at specific depths.
    """
    np.random.seed(42)
    
    # Create measurement points
    n_points = 15
    x_coords = np.random.uniform(100, 1000, n_points)
    y_coords = np.random.uniform(200, 800, n_points)
    depths = np.random.uniform(10, 50, n_points)
    
    data = []
    for i, (x, y, depth) in enumerate(zip(x_coords, y_coords, depths)):
        data.append({
            'StationID': f'MW-{i+1:03d}',
            'X_Coord': x,
            'Y_Coord': y,
            'Depth': depth,
            'Description': 'Water Level',
            'Value': np.random.uniform(5, 15)  # Some measured value
        })
    
    return pd.DataFrame(data)


def example_1_basic_interactive():
    """
    Example 1: Basic usage with interactive line drawing.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Interactive Cross-Section")
    print("="*60)
    
    # Create sample data
    df = create_sample_lithologic_data()
    print(f"Created {len(df)} lithologic records from {df['StationID'].nunique()} boreholes")
    
    # Initialize tool
    xsec = CrossSectionTool(df)
    
    # Interactive line drawing (comment out if running non-interactively)
    # xsec.set_line_interactive()
    
    # For non-interactive demonstration, use programmatic approach
    xsec.set_line_programmatic([[100, 200], [1000, 800]])
    
    # Build cross-section
    xsec_data = xsec.build_cross_section(search_distance=100)
    
    # Define custom color scheme
    color_scheme = {
        'Sand': '#FFFF99',
        'Clay': '#CC9966',
        'Gravel': '#999999',
        'Silt': '#CCCC99',
        'Sandy Clay': '#CC9999'
    }
    
    # Plot cross-section
    fig = xsec.plot_cross_section(
        color_scheme=color_scheme,
        title='Example Cross-Section A-A\'',
        ylabel='Elevation (ft)',
        figsize=(14, 6)
    )
    
    # Plot map view
    fig_map = xsec.plot_map_view(figsize=(8, 8))
    
    return xsec, fig, fig_map


def example_2_programmatic_with_dem():
    """
    Example 2: Programmatic line definition with DEM.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Programmatic Cross-Section (No DEM)")
    print("="*60)
    
    df = create_sample_lithologic_data()
    
    # Custom column mapping (if your data uses different column names)
    column_mapping = {
        'station_id': 'StationID',
        'x_coord': 'X_Coord',
        'y_coord': 'Y_Coord',
        'top_depth': 'Top_Depth',
        'bottom_depth': 'Bottom_Depth',
        'description': 'Description'
    }
    
    xsec = CrossSectionTool(df, column_mapping=column_mapping)
    
    # Define line programmatically
    endpoints = [[150, 250], [950, 750]]
    xsec.set_line_programmatic(endpoints)
    
    # Build cross-section with reference elevation
    xsec.build_cross_section(
        search_distance=150,
        reference_elevation=100.0,  # Constant elevation since no DEM
        use_elevation=True
    )
    
    # Plot
    fig = xsec.plot_cross_section(
        title='Cross-Section B-B\' (Constant Reference Elevation = 100 ft)',
        ylabel='Elevation (ft)'
    )
    
    # Export data
    xsec_data = xsec.get_cross_section_data()
    print(f"\nCross-section contains {len(xsec_data)} records")
    print("\nFirst few records:")
    print(xsec_data[['StationID', 'dist_along_line', 'top_elevation', 
                     'bottom_elevation', 'Description']].head())
    
    return xsec, fig


def example_3_point_data():
    """
    Example 3: Cross-section with point data (not intervals).
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Point Data Cross-Section")
    print("="*60)
    
    df = create_sample_point_data()
    print(f"Created {len(df)} point measurements")
    
    # Specify column mapping for point data
    column_mapping = {
        'station_id': 'StationID',
        'x_coord': 'X_Coord',
        'y_coord': 'Y_Coord',
        'depth': 'Depth',
        'description': 'Description',
        'value': 'Value'
    }
    
    xsec = CrossSectionTool(df, column_mapping=column_mapping)
    xsec.set_line_programmatic([[100, 200], [1000, 800]])
    
    xsec.build_cross_section(
        search_distance=200,
        reference_elevation=50.0
    )
    
    # Plot with single color for all points
    color_scheme = {'Water Level': 'blue'}
    
    fig = xsec.plot_cross_section(
        color_scheme=color_scheme,
        title='Water Level Cross-Section',
        ylabel='Elevation (ft)',
        bar_width=20
    )
    
    return xsec, fig


def example_4_quick_function():
    """
    Example 4: Using the quick_cross_section convenience function.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Quick Cross-Section Function")
    print("="*60)
    
    df = create_sample_lithologic_data()
    
    # One-liner to create cross-section
    fig, xsec_data = quick_cross_section(
        data=df,
        endpoints=[[200, 300], [900, 700]],
        search_distance=100,
        title='Quick Cross-Section',
        figsize=(12, 5)
    )
    
    print(f"Created cross-section with {len(xsec_data)} records")
    
    return fig, xsec_data


def example_5_export_data():
    """
    Example 5: Exporting cross-section data.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Exporting Cross-Section Data")
    print("="*60)
    
    df = create_sample_lithologic_data()
    
    xsec = CrossSectionTool(df)
    xsec.set_line_programmatic([[100, 200], [1000, 800]])
    xsec.build_cross_section(search_distance=100)
    
    # Export to different formats
    try:
        xsec.export_cross_section('cross_section_data.csv', format='csv')
        print("✓ Exported to CSV")
    except Exception as e:
        print(f"✗ CSV export failed: {e}")
    
    try:
        xsec.export_cross_section('cross_section_data.xlsx', format='excel')
        print("✓ Exported to Excel")
    except Exception as e:
        print(f"✗ Excel export failed: {e}")
    
    # Get data for further analysis
    xsec_data = xsec.get_cross_section_data()
    
    print(f"\nData columns: {list(xsec_data.columns)}")
    print(f"Data shape: {xsec_data.shape}")
    
    return xsec


def example_6_multiple_sections():
    """
    Example 6: Creating multiple cross-sections from the same dataset.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Multiple Cross-Sections")
    print("="*60)
    
    df = create_sample_lithologic_data()
    
    # Define multiple section lines
    sections = {
        'Section 1': [[100, 200], [1000, 300]],
        'Section 2': [[200, 250], [900, 750]],
        'Section 3': [[150, 400], [950, 600]]
    }
    
    color_scheme = {
        'Sand': '#FFFF99',
        'Clay': '#CC9966',
        'Gravel': '#999999',
        'Silt': '#CCCC99',
        'Sandy Clay': '#CC9999'
    }
    
    figures = {}
    
    for section_name, endpoints in sections.items():
        print(f"\nCreating {section_name}...")
        
        xsec = CrossSectionTool(df)
        xsec.set_line_programmatic(endpoints)
        xsec.build_cross_section(search_distance=120)
        
        fig = xsec.plot_cross_section(
            color_scheme=color_scheme,
            title=section_name,
            figsize=(12, 5)
        )
        
        figures[section_name] = fig
    
    return figures


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" CrossSectionTool - Example Usage Demonstrations")
    print("="*70)
    
    # Run examples
    print("\nRunning examples... (close plot windows to continue)")
    
    # Example 1: Basic interactive (or programmatic)
    xsec1, fig1, fig_map1 = example_1_basic_interactive()
    plt.show()
    
    # Example 2: Programmatic with constant elevation
    xsec2, fig2 = example_2_programmatic_with_dem()
    plt.show()
    
    # Example 3: Point data
    xsec3, fig3 = example_3_point_data()
    plt.show()
    
    # Example 4: Quick function
    fig4, data4 = example_4_quick_function()
    plt.show()
    
    # Example 5: Export
    xsec5 = example_5_export_data()
    
    # Example 6: Multiple sections
    figs6 = example_6_multiple_sections()
    plt.show()
    
    print("\n" + "="*70)
    print(" All examples completed!")
    print("="*70)
    print("\nKey takeaways:")
    print("  1. Use set_line_interactive() for manual line drawing")
    print("  2. Use set_line_programmatic() for automated workflows")
    print("  3. Customize column names with column_mapping parameter")
    print("  4. Export data to CSV, Excel, or Shapefile")
    print("  5. Use quick_cross_section() for simple one-liners")
    print("\nFor more information, see the docstrings in CrossSectionTool class")
