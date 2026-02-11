# Cross-Section Tool for Geological Data

A comprehensive Python tool for creating geological cross-sections from borehole lithologic data or point measurements. Supports both interactive line drawing and programmatic workflows.

## Features

- ✅ **Interactive & Programmatic**: Draw cross-section lines interactively or define them programmatically
- ✅ **Flexible Data Input**: Works with lithologic interval data or point measurements
- ✅ **GeoDataFrame Support**: Accepts both pandas DataFrames and geopandas GeoDataFrames
- ✅ **Dated Bathymetry**: Match cores to nearest-year bathymetry from a folder of dated rasters
- ✅ **DEM Integration**: Sample ground surface elevations from single or multiple dated DEM rasters
- ✅ **Multiple Ground Surfaces**: Plot multiple dated bathymetry surfaces on same cross-section
- ✅ **Value-Based Coloring**: Color intervals by numeric values with customizable thresholds
- ✅ **Customizable Visualization**: Full control over colors, widths, and styling
- ✅ **Data Export**: Export cross-section data to CSV, Excel, or Shapefiles
- ✅ **Map View**: Generate accompanying map views showing section location

## To dos
 - add data validation step to ensure no data values are not present or adjusted
 - change default bar width to 5% of cross-section length
 - add ability to switch between x and y distances
 - add elevation column plotting - bypass DEM ground surface and plot directly from the elevation column
 - read or specify units
 - annotations (location ids, other?)

## Installation

### Requirements

```bash
pip install numpy pandas matplotlib shapely geopandas
```

Optional (for DEM support):
```bash
pip install rasterio
```

### Quick Start

1. Download `cross_section_tool.py`
2. Place it in your project directory or Python path
3. Import and use:

```python
from cross_section_tool import CrossSectionTool
```

## Usage

### Basic Example

```python
import pandas as pd
from cross_section_tool import CrossSectionTool

# Load your data
df = pd.read_csv('borehole_data.csv')

# Create tool instance
xsec = CrossSectionTool(df)

# Define cross-section line programmatically
xsec.set_line_programmatic([[100, 200], [500, 600]])

# Build cross-section
xsec.build_cross_section(search_distance=50)

# Plot
fig = xsec.plot_cross_section()
plt.show()

# Get data
xsec_data = xsec.get_cross_section_data()
```

### Interactive Line Drawing

```python
# Let user draw the line on a map
xsec.set_line_interactive()
```

### With Custom Column Names

```python
column_mapping = {
    'station_id': 'Well_ID',
    'x_coord': 'Easting',
    'y_coord': 'Northing',
    'top_depth': 'Top_Ft',
    'bottom_depth': 'Bottom_Ft',
    'description': 'Lithology'
}

xsec = CrossSectionTool(df, column_mapping=column_mapping)
```

### With DEM for Ground Surface

```python
xsec.build_cross_section(
    search_distance=100,
    dem_path='elevation.tif',
    use_elevation=True
)
```

### With Constant Reference Elevation

```python
xsec.build_cross_section(
    search_distance=100,
    reference_elevation=500.0,  # All points at 500 ft elevation
    use_elevation=True
)
```

### With Dated Bathymetry Rasters (NEW!)

Match cores to bathymetry from the nearest year based on sample dates:

```python
# Your data must have a SampleDate column
df = pd.read_csv('cores.csv')  # Must contain SampleDate column

# Initialize with bathymetry folder containing dated .tif files
# Files should have year in filename: 'bathy_2015.tif', 'bathymetry_2018.tif', '2020.tif'
xsec = CrossSectionTool(
    data=df,
    bathy_folder='path/to/bathymetry_rasters/'
)

xsec.set_line_programmatic([[100, 200], [500, 600]])

# Build - each core uses bathymetry from nearest year to its sample date
xsec.build_cross_section(
    search_distance=100,
    sample_num=100  # Points to sample along line for ground surface
)

# Plot with single surface (most recent year)
fig1 = xsec.plot_cross_section(plot_all_ground_surfaces=False)

# Plot with all dated bathymetry surfaces
fig2 = xsec.plot_cross_section(plot_all_ground_surfaces=True)
```

### Using GeoDataFrame Input (NEW!)

```python
import geopandas as gpd

# Load from shapefile, geopackage, etc.
gdf = gpd.read_file('cores.shp')

# Tool automatically extracts X, Y from geometry if needed
xsec = CrossSectionTool(
    data=gdf,
    bathy_folder='bathymetry_rasters/'  # Optional
)

# Works the same as DataFrame input
xsec.set_line_interactive()
xsec.build_cross_section(search_distance=50, sample_num=100)
fig = xsec.plot_cross_section()
```

### Custom Colors

```python
color_scheme = {
    'Sand': '#FFFF99',
    'Clay': '#CC9966',
    'Gravel': '#999999',
    'Silt': '#CCCC99'
}

fig = xsec.plot_cross_section(color_scheme=color_scheme)
```

### Value-Based Coloring (NEW!)

Color intervals by numeric values instead of descriptions. Perfect for contamination levels, concentrations, or any numeric data.

#### Simple Binary Threshold

```python
# Color by a single threshold: values > 1000 = red, values <= 1000 = green
fig = xsec.plot_cross_section(
    color_by='value',
    value_threshold=1000
)
```

#### Multiple Thresholds (Range-Based Coloring)

```python
# Define multiple ranges with custom colors
# Ranges: (-∞, 500], (500, 1000], (1000, 2000], (2000, +∞)
fig = xsec.plot_cross_section(
    color_by='value',
    value_bins=[500, 1000, 2000],
    value_colors=['blue', 'green', 'orange', 'red']
)
```

#### Contamination Example

```python
# Classify contamination levels
fig = xsec.plot_cross_section(
    color_by='value',
    value_bins=[10, 50, 100],  # Clean, Low, Medium, High
    value_colors=['green', 'yellow', 'orange', 'red'],
    title='Contamination Levels (mg/kg)'
)
```

**Important Notes:**
- Your data must have a `Value` column (or specify in `column_mapping`)
- `value_colors` length must equal `len(value_bins) + 1`
- Missing/NaN values default to light gray
- Works with both interval and point data

### Quick One-Liner

```python
from cross_section_tool import quick_cross_section

fig, data = quick_cross_section(
    df, 
    endpoints=[[100, 200], [500, 600]],
    search_distance=50
)
```

## Data Format Requirements

### Lithologic Interval Data

Your DataFrame should contain these columns (customizable via `column_mapping`):

| Column | Description | Example |
|--------|-------------|---------|
| StationID | Unique identifier for each boring | BH-001 |
| X_Coord | X or Easting coordinate | 1234.56 |
| Y_Coord | Y or Northing coordinate | 5678.90 |
| Top_Depth | Top depth of interval | 0.0 |
| Bottom_Depth | Bottom depth of interval | 10.5 |
| Description | Lithologic description | Sand |
| SampleDate | Sample date (for bathymetry matching) | 2015-06-20 |
| Value | Numeric value (for value-based coloring) | 850.5 |

**Notes:** 
- `SampleDate` is optional - required only when using dated bathymetry rasters
- `Value` is optional - required only when using value-based coloring (`color_by='value'`)

**Example CSV:**
```csv
StationID,X_Coord,Y_Coord,Top_Depth,Bottom_Depth,Description,SampleDate,Value
BH-001,100,200,0,5,Sand,2015-06-20,450
BH-001,100,200,5,15,Clay,2015-06-20,1200
BH-002,150,250,0,10,Gravel,2018-08-15,75
```

### Point Data

For point measurements (e.g., water levels):

| Column | Description | Example |
|--------|-------------|---------|
| StationID | Unique identifier | MW-001 |
| X_Coord | X or Easting coordinate | 1234.56 |
| Y_Coord | Y or Northing coordinate | 5678.90 |
| Depth | Depth of measurement | 25.3 |
| Description | Description | Water Level |
| Value | Numeric value (for value-based coloring or data) | 15.2 |
| SampleDate | Sample date (optional, for bathymetry) | 2020-03-15 |

### GeoDataFrame Input

When using a GeoDataFrame:
- The tool automatically extracts X_Coord and Y_Coord from the geometry column
- All other columns remain the same as above
- Geometry can be Point or any type with coordinates

## Dated Bathymetry Features

### Overview

The tool can automatically match cores/samples to bathymetry rasters from different years based on sample dates. This is useful when you have:
- Historical core data spanning multiple years
- Bathymetry/DEM data collected at different times
- Need to compare samples against the ground surface from their collection year

### Bathymetry Folder Setup

1. **Create a folder** containing your bathymetry .tif rasters
2. **Name files with year** - the tool looks for 4-digit years in filenames:
   - ✅ Good: `bathy_2015.tif`, `bathymetry_2018.tif`, `2020.tif`, `dem_2022_final.tif`
   - ❌ Bad: `bathymetry.tif`, `bathy_old.tif`, `surface_15.tif`

3. **Example folder structure:**
   ```
   bathymetry_rasters/
   ├── bathy_2010.tif
   ├── bathy_2015.tif
   ├── bathy_2018.tif
   └── bathy_2022.tif
   ```

### How Matching Works

1. Tool scans folder and creates year → file mapping
2. For each core/sample, extracts year from SampleDate column
3. Finds closest available bathymetry year
4. Uses that raster for ground elevation

**Example:**
- Core sampled on `2016-07-15`
- Available bathymetry years: 2010, 2015, 2018, 2022
- Tool uses 2015 bathymetry (closest year)

### Plotting Multiple Surfaces

You can visualize all dated bathymetry surfaces on one cross-section:

```python
# Plot all available ground surfaces
fig = xsec.plot_cross_section(
    plot_ground_surface=True,
    plot_all_ground_surfaces=True  # Shows all years
)
```

This creates a plot with multiple colored lines, each representing bathymetry from a different year.

### Checking Which Bathymetry Was Used

```python
# After building cross-section
xsec_data = xsec.get_cross_section_data()

# Check which bathymetry year was assigned to each station
if 'bathy_year' in xsec_data.columns:
    summary = xsec_data.groupby('StationID')['bathy_year'].first()
    print(summary)
```

## API Reference

### CrossSectionTool Class

#### Initialization

```python
CrossSectionTool(data, column_mapping=None, bathy_folder=None)
```

**Parameters:**
- `data` (pd.DataFrame or gpd.GeoDataFrame): Input dataset
- `column_mapping` (dict, optional): Map standard names to your column names
- `bathy_folder` (str, optional): Path to folder containing dated bathymetry .tif files

#### Methods

##### set_line_interactive()
```python
xsec.set_line_interactive(figsize=(10, 8), background_data=None)
```
Interactively draw cross-section line by clicking two points on a map.

##### set_line_programmatic()
```python
xsec.set_line_programmatic(endpoints)
```
Define cross-section line with coordinates.
- `endpoints`: `[[x1, y1], [x2, y2]]`

##### build_cross_section()
```python
xsec.build_cross_section(search_distance, dem_path=None, sample_num=100,
                         reference_elevation=0.0, use_elevation=True)
```
Project data onto cross-section line.

**Parameters:**
- `search_distance` (float): Maximum perpendicular distance from line
- `dem_path` (str, optional): Path to single DEM raster (ignored if bathy_folder was provided)
- `sample_num` (int, optional): Number of points to sample along line for ground surface profile (default: 100)
- `reference_elevation` (float): Constant elevation if no DEM
- `use_elevation` (bool): Convert depths to elevations

**Returns:** pd.DataFrame with cross-section data

**Note:** If `bathy_folder` was provided during initialization, the tool uses dated bathymetry matching instead of a single DEM.

##### plot_cross_section()
```python
fig = xsec.plot_cross_section(color_scheme=None, figsize=(14, 6),
                               vertical_exaggeration=1.0, bar_width=None,
                               plot_ground_surface=True, plot_all_ground_surfaces=False,
                               ylabel='Elevation', title=None, savepath=None, dpi=300,
                               color_by=None, value_threshold=None, 
                               value_bins=None, value_colors=None)
```
Generate cross-section plot.

**Parameters:**
- `color_scheme` (dict, optional): Mapping of descriptions to colors (ignored if color_by='value')
- `figsize` (tuple): Figure size (width, height)
- `vertical_exaggeration` (float): Vertical exaggeration factor
- `bar_width` (float, optional): Width of bars (auto-calculated if None)
- `plot_ground_surface` (bool): Whether to plot ground surface line
- `plot_all_ground_surfaces` (bool): If True, plot all dated bathymetry surfaces (only when using bathy_folder)
- `ylabel` (str): Y-axis label
- `title` (str, optional): Plot title
- `savepath` (str, optional): Path to save figure
- `dpi` (int): Resolution for saved figure
- `color_by` (str, optional): 'description' (default) or 'value' - coloring mode
- `value_threshold` (float, optional): Binary threshold for value coloring (default: 1000)
- `value_bins` (list of float, optional): Thresholds for multi-range value coloring
- `value_colors` (list of str, optional): Colors for each range (length = len(value_bins) + 1)

##### plot_map_view()
```python
fig = xsec.plot_map_view(figsize=(8, 8), show_all_data=True,
                         background_data=None)
```
Generate map view showing section location.

##### get_cross_section_data()
```python
xsec_data = xsec.get_cross_section_data()
```
Return cross-section data as DataFrame.

##### export_cross_section()
```python
xsec.export_cross_section(filepath, format='csv')
```
Export data to file.
- `format`: 'csv', 'excel', or 'shapefile'

## Examples

See `example_usage.py` for comprehensive examples including:

1. **Example 1**: Basic interactive cross-section
2. **Example 2**: Programmatic with constant elevation
3. **Example 3**: Point data (water levels)
4. **Example 4**: Quick one-liner function
5. **Example 5**: Data export
6. **Example 6**: Multiple sections from same dataset

Run examples:
```bash
python example_usage.py
```

## Advanced Usage

### Multiple Cross-Sections

```python
sections = {
    'A-A': [[100, 200], [500, 600]],
    'B-B': [[150, 250], [450, 550]]
}

for name, endpoints in sections.items():
    xsec = CrossSectionTool(df)
    xsec.set_line_programmatic(endpoints)
    xsec.build_cross_section(search_distance=50)
    fig = xsec.plot_cross_section(title=f'Section {name}')
    plt.savefig(f'section_{name}.png', dpi=300)
```

### Custom Styling

```python
fig = xsec.plot_cross_section(
    figsize=(16, 8),
    vertical_exaggeration=2.0,
    bar_width=30,
    ylabel='Elevation (m MSL)',
    title='Cross-Section Through Aquifer',
    savepath='my_xsec.png',
    dpi=600
)
```

### Working with GIS Data

```python
import geopandas as gpd

# Load GIS data
gdf = gpd.read_file('boreholes.shp')

# Convert to DataFrame
df = pd.DataFrame(gdf.drop(columns='geometry'))
df['X_Coord'] = gdf.geometry.x
df['Y_Coord'] = gdf.geometry.y

# Create cross-section
xsec = CrossSectionTool(df)
# ... continue as normal
```

## Output Data Structure

The `get_cross_section_data()` method returns a DataFrame with these added columns:

- `dist_to_line`: Perpendicular distance to section line
- `proj_x`, `proj_y`: Projected coordinates on line
- `dist_along_line`: Distance along section line (0 at start)
- `ground_elevation`: Ground surface elevation
- `top_elevation`, `bottom_elevation`: Layer elevations (interval data)
- `elevation`: Point elevation (point data)

## Troubleshooting

### No points found within search distance
- Increase `search_distance` parameter
- Check coordinate system consistency
- Verify cross-section line intersects your data

### DEM not working
- Ensure `rasterio` is installed: `pip install rasterio`
- Check DEM file path is correct
- Verify DEM and data use same coordinate system
- Use `reference_elevation` as fallback

### Colors not showing correctly
- Provide explicit `color_scheme` dictionary
- Check lithology names match exactly (case-sensitive)

### Interactive line drawing not working
- Ensure matplotlib backend supports interaction
- Try: `matplotlib.use('TkAgg')` before importing pyplot
- Use programmatic approach if running in notebook/script

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - feel free to use and modify for your projects.

## Citation

If you use this tool in research, please cite:
```
CrossSectionTool - Geological Cross-Section Visualization
https://github.com/wgnesda-fmenv/CrossSectionTool
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

## Acknowledgments

Based on original xsec_tool.py with enhancements for:
- Better code organization
- Comprehensive documentation
- Flexible data handling
- Modern Python practices
