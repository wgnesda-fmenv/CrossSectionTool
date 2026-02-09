# Cross-Section Tool for Geological Data

A comprehensive Python tool for creating geological cross-sections from borehole lithologic data or point measurements. Supports both interactive line drawing and programmatic workflows.

## Features

- ✅ **Interactive & Programmatic**: Draw cross-section lines interactively or define them programmatically
- ✅ **Flexible Data Input**: Works with lithologic interval data or point measurements
- ✅ **DEM Integration**: Sample ground surface elevations from DEM rasters
- ✅ **Customizable Visualization**: Full control over colors, widths, and styling
- ✅ **Data Export**: Export cross-section data to CSV, Excel, or Shapefiles
- ✅ **Map View**: Generate accompanying map views showing section location
- ✅ **Well-Documented**: Comprehensive docstrings and examples

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

**Example CSV:**
```csv
StationID,X_Coord,Y_Coord,Top_Depth,Bottom_Depth,Description
BH-001,100,200,0,5,Sand
BH-001,100,200,5,15,Clay
BH-002,150,250,0,10,Gravel
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
| Value | Optional numeric value | 15.2 |

## API Reference

### CrossSectionTool Class

#### Initialization

```python
CrossSectionTool(data, column_mapping=None)
```

**Parameters:**
- `data` (pd.DataFrame): Input dataset
- `column_mapping` (dict, optional): Map standard names to your column names

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
xsec.build_cross_section(search_distance, dem_path=None, 
                         reference_elevation=0.0, use_elevation=True)
```
Project data onto cross-section line.

**Parameters:**
- `search_distance` (float): Maximum perpendicular distance from line
- `dem_path` (str, optional): Path to DEM raster
- `reference_elevation` (float): Constant elevation if no DEM
- `use_elevation` (bool): Convert depths to elevations

**Returns:** pd.DataFrame with cross-section data

##### plot_cross_section()
```python
fig = xsec.plot_cross_section(color_scheme=None, figsize=(14, 6),
                               vertical_exaggeration=1.0, bar_width=None,
                               plot_ground_surface=True, ylabel='Elevation',
                               title=None, savepath=None, dpi=300)
```
Generate cross-section plot.

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
https://github.com/your-repo/cross-section-tool
```

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

## Acknowledgments

Based on original xsec_tool.py with enhancements for:
- Better code organization
- Comprehensive documentation
- Flexible data handling
- Modern Python practices
