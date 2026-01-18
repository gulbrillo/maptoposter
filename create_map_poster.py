import inspect
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
import argparse
from math import cos, radians
from shapely.geometry import box as shapely_box
import pandas as pd
import requests

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

# Poster size (inches) used for output. This defines the intended aspect ratio.
POSTER_FIGSIZE = (12, 16)  # width, height
POSTER_ASPECT = POSTER_FIGSIZE[1] / POSTER_FIGSIZE[0]  # height/width = 16/12 = 4/3

# Earth approx: meters per degree latitude (varies slightly, but this is fine for city-scale posters)
METERS_PER_DEG_LAT = 111_320.0


def load_fonts():
    """
    Load Roboto fonts from the fonts directory.
    Returns dict with font paths for different weights.
    """
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }

    for weight, path in fonts.items():
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None

    return fonts


FONTS = load_fonts()


def generate_output_filename(city, country, theme_name, display_name=None, display_country=None):
    """
    Generate unique output filename with city/country (display overrides), theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    name_source = display_name if display_name and display_name.strip() else city
    country_source = display_country if display_country and display_country.strip() else country

    city_slug = name_source.lower().strip().replace(' ', '_')
    country_slug = country_source.lower().strip().replace(' ', '_')
    filename = f"{city_slug}_{country_slug}_{theme_name}_{timestamp}.png"
    return os.path.join(POSTERS_DIR, filename)


def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []

    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith('.json'):
            themes.append(file[:-5])  # Remove .json extension
    return themes


def load_theme(theme_name="feature_based"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")

    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "railway": "#5A5A5A",
            "road_default": "#3A3A3A"
        }

    with open(theme_file, 'r') as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if 'description' in theme:
            print(f"  {theme['description']}")
        return theme


THEME = None  # loaded later


def create_gradient_fade(ax, color, location='bottom', zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))

    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]

    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end

    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect='auto',
        cmap=custom_cmap,
        zorder=zorder,
        origin='lower'
    )


def get_edge_colors_by_type(G):
    """
    Assign colors to edges based on road type hierarchy.
    """
    edge_colors = []
    for _, _, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'

        if highway in ['motorway', 'motorway_link']:
            color = THEME['road_motorway']
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            color = THEME['road_primary']
        elif highway in ['secondary', 'secondary_link']:
            color = THEME['road_secondary']
        elif highway in ['tertiary', 'tertiary_link']:
            color = THEME['road_tertiary']
        elif highway in ['residential', 'living_street', 'unclassified']:
            color = THEME['road_residential']
        else:
            color = THEME['road_default']

        edge_colors.append(color)

    return edge_colors


def get_edge_widths_by_type(G):
    """
    Assign line widths to edges based on road type.
    """
    edge_widths = []
    for _, _, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'

        if highway in ['motorway', 'motorway_link']:
            width = 1.2
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            width = 1.0
        elif highway in ['secondary', 'secondary_link']:
            width = 0.8
        elif highway in ['tertiary', 'tertiary_link']:
            width = 0.6
        else:
            width = 0.4

        edge_widths.append(width)

    return edge_widths


def get_coordinates(city, country):
    """
    Fetch coordinates for a given city and country using geopy (Nominatim).
    Includes rate limiting to be respectful to the geocoding service.
    """
    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster")
    time.sleep(1)
    location = geolocator.geocode(f"{city}, {country}")

    if location:
        print(f"✓ Found: {location.address}")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        return (location.latitude, location.longitude)

    raise ValueError(f"Could not find coordinates for {city}, {country}")


def compute_bbox_from_center(point, half_width_m, half_height_m):
    """
    Compute an irregular bounding box centered at (lat, lon), with independent half-width/half-height in meters.

    Returns: (north, south, east, west)
    """
    lat, lon = point

    dlat = half_height_m / METERS_PER_DEG_LAT
    meters_per_deg_lon = METERS_PER_DEG_LAT * max(cos(radians(lat)), 1e-6)
    dlon = half_width_m / meters_per_deg_lon

    north = lat + dlat
    south = lat - dlat
    east = lon + dlon
    west = lon - dlon

    return (north, south, east, west)


def bbox_to_poly_lonlat(north, south, east, west):
    """
    Build a shapely polygon from bbox in lon/lat.
    """
    return shapely_box(west, south, east, north)


def polygon_area_m2(poly_lonlat):
    """
    Return projected area in m^2 for a lon/lat polygon using OSMnx projection helper.
    """
    try:
        poly_proj, _ = ox.projection.project_geometry(poly_lonlat)
        return float(poly_proj.area)
    except Exception:
        return None


def debug_bbox_step(debug, label, north, south, east, west):
    """
    Print debug info for a bbox and its projected area.
    """
    if not debug:
        return

    poly = bbox_to_poly_lonlat(north, south, east, west)
    west_b, south_b, east_b, north_b = poly.bounds

    center_lat = (north + south) / 2.0
    meters_per_deg_lon = METERS_PER_DEG_LAT * max(cos(radians(center_lat)), 1e-6)
    width_m_approx = (east - west) * meters_per_deg_lon
    height_m_approx = (north - south) * METERS_PER_DEG_LAT

    area_m2 = polygon_area_m2(poly)
    print("\n[DEBUG]", label)
    print(f"  bbox (north, south, east, west): {north:.8f}, {south:.8f}, {east:.8f}, {west:.8f}")
    print(f"  poly.bounds (west, south, east, north): {west_b:.8f}, {south_b:.8f}, {east_b:.8f}, {north_b:.8f}")
    print(f"  approx width/height (m): {width_m_approx:,.1f} x {height_m_approx:,.1f}")
    if area_m2 is not None:
        print(f"  projected area (m^2): {area_m2:,.0f}  (~{area_m2/1e6:,.2f} km^2)")
    else:
        print("  projected area (m^2): <unavailable>")
    print(f"  ox.settings.max_query_area_size: {ox.settings.max_query_area_size!r}")
    if area_m2 is not None and ox.settings.max_query_area_size:
        try:
            ratio = area_m2 / float(ox.settings.max_query_area_size)
            print(f"  area / max_query_area_size: {ratio:,.1f}x")
        except Exception:
            pass


def debug_graph_bounds(debug, label, G):
    """
    Print bounds of a graph after it's downloaded (based on node coordinates).
    """
    if not debug or G is None:
        return
    try:
        nodes = ox.utils_graph.graph_to_gdfs(G, nodes=True, edges=False, node_geometry=True, fill_edge_geometry=False)
        if nodes is None or nodes.empty:
            return
        west, south, east, north = nodes.total_bounds
        print("\n[DEBUG]", label)
        print(f"  graph node bounds (west, south, east, north): {west:.8f}, {south:.8f}, {east:.8f}, {north:.8f}")
    except Exception as e:
        print("\n[DEBUG]", label)
        print("  <could not compute graph bounds>", repr(e))


def graph_from_polygon_compat(poly_lonlat, network_type='all'):
    """
    Compatibility wrapper for graph_from_polygon.
    """
    return ox.graph_from_polygon(poly_lonlat, network_type=network_type, truncate_by_edge=True, simplify=True)


def graph_from_point_network(point, dist_network_m, network_type='all'):
    """
    Fetch road graph using dist_type='network' (roads only).
    """
    return ox.graph_from_point(point, dist=dist_network_m, dist_type='network', network_type=network_type)


def features_from_polygon_compat(poly_lonlat, tags):
    """
    Compatibility wrapper for polygon features across OSMnx versions.
    Newer: features_from_polygon
    Older: geometries_from_polygon
    """
    try:
        return ox.features_from_polygon(poly_lonlat, tags=tags)
    except AttributeError:
        return ox.geometries_from_polygon(poly_lonlat, tags=tags)


def _ensure_polygon_list(geom):
    if geom is None:
        return []
    if hasattr(geom, "geoms"):
        return list(geom.geoms)
    return [geom]


def _subdivide_polygon_for_progress(poly_lonlat):
    try:
        subdivided = ox.utils_geo._consolidate_subdivide_geometry(poly_lonlat)
    except Exception:
        subdivided = poly_lonlat
    return _ensure_polygon_list(subdivided)


def _polygon_to_overpass_poly(poly):
    if poly is None or not hasattr(poly, "exterior"):
        return ""
    coords = list(poly.exterior.coords)
    return " ".join([f"{lat:.6f} {lon:.6f}" for lon, lat in coords])


def _build_overpass_filters(tags):
    filters = []
    for key, value in tags.items():
        if value is True:
            filters.append(f'["{key}"]')
        elif isinstance(value, (list, set, tuple)):
            for item in value:
                filters.append(f'["{key}"="{item}"]')
        else:
            filters.append(f'["{key}"="{value}"]')
    return filters


def _overpass_count(poly_lonlat, tags, timeout=180):
    polygons = _ensure_polygon_list(poly_lonlat)
    if not polygons:
        return None

    filters = _build_overpass_filters(tags)
    if not filters:
        return None

    endpoint = getattr(getattr(ox, "settings", None), "overpass_url", None)
    if not endpoint:
        endpoint = "https://overpass-api.de/api/interpreter"

    total = 0
    for polygon in polygons:
        poly_str = _polygon_to_overpass_poly(polygon)
        if not poly_str:
            return None
        query_lines = "\n".join([f'  nwr{filter_clause}(poly:"{poly_str}");' for filter_clause in filters])
        query = f"[out:json][timeout:{timeout}];(\n{query_lines}\n);out count;"

        try:
            response = requests.post(endpoint, data={"data": query}, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None

        for element in payload.get("elements", []):
            if element.get("type") == "count":
                tags_payload = element.get("tags", {})
                for key in ("nodes", "ways", "relations"):
                    value = tags_payload.get(key)
                    if value is not None:
                        total += int(value)
    return total or None


def _concat_feature_frames(frames):
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return None
    combined = pd.concat(frames)
    if combined.index.has_duplicates:
        combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def fetch_features_with_progress(poly_lonlat, tags, task_label):
    polygons = _subdivide_polygon_for_progress(poly_lonlat)
    expected_total = _overpass_count(poly_lonlat, tags)
    total_units = expected_total if expected_total else max(len(polygons), 1)
    unit = "obj" if expected_total else "req"

    frames = []
    completed = 0
    fetched_objects = 0
    fetched_requests = 0
    with tqdm(
        total=total_units,
        desc=f"{task_label} details",
        unit=unit,
        position=1,
        leave=False,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
        dynamic_ncols=True
    ) as detail_pbar:
        for polygon in polygons:
            try:
                result = features_from_polygon_compat(polygon, tags)
            except Exception:
                result = None

            count = 0
            if result is not None and not result.empty:
                frames.append(result)
                count = len(result.index.unique())

            if expected_total:
                completed += count
                if completed > detail_pbar.total:
                    detail_pbar.total = completed
                detail_pbar.update(count)
                fetched_objects += count
                fetched_requests += 1
                detail_pbar.set_postfix_str(
                    f"objects={fetched_objects:,} requests={fetched_requests}/{len(polygons)}"
                )
            else:
                fetched_objects += count
                fetched_requests += 1
                detail_pbar.update(1)
                detail_pbar.set_postfix_str(
                    f"objects={fetched_objects:,} requests={fetched_requests}/{len(polygons)}"
                )

    return _concat_feature_frames(frames)


def spaced_caps(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return "  ".join(list(text.upper()))


def fit_font_size_to_axes_width(fig, ax, text, fontprops, max_width_frac=0.90, min_size=12):
    """
    Shrinks font size until rendered text fits within a fraction of the axes width.
    Uses the actual renderer to measure text bbox in pixels.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w_px = ax.get_window_extent(renderer=renderer).width
    allowed_px = ax_w_px * max_width_frac

    size = float(fontprops.get_size_in_points())
    if size <= 0:
        size = 1.0

    test = ax.text(
        0.5, 0.5, text,
        transform=ax.transAxes,
        fontproperties=fontprops,
        ha='center', va='center',
        alpha=0.0
    )

    try:
        while True:
            fig.canvas.draw()
            bbox = test.get_window_extent(renderer=renderer)
            if bbox.width <= allowed_px or size <= min_size:
                break

            ratio = allowed_px / max(bbox.width, 1e-6)
            step = max(0.85, min(0.98, ratio))
            size = max(min_size, size * step)

            fontprops.set_size(size)
            test.set_fontproperties(fontprops)

        return fontprops
    finally:
        test.remove()


def _plot_polygons_only(gdf, ax, facecolor, zorder):
    """
    Bugfix (PR #16): filter out Point/MultiPoint geometries so they don't render as dots.
    We only plot Polygon/MultiPolygon geometries.
    """
    if gdf is None or not hasattr(gdf, "empty") or gdf.empty:
        return
    if not hasattr(gdf, "geometry") or gdf.geometry is None:
        return

    try:
        geom_types = gdf.geometry.geom_type
        polys = gdf[geom_types.isin(["Polygon", "MultiPolygon"])]
    except Exception:
        polys = gdf

    if hasattr(polys, "empty") and not polys.empty:
        polys.plot(ax=ax, facecolor=facecolor, edgecolor="none", zorder=zorder)


def _plot_lines_only(gdf, ax, color, linewidth, zorder):
    """
    Plot LineString/MultiLineString geometries only.
    """
    if gdf is None or not hasattr(gdf, "empty") or gdf.empty:
        return
    if not hasattr(gdf, "geometry") or gdf.geometry is None:
        return

    try:
        geom_types = gdf.geometry.geom_type
        lines = gdf[geom_types.isin(["LineString", "MultiLineString"])]
    except Exception:
        lines = gdf

    if hasattr(lines, "empty") and not lines.empty:
        lines.plot(ax=ax, color=color, linewidth=linewidth, zorder=zorder)


def create_poster(
    city,
    country,
    display_name,
    display_country,
    point,
    dist_x,
    dist_y,
    output_file,
    debug=False,
    roads_network=False,
    no_credits=False,
    road_type="all",
    no_rail=False
):
    print(f"\nGenerating map for {city}, {country}...")

    north, south, east, west = compute_bbox_from_center(point, dist_x, dist_y)
    poly_lonlat = bbox_to_poly_lonlat(north, south, east, west)

    if debug:
        print("\n[DEBUG] OSMnx version:", getattr(ox, "__version__", "<unknown>"))
        debug_bbox_step(debug, "Target bbox computed from center + dist_x/dist_y", north, south, east, west)

    # roads network distance requirement: dist_network = dist_y * 2.5
    dist_network_m = int(round(dist_y * 2.5))

    total_steps = 3 if no_rail else 4
    with tqdm(
        total=total_steps,
        desc="Fetching map data",
        unit="step",
        position=0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
        dynamic_ncols=True
    ) as pbar:

        # 1) Roads (graph)
        pbar.set_description("Downloading street network")
        with tqdm(
            total=1,
            desc="Street network details",
            unit="req",
            position=1,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
            dynamic_ncols=True
        ) as detail_pbar:
            if roads_network:
                if debug:
                    print("\n[DEBUG] Roads mode: dist_type='network'")
                    print(f"  center (lat, lon): {point[0]:.8f}, {point[1]:.8f}")
                    print(f"  dist_network_m (m): {dist_network_m}")
                G = graph_from_point_network(point, dist_network_m, network_type=road_type)
                debug_graph_bounds(debug, "Roads graph bounds (after network download)", G)
            else:
                if debug:
                    west_b, south_b, east_b, north_b = poly_lonlat.bounds
                    debug_bbox_step(debug, "Roads request (graph_from_polygon) bounds", north_b, south_b, east_b, west_b)
                G = graph_from_polygon_compat(poly_lonlat, network_type=road_type)
                debug_graph_bounds(debug, "Roads graph bounds (after polygon download)", G)
            detail_pbar.update(1)

        pbar.update(1)
        time.sleep(0.5)

        # 2) Water
        pbar.set_description("Downloading water features")
        try:
            if debug:
                west_b, south_b, east_b, north_b = poly_lonlat.bounds
                debug_bbox_step(debug, "Water request (features_from_polygon) bounds", north_b, south_b, east_b, west_b)
            water = fetch_features_with_progress(
                poly_lonlat,
                tags={'natural': 'water', 'waterway': 'riverbank'},
                task_label="Water features"
            )
        except Exception:
            water = None
        pbar.update(1)
        time.sleep(0.3)

        # 3) Parks
        pbar.set_description("Downloading parks/green spaces")
        try:
            if debug:
                west_b, south_b, east_b, north_b = poly_lonlat.bounds
                debug_bbox_step(debug, "Parks request (features_from_polygon) bounds", north_b, south_b, east_b, west_b)
            parks = fetch_features_with_progress(
                poly_lonlat,
                tags={'leisure': 'park', 'landuse': 'grass'},
                task_label="Parks/green spaces"
            )
        except Exception:
            parks = None
        pbar.update(1)
        time.sleep(0.3)

        # 4) Railways
        railways = None
        if not no_rail:
            pbar.set_description("Downloading railways")
            try:
                if debug:
                    west_b, south_b, east_b, north_b = poly_lonlat.bounds
                    debug_bbox_step(debug, "Railways request (features_from_polygon) bounds", north_b, south_b, east_b, west_b)
                railways = fetch_features_with_progress(
                    poly_lonlat,
                    tags={'railway': True},
                    task_label="Railways"
                )
            except Exception:
                railways = None
            pbar.update(1)

    print("✓ All data downloaded successfully!")

    print("Rendering map...")
    fig, ax = plt.subplots(figsize=POSTER_FIGSIZE, facecolor=THEME['bg'])
    ax.set_facecolor(THEME['bg'])
    ax.set_position([0, 0, 1, 1])

    # Layer 1: Polygons (PR #16 bugfix: avoid plotting point geometries as dots)
    _plot_polygons_only(water, ax=ax, facecolor=THEME['water'], zorder=0)
    _plot_polygons_only(parks, ax=ax, facecolor=THEME['parks'], zorder=0.5)

    # Layer 2: Roads
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(G)
    edge_widths = get_edge_widths_by_type(G)

    ox.plot_graph(
        G, ax=ax, bgcolor=THEME['bg'],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False, close=False
    )

    # Force roads above polygons
    for coll in ax.collections:
        coll.set_zorder(5)

    # Layer 2.5: Railways
    if not no_rail:
        _plot_lines_only(railways, ax=ax, color=THEME['railway'], linewidth=0.6, zorder=6)

    # If roads were fetched via network distance, enforce the poster bbox extents so the
    # gradients/features remain aligned to the intended poster framing.
    if roads_network:
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        if debug:
            print("\n[DEBUG] Enforced axes limits to poster bbox (because --network is enabled)")
            print(f"  xlim (west,east): {west:.8f}, {east:.8f}")
            print(f"  ylim (south,north): {south:.8f}, {north:.8f}")

    # Layer 3: Gradients (Top and Bottom)
    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)

    # Typography
    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=60)
        font_sub = FontProperties(fname=FONTS['light'], size=22)
        font_coords = FontProperties(fname=FONTS['regular'], size=14)
        font_attr = FontProperties(fname=FONTS['light'], size=8)
    else:
        font_main = FontProperties(family='monospace', weight='bold', size=60)
        font_sub = FontProperties(family='monospace', weight='normal', size=22)
        font_coords = FontProperties(family='monospace', size=14)
        font_attr = FontProperties(family='monospace', size=8)

    spaced_display = spaced_caps(display_name)
    font_main = fit_font_size_to_axes_width(fig, ax, spaced_display, font_main, max_width_frac=0.88, min_size=18)

    # --- BOTTOM TEXT ---
    ax.text(0.5, 0.14, spaced_display, transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_main, zorder=11)

    ax.text(0.5, 0.10, display_country.upper(), transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)

    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0:
        coords = coords.replace("E", "W")

    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)

    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes,
            color=THEME['text'], linewidth=1, zorder=11)

    # --- ATTRIBUTION (bottom right) ---
    if not no_credits:
        ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
                color=THEME['text'], alpha=0.5, ha='right', va='bottom',
                fontproperties=font_attr, zorder=11)

    # Save
    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=300, facecolor=THEME['bg'])
    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    """Print usage examples."""
    print(f"""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Notes about --distance:
  --distance (-d) is interpreted as the HALF-WIDTH (x) of the map in meters.
  The HALF-HEIGHT (y) is computed automatically from the poster aspect ratio:
    y = x * (height/width) = x * {POSTER_FIGSIZE[1]}/{POSTER_FIGSIZE[0]} = x * {POSTER_ASPECT:.6f}

  Example: -d 3000 -> x=3000m, y=4000m (for a 12:16 poster)

Roads network mode:
  --network (or -network) uses dist_type='network' for ROADS ONLY.
  In this mode, the road network distance is:
    dist_network = y / 2

  Example: -d 3000 -> y=4000 -> dist_network=2000m

Debug:
  --debug-bbox prints detailed bbox/polygon bounds and projected areas for each data request step.

Examples:
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 3000
  python create_map_poster.py -c "Paris" -C "France" -t noir -d 3000 --network
  python create_map_poster.py -c "Paris" -C "France" -t noir -d 3000 --network --debug-bbox
  python create_map_poster.py -c "Portland" -C "USA" -t noir -d 3000 --road-type drive
""")


def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return

    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, 'r') as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except Exception:
            display_name = theme_name
            description = ''
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city Paris --country France --theme noir --distance 3000 --network
  python create_map_poster.py --city Paris --country France --theme noir --distance 3000 --network --debug-bbox
  python create_map_poster.py --list-themes

Distance behavior:
  --distance (-d) sets HALF-WIDTH (x). HALF-HEIGHT (y) is derived from aspect {POSTER_FIGSIZE[0]}:{POSTER_FIGSIZE[1]}.
Road network behavior (optional):
  --network uses dist_type='network' for ROADS ONLY, with dist_network = y/2.
"""
    )

    parser.add_argument('--city', '-c', type=str, help='City name (used for lookup)')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--display-name', '-n', type=str, default=None,
                        help='Preferred printed name on poster (optional; defaults to --city)')
    parser.add_argument('--display-country', '-N', type=str, default=None,
                        help='Preferred printed country on poster (optional; defaults to --country)')
    parser.add_argument('--theme', '-t', type=str, default='feature_based',
                        help='Theme name (default: feature_based)')
    parser.add_argument('--distance', '-d', type=int, default=29000,
                        help='HALF-WIDTH (x) in meters (default: 29000). HALF-HEIGHT (y) is computed from poster aspect ratio.')
    parser.add_argument('--debug-bbox', action='store_true',
                        help='Print debug messages showing requested bbox/polygon areas per step')
    # As requested: support both --network and -network (unusual, but works).
    parser.add_argument('--network', '-network', dest='roads_network', action='store_true',
                        help="Use dist_type='network' for ROADS ONLY. Road dist = (derived y)/2.")
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')
    parser.add_argument('--no-credits', action='store_true',
                        help='Do not render the © OpenStreetMap contributors credit text')
    parser.add_argument('--no-rail', action='store_true',
                        help='Skip downloading and rendering rail networks')
    parser.add_argument('--road-type', '-r', type=str, default='all',
                        help="Road network type for OSMnx (default: all). Examples: drive, walk, bike.")


    args = parser.parse_args()

    if len(os.sys.argv) == 1:
        print_examples()
        os.sys.exit(0)

    if args.list_themes:
        list_themes()
        os.sys.exit(0)

    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        os.sys.exit(1)

    available_themes = get_available_themes()
    if args.theme not in available_themes:
        print(f"Error: Theme '{args.theme}' not found.")
        print(f"Available themes: {', '.join(available_themes)}")
        os.sys.exit(1)

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    THEME = load_theme(args.theme)

    display_name = args.display_name.strip() if args.display_name else args.city
    display_country = args.display_country.strip() if args.display_country else args.country

    dist_x = int(args.distance)
    dist_y = int(round(dist_x * POSTER_ASPECT))

    try:
        coords = get_coordinates(args.city, args.country)
        output_file = generate_output_filename(
            args.city,
            args.country,
            args.theme,
            display_name=display_name,
            display_country=display_country,
        )
        create_poster(
            args.city, args.country, display_name, display_country,
            coords, dist_x, dist_y, output_file,
            debug=args.debug_bbox,
            roads_network=args.roads_network,
            no_credits=args.no_credits,
            road_type=args.road_type,
            no_rail=args.no_rail
        )

        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        os.sys.exit(1)
