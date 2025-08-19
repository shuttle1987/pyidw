"""
Created on 31 Jan, 2021
Update on 02 Feb, 2022
@author: Md. Yahya Tamim
version: 0.2.14
"""

import numpy as np
from math import sqrt, floor, ceil
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.crs import CRS  # pylint: disable=no-name-in-module
from rasterio.transform import from_bounds

# import fiona
from rasterio.enums import Resampling
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
# from matplotlib import colors


def show_map(
    input_raster="",
    colormap: str = "coolwarm",
    image_size: float = 1.5,
    return_figure: bool = False,
):
    """Legacy function to show a map or potentially return it based on the `return_figure` parameter"""
    with rasterio.open(input_raster) as image_data:
        my_matrix = image_data.read(1)
        my_matrix = np.ma.masked_where(my_matrix == 32767, my_matrix)
        fig, ax = plt.subplots()
        image_hidden = ax.imshow(my_matrix, cmap=colormap)
        plt.close()

        fig, ax = plt.subplots()
        fig.set_facecolor("w")
        width = fig.get_size_inches()[0] * image_size
        height = fig.get_size_inches()[1] * image_size
        fig.set_size_inches(w=width, h=height)
        image = show(image_data, cmap=colormap, ax=ax)
        cbar = fig.colorbar(image_hidden, ax=ax, pad=0.02)
        if not return_figure:
            plt.show()
        else:
            return fig, ax, cbar


#################################################


def crop_resize(
    input_raster_filename="", extent_shapefile_name="", max_height_or_width=250
):
    # Here, co-variable raster file (elevation in this case) is cropped and resized using rasterio.
    BD = gpd.read_file(extent_shapefile_name)
    elevation = rasterio.open(input_raster_filename)

    # Using mask method from rasterio.mask to clip study area from larger elevation file.
    cropped_data, cropped_transform = mask(
        dataset=elevation, shapes=BD.geometry, crop=True, all_touched=True
    )
    cropped_meta = elevation.meta
    cropped_meta.update(
        {
            "height": cropped_data.shape[-2],
            "width": cropped_data.shape[-1],
            "transform": cropped_transform,
        }
    )

    cropped_filename = input_raster_filename.rsplit(".", 1)[0] + "_cropped.tif"
    with rasterio.open(cropped_filename, "w", **cropped_meta) as cropped_file:
        cropped_file.write(
            cropped_data
        )  # Save the cropped file as cropped_elevation.tif to working directory.

    # Calculate resampling factor for resizing the elevation file, this is done to reduce calculation time.
    # The default value of 250 is chosen for optimal result, it can be more or less depending on users desire.
    resampling_factor = max_height_or_width / max(rasterio.open(cropped_filename).shape)

    # Reshape/resize the cropped elevation file and save it to working directory.
    with rasterio.open(cropped_filename, "r") as cropped_elevation:
        resampled_elevation = cropped_elevation.read(
            out_shape=(
                cropped_elevation.count,
                int(cropped_elevation.height * resampling_factor),
                int(cropped_elevation.width * resampling_factor),
            ),
            resampling=Resampling.bilinear,
        )

        resampled_transform = (
            cropped_elevation.transform
            * cropped_elevation.transform.scale(
                cropped_elevation.width / resampled_elevation.shape[-1],
                cropped_elevation.height / resampled_elevation.shape[-2],
            )
        )

        resampled_meta = cropped_elevation.meta
        resampled_meta.update(
            {
                "height": resampled_elevation.shape[-2],
                "width": resampled_elevation.shape[-1],
                "dtype": np.float64,
                "transform": resampled_transform,
            }
        )

        resampled_filename = input_raster_filename.rsplit(".", 1)[0] + "_resized.tif"
        with rasterio.open(resampled_filename, "w", **resampled_meta) as resampled_file:
            resampled_file.write(
                resampled_elevation
            )  # Save the resized file as resampled_elevation.tif in working directory.


#################################################


def blank_raster(extent_shapefile=""):
    calculationExtent = gpd.read_file(extent_shapefile)

    # calculationExtent should be a single row with info about the extent,
    # so we take that first row to extract the bounds
    minX = floor(calculationExtent.bounds.iloc[0].minx)
    minY = floor(calculationExtent.bounds.iloc[0].miny)
    maxX = ceil(calculationExtent.bounds.iloc[0].maxx)
    maxY = ceil(calculationExtent.bounds.iloc[0].maxy)
    longRange = sqrt((minX - maxX) ** 2)
    latRange = sqrt((minY - maxY) ** 2)

    gridWidth = 400
    pixelPD = gridWidth / longRange  # Pixel Per Degree
    gridHeight = floor(pixelPD * latRange)
    BlankGrid = np.ones([gridHeight, gridWidth])

    blank_filename = extent_shapefile.rsplit(".", 1)[0] + "_blank.tif"

    with rasterio.open(
        blank_filename,
        "w",
        driver="GTiff",
        height=BlankGrid.shape[0],
        width=BlankGrid.shape[1],
        count=1,
        dtype=BlankGrid.dtype,  # BlankGrid.dtype, np.float32, np.int16
        crs=CRS.from_string(calculationExtent.crs.srs),
        transform=from_bounds(
            minX, minY, maxX, maxY, BlankGrid.shape[1], BlankGrid.shape[0]
        ),
        nodata=32767,
    ) as dst:
        dst.write(BlankGrid, 1)


#################################################


# def standard_idw(lon, lat, elev, longs, lats, elevs, d_values, id_power, p_degree, s_radius):
def standard_idw(lon, lat, longs, lats, d_values, id_power, s_radius):
    """regression_idw is responsible for mathematic calculation of IDW interpolation with regression as a covariable."""
    calc_arr = np.zeros(
        shape=(len(longs), 6)
    )  # create an empty array shape of (total no. of observation * 6)
    calc_arr[:, 0] = longs  # First column will be Longitude of known data points.
    calc_arr[:, 1] = lats  # Second column will be Latitude of known data points.
    #     calc_arr[:, 2] = elevs    # Third column will be Elevation of known data points.
    calc_arr[:, 3] = (
        d_values  # Fourth column will be Observed data value of known data points.
    )

    # Fifth column is weight value from IDW formula `w = 1 / (d(x, x_i)^power + 1)`
    # >> constant 1 is to prevent int divide by zero when distance is zero.
    calc_arr[:, 4] = 1 / (
        np.sqrt((calc_arr[:, 0] - lon) ** 2 + (calc_arr[:, 1] - lat) ** 2) ** id_power
        + 1
    )

    # Sort the array in ascending order based on column_5 (weight) `np.argsort(calc_arr[:,4])`
    # and exclude all the rows outside of search radius `[ - s_radius :, :]`
    calc_arr = calc_arr[np.argsort(calc_arr[:, 4])][-s_radius:, :]

    # Sixth column is the multiplicative product of inverse distance weight and actual value.
    calc_arr[:, 5] = calc_arr[:, 3] * calc_arr[:, 4]
    # Divide sum of weighted value vy sum of weights to get IDW interpolation.
    idw = calc_arr[:, 5].sum() / calc_arr[:, 4].sum()
    return idw


#################################################


def idw_interpolation(
    input_point_shapefile="",
    extent_shapefile="",
    column_name="",
    power: int = 2,
    search_radius: int = 1,
    output_resolution: int = 250,
    *,
    render_map=True,
):
    """
    Perform an interpolation of the data in the shapefile and extent using `column_name` to specify
    which column the data to be interpolated resides.

    render_map: toggle for rendering the the map, defaults as True
    """
    blank_raster(extent_shapefile)

    blank_filename = extent_shapefile.rsplit(".", 1)[0] + "_blank.tif"
    crop_resize(
        input_raster_filename=blank_filename,
        extent_shapefile_name=extent_shapefile,
        max_height_or_width=output_resolution,
    )

    resized_raster_name = blank_filename.rsplit(".", 1)[0] + "_resized.tif"
    #     baseRasterFile = rasterio.open(resized_raster_name) # baseRasterFile stands for resampled elevation.

    with rasterio.open(resized_raster_name) as baseRasterFile:
        inputPoints = gpd.read_file(input_point_shapefile)
        # obser_df stands for observation_dataframe, lat, lon, data_value for each station will be stored here.
        obser_df = pd.DataFrame()
        obser_df["station_name"] = inputPoints.iloc[:, 0]

        # create two list of indexes of station longitude, latitude in elevation raster file.
        lons, lats = baseRasterFile.index(
            [lon for lon in inputPoints.geometry.x],
            [lat for lat in inputPoints.geometry.y],
        )
        obser_df["lon_index"] = lons
        obser_df["lat_index"] = lats
        obser_df["data_value"] = inputPoints[column_name]

        idw_array = baseRasterFile.read(1)
        for x in range(baseRasterFile.height):
            for y in range(baseRasterFile.width):
                if baseRasterFile.read(1)[x][y] == 32767:
                    continue
                else:
                    idw_array[x][y] = standard_idw(
                        lon=x,
                        lat=y,
                        longs=obser_df.lon_index,
                        lats=obser_df.lat_index,
                        d_values=obser_df.data_value,
                        id_power=power,
                        s_radius=search_radius,
                    )

        output_filename = input_point_shapefile.rsplit(".", 1)[0] + "_idw.tif"
        with rasterio.open(output_filename, "w", **baseRasterFile.meta) as std_idw:
            std_idw.write(idw_array, 1)

        if render_map:
            show_map(output_filename)


#################################################


def accuracy_standard_idw(
    input_point_shapefile="",
    extent_shapefile="",
    column_name="",
    power: int = 2,
    search_radius: int = 1,
    output_resolution: int = 250,
):
    blank_raster(extent_shapefile)

    blank_filename = extent_shapefile.rsplit(".", 1)[0] + "_blank.tif"
    crop_resize(
        input_raster_filename=blank_filename,
        extent_shapefile_name=extent_shapefile,
        max_height_or_width=output_resolution,
    )

    resized_raster_name = blank_filename.rsplit(".", 1)[0] + "_resized.tif"

    with rasterio.open(resized_raster_name) as baseRasterFile:
        inputPoints = gpd.read_file(input_point_shapefile)
        # obser_df stands for observation_dataframe, lat, lon, data_value for each station will be stored here.
        obser_df = pd.DataFrame()
        obser_df["station_name"] = inputPoints.iloc[:, 0]

        # create two list of indexes of station longitude, latitude in elevation raster file.
        lons, lats = baseRasterFile.index(
            [lon for lon in inputPoints.geometry.x],
            [lat for lat in inputPoints.geometry.y],
        )
        obser_df["lon_index"] = lons
        obser_df["lat_index"] = lats
        obser_df["data_value"] = inputPoints[column_name]
        obser_df["predicted"] = 0.0

        cv = LeaveOneOut()
        for train_ix, test_ix in cv.split(obser_df):
            test_point = obser_df.iloc[test_ix[0]]
            train_df = obser_df.iloc[train_ix]

            obser_df.loc[test_ix[0], "predicted"] = standard_idw(
                lon=test_point.lon_index,
                lat=test_point.lon_index,
                longs=train_df.lon_index,
                lats=train_df.lat_index,
                d_values=train_df.data_value,
                id_power=power,
                s_radius=search_radius,
            )
        return obser_df.data_value.to_list(), obser_df.predicted.to_list()


#################################################


def regression_idw(
    lon,
    lat,
    elev,
    longs,
    lats,
    elevs,
    d_values,
    id_power,
    p_degree,
    s_radius,
    x_max,
    x_min,
):
    """regression_idw is responsible for mathematic calculation of idw interpolation with regression as a covariable."""
    calc_arr = np.zeros(
        shape=(len(longs), 6)
    )  # create an empty array shape of (total no. of observation * 6)
    calc_arr[:, 0] = longs  # First column will be Longitude of known data points.
    calc_arr[:, 1] = lats  # Second column will be Latitude of known data points.
    calc_arr[:, 2] = elevs  # Third column will be Elevation of known data points.
    calc_arr[:, 3] = (
        d_values  # Fourth column will be Observed data value of known data points.
    )

    # Fifth column is weight value from idw formula " w = 1 / (d(x, x_i)^power + 1)"
    # >> constant 1 is to prevent int divide by zero when distance is zero.
    calc_arr[:, 4] = 1 / (
        np.sqrt((calc_arr[:, 0] - lon) ** 2 + (calc_arr[:, 1] - lat) ** 2) ** id_power
        + 1
    )

    # Sort the array in ascending order based on column_5 (weight) "np.argsort(calc_arr[:,4])"
    # and exclude all the rows outside of search radius "[ - s_radius :, :]"
    calc_arr = calc_arr[np.argsort(calc_arr[:, 4])][-s_radius:, :]

    # Sixth column is multiplicative product of inverse distant weight and actual value.
    calc_arr[:, 5] = calc_arr[:, 3] * calc_arr[:, 4]
    # Divide sum of weighted value vy sum of weights to get IDW interpolation.
    idw = calc_arr[:, 5].sum() / calc_arr[:, 4].sum()

    # Create polynomial regression equation where independent variable is elevation and dependent variable is data_value.
    # Then, calculate R_squared value for just fitted polynomial equation.
    poly_reg = np.poly1d(np.polyfit(x=calc_arr[:, 2], y=calc_arr[:, 3], deg=p_degree))
    r_squared = r2_score(calc_arr[:, 3], poly_reg(calc_arr[:, 2]))

    regression_idw_combined = (1 - r_squared) * idw + r_squared * poly_reg(elev)
    if regression_idw_combined >= x_min and regression_idw_combined <= x_max:
        return regression_idw_combined
    elif regression_idw_combined < x_min:
        return x_min
    elif regression_idw_combined > x_max:
        return x_max


#################################################


class sigmoidStandardization:
    def __init__(self, input_array):
        self.in_array = input_array
        self.arr_mean = self.in_array.mean()
        self.arr_std = self.in_array.std()

    def transform(self, number):
        self.transformed = 1 / (1 + np.exp(-(number - self.arr_mean) / self.arr_std))
        return self.transformed

    def inverse_transform(self, number):
        self.reverse_transformed = (
            np.log(number / (1 - number)) * self.arr_std + self.arr_mean
        )
        return self.reverse_transformed


#################################################


def regression_idw_interpolation(
    input_point_shapefile="",
    input_raster_file="",
    extent_shapefile="",
    column_name="",
    power: int = 2,
    polynomial_degree: int = 1,
    search_radius: int = 1,
    output_resolution: int = 250,
):
    crop_resize(
        input_raster_filename=input_raster_file,
        extent_shapefile_name=extent_shapefile,
        max_height_or_width=output_resolution,
    )

    metStat = gpd.read_file(
        input_point_shapefile
    )  # metStat stands for meteorological stations.

    resampled_filename = input_raster_file.rsplit(".", 1)[0] + "_resized.tif"

    with rasterio.open(resampled_filename) as re_elevation:
        # obser_df stands for observation_dataframe, lat, lon, elevation, data_value for each station will be stored here.
        obser_df = pd.DataFrame()
        obser_df["station_name"] = metStat.iloc[:, 0]

        # create two list of indexes of station longitude, latitude in elevation raster file.
        lons, lats = re_elevation.index(
            [lon for lon in metStat.geometry.x], [lat for lat in metStat.geometry.y]
        )
        obser_df["lon_index"] = lons
        obser_df["lat_index"] = lats
        obser_df["elevation"] = re_elevation.read(1)[
            lons, lats
        ]  # read elevation data for each station.
        obser_df["data_value"] = metStat[column_name]
        obser_df["predicted"] = 0.0

        raster_transform = sigmoidStandardization(obser_df["elevation"])
        obser_df["trnsfrmd_raster"] = raster_transform.transform(obser_df["elevation"])

        upper_range = obser_df["data_value"].max() + obser_df["data_value"].std()
        lower_range = obser_df["data_value"].min() - obser_df["data_value"].std()

        regression_idw_array = re_elevation.read(1)
        for x in range(re_elevation.height):
            for y in range(re_elevation.width):
                if re_elevation.read(1)[x][y] == 32767:
                    continue
                else:
                    regression_idw_array[x][y] = regression_idw(
                        lon=x,
                        lat=y,
                        elev=raster_transform.transform(re_elevation.read(1)[x][y]),
                        longs=obser_df.lon_index,
                        lats=obser_df.lat_index,
                        elevs=obser_df["trnsfrmd_raster"],
                        d_values=obser_df.data_value,
                        id_power=power,
                        p_degree=polynomial_degree,
                        s_radius=search_radius,
                        x_max=upper_range,
                        x_min=lower_range,
                    )

        output_filename = (
            input_point_shapefile.rsplit(".", 1)[0] + "_regression_idw.tif"
        )
        with rasterio.open(output_filename, "w", **re_elevation.meta) as reg_idw:
            reg_idw.write(regression_idw_array, 1)

        show_map(output_filename)


#################################################


def accuracy_regression_idw(
    input_point_shapefile="",
    input_raster_file="",
    extent_shapefile="",
    column_name="",
    power: int = 2,
    polynomial_degree: int = 1,
    search_radius: int = 1,
    output_resolution: int = 250,
):
    crop_resize(
        input_raster_filename=input_raster_file,
        extent_shapefile_name=extent_shapefile,
        max_height_or_width=output_resolution,
    )

    metStat = gpd.read_file(
        input_point_shapefile
    )  # metStat stands for meteorological stations.

    resampled_filename = input_raster_file.rsplit(".", 1)[0] + "_resized.tif"

    with rasterio.open(resampled_filename) as re_elevation:
        # obser_df stands for observation_dataframe, lat, lon, elevation, data_value for each station will be stored here.
        obser_df = pd.DataFrame()
        obser_df["station_name"] = metStat.iloc[:, 0]

        # create two list of indexes of station longitude, latitude in elevation raster file.
        lons, lats = re_elevation.index(
            [lon for lon in metStat.geometry.x], [lat for lat in metStat.geometry.y]
        )
        obser_df["lon_index"] = lons
        obser_df["lat_index"] = lats
        obser_df["elevation"] = re_elevation.read(1)[
            lons, lats
        ]  # read elevation data for each station.
        obser_df["data_value"] = metStat[column_name]
        obser_df["predicted"] = 0.0
        raster_transform = sigmoidStandardization(obser_df["elevation"])
        obser_df["trnsfrmd_raster"] = raster_transform.transform(obser_df["elevation"])
        upper_range = obser_df["data_value"].max() + obser_df["data_value"].std()
        lower_range = obser_df["data_value"].min() - obser_df["data_value"].std()

        cv = LeaveOneOut()
        for train_ix, test_ix in cv.split(obser_df):
            test_point = obser_df.iloc[test_ix[0]]
            train_df = obser_df.iloc[train_ix]

            obser_df.loc[test_ix[0], "predicted"] = regression_idw(
                lon=test_point.lon_index,
                lat=test_point.lon_index,
                elev=test_point["trnsfrmd_raster"],
                longs=train_df.lon_index,
                lats=train_df.lat_index,
                elevs=train_df["trnsfrmd_raster"],
                d_values=train_df.data_value,
                id_power=power,
                p_degree=polynomial_degree,
                s_radius=search_radius,
                x_max=upper_range,
                x_min=lower_range,
            )

        return obser_df.data_value.to_list(), obser_df.predicted.to_list()
