import pytest

def test_interpolation_regression(pytestconfig):
    """This test case is to ensure a type error issue is resolved"""
    from pyidw import idw

    example_data_directory = pytestconfig.rootdir / "tests" / "example_data"

    bangladesh_temperature_file = example_data_directory / "Bangladesh_Temperature.shp"
    bangladesh_extent_file = example_data_directory / "Bangladesh_Border.shp"
    idw.idw_interpolation(
        input_point_shapefile=str(bangladesh_temperature_file),
        extent_shapefile=str(bangladesh_extent_file),
        column_name="Max_Temp",
        power=2,
        search_radius=10,
        output_resolution=250,
        render_map=False,
    )

@pytest.mark.filterwarnings("error")
def test_futurewarning_idw_interpolation(pytestconfig):
    """This test case is to ensure the way in which the bounds are extracted will continue
    to work with future versions of geopandas"""
    from pyidw import idw

    example_data_directory = pytestconfig.rootdir / "tests" / "example_data"

    bangladesh_temperature_file = example_data_directory / "Bangladesh_Temperature.shp"
    bangladesh_extent_file = example_data_directory / "Bangladesh_Border.shp"
    idw.idw_interpolation(
        input_point_shapefile=str(bangladesh_temperature_file),
        extent_shapefile=str(bangladesh_extent_file),
        column_name="Max_Temp",
        power=2,
        search_radius=10,
        output_resolution=250,
        render_map=False,
    )

def test_index_error_bug(pytestconfig):
    """
    The implementation of `regression_idw_interpolation` caused an IndexError to be raised
    after updating various underlying dependencies.
    
    This test case serves as a regression test for that.
    """
    from pyidw import idw

    example_data_directory = pytestconfig.rootdir / "tests" / "example_data"

    bangladesh_temperature_file = example_data_directory / "Bangladesh_Temperature.shp"
    bangladesh_extent_file = example_data_directory / "Bangladesh_Border.shp"
    elevation_input_raster_filename = example_data_directory / "Bangladesh_Elevation.tiff"

    idw.regression_idw_interpolation(
        input_point_shapefile=str(bangladesh_temperature_file),
        input_raster_file=str(elevation_input_raster_filename),
        extent_shapefile=str(bangladesh_extent_file),
        column_name="Min_Temp",
        power=2,
        polynomial_degree=1,
        search_radius=5,
        output_resolution=250,
        render_map=False,
    )


def test_index_error_bug_accuracy_regression_idw(pytestconfig):
    """
    The implementation of `accuracy_regression_idw` caused an IndexError to be raised
    after updating various underlying dependencies.

    This test case serves as a regression test for that.
    """
    from pyidw import idw

    example_data_directory = pytestconfig.rootdir / "tests" / "example_data"

    bangladesh_temperature_file = example_data_directory / "Bangladesh_Temperature.shp"
    bangladesh_extent_file = example_data_directory / "Bangladesh_Border.shp"
    elevation_input_raster_filename = example_data_directory / "Bangladesh_Elevation.tiff"

    original_value, interpolated_value = idw.accuracy_regression_idw(
        input_point_shapefile=str(bangladesh_temperature_file),
        input_raster_file=str(elevation_input_raster_filename),
        extent_shapefile=str(bangladesh_extent_file),
        column_name="Min_Temp",
        power=2,
        polynomial_degree=1,
        search_radius=5,
        output_resolution=250,
    )