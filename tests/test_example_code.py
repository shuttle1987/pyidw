def test_interpolation_regression():
    """This test case is to ensure a type error issue is resolved"""
    from pyidw import idw

    from pathlib import Path
    example_data_directory = Path("example_data")

    bangladesh_temperature_file = example_data_directory / "Bangladesh_Temperature.shp"
    bangladesh_extent_file = example_data_directory / "Bangladesh_Border.shp"
    idw.idw_interpolation(
        input_point_shapefile=str(bangladesh_temperature_file),
        extent_shapefile=str(bangladesh_extent_file),
        column_name="Max_Temp",
        power=2,
        search_radius=10,
        output_resolution=250,
    )
