""" MOPITT gridded data File reader
    updated 2023-08 rrb
         * Added units
    updated 2022-10 rrb
         * DataSet instead of DataArray
    created 2021-12 rrb
"""
import glob

import pandas as pd
import xarray as xr


def get_start_time(filename):
    """Method to read the time in MOPITT level 3 hdf files.

    Parameters
    ----------
    filename : str
        Path to the file.

    Returns
    -------
    pandas.Timestamp or None or 0
    """
    import h5py

    structure = "/HDFEOS/ADDITIONAL/FILE_ATTRIBUTES"

    try:
        inFile = h5py.File(filename, "r")
    except Exception:
        print("ERROR: CANNOT OPEN " + filename)
        return 0

    grp = inFile[structure]
    k = grp.attrs
    startTimeBytes = k.get("StartTime", default=None)
    startTime = pd.to_datetime(startTimeBytes[0], unit="s", origin="1993-01-01 00:00:00")

    try:
        inFile.close()
    except Exception:
        print("ERROR CANNOT CLOSE " + filename)
        return 0

    return startTime


def loadAndExtractGriddedHDF(filename, varname):
    """Method to open MOPITT gridded hdf files.
    Masks data that is missing (turns into np.nan).

    Parameters
    ----------
    filename : str
        Path to the file.
    varname : str
        The variable to load from the MOPITT file.

    Returns
    -------
    xarray.Dataset
    """
    import h5py

    # initialize into dataset
    ds = xr.Dataset()

    # load the dimensions
    he5_load = h5py.File(filename, mode="r")
    lat = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/Latitude"][:]
    lon = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/Longitude"][:]
    alt = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/Pressure2"][:]
    alt_short = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/Pressure"][:]

    # 2D or 3D variables to choose from
    variable_dict = {
        "column": "/HDFEOS/GRIDS/MOP03/Data Fields/RetrievedCOTotalColumnDay",
        "apriori_col": "/HDFEOS/GRIDS/MOP03/Data Fields/APrioriCOTotalColumnDay",
        "apriori_surf": "/HDFEOS/GRIDS/MOP03/Data Fields/APrioriCOSurfaceMixingRatioDay",
        "pressure_surf": "/HDFEOS/GRIDS/MOP03/Data Fields/SurfacePressureDay",
        "ak_col": "/HDFEOS/GRIDS/MOP03/Data Fields/TotalColumnAveragingKernelDay",
        "ak_prof": "/HDFEOS/GRIDS/MOP03/Data Fields/APrioriCOMixingRatioProfileDay",
    }
    try:
        data_loaded = he5_load[variable_dict[varname]][:]
    except Exception:
        print("ERROR: Cannot load " + varname + " from " + filename)
        return 0

    he5_load.close()

    # DEBEG
    # print(data_loaded.shape)

    # create xarray DataArray
    if varname == "column":
        ds[varname] = xr.DataArray(data_loaded, dims=["lon", "lat"], coords=[lon, lat],
                                   attrs={"long_name":"Retrieved CO Total Column", "units":"molec/cm^2",})
        # missing value -> nan
        ds[varname] = ds[varname].where(ds[varname] != -9999.0)
    elif varname == "ak_col":
        ds[varname] = xr.DataArray(data_loaded, dims=["lon", "lat", "alt"], coords=[lon, lat, alt],
                                   attrs={"long_name":"Total Column Averaging Kernel", "units":"mol/(cm^2 log(VMR))",})
    elif varname == "apriori_prof":
        ds[varname] = xr.DataArray(data_loaded, dims=["lon", "lat", "alt"], coords=[lon, lat, alt_short],
                                   attrs={'long_name':'A Priori CO Mixing Ratio Profile', 'units':'ppbv',})

    return ds


def read_mopittdataset(files, varname):
    """Loop through files to open the MOPITT level 3 data.

    Parameters
    ----------
    files : str or list of str
        The full path to the file or files. Can take a file template with wildcard (*) symbol.
    varname : str
        The variable to load from the MOPITT file.

    Returns
    -------
    xarray.Dataset
    """

    count = 0
    filelist = sorted(glob.glob(files, recursive=False))

    for filename in filelist:
        print(filename)
        data = loadAndExtractGriddedHDF(filename, varname)
        time = get_start_time(filename)
        data = data.expand_dims(axis=0, time=[time])
        if count == 0:
            full_dataset = data
            count += 1
        else:
            full_dataset = xr.concat([full_dataset, data], "time")

    return full_dataset
