import warnings
import numpy as np
import xarray as xr


def open_mfdataset(files, var_list=None, surf_only=False, **kwargs):
    """Method to open Chimere model netcdf output files.
    Parameters
    ----------
    files : list[str]
        files is a list of path(s) of the file(s).
    var_list: list[str]
        list of variable names meant to be kept for the analysis.
    surf_only: bool
        boolean flag specifying if only surface data (layer 0) should be kept for analysis.
    Returns
    -------
    xarray.Dataset
        Chimere model dataset in standard format for use
        in MELODIES-MONET
    """
    if not isinstance(files, (list, tuple, np.ndarray)):
        files = [files]

    datasets = []
    for file in files:
        datasets.append(xr.open_dataset(file))

    # get the data_vars wanted
    if var_list is None:
        var_list = []

    if not surf_only:
        var_list = var_list + list_met_3D_vars(datasets[0])

    drop_data_vars = set(list(datasets[0].data_vars)) - set(var_list)

    for n, ds in enumerate(datasets):
        datasets[n] = ds.drop_vars(drop_data_vars, errors="ignore")

    xrds = xr.concat(datasets, "time_counter")

    xrds = xrds.rename(
        {"nav_lat": "latitude", "nav_lon": "longitude", "time_counter": "time", "bottom_top": "z"}
    )

    if not surf_only:
        rename_dict = {}
        if "temp" in xrds.variables:
            rename_dict["temp"] = "temperature_k"
        if "pres" in xrds.variables:
            rename_dict["pres"] = "pres_pa_mid"
        if "thlay" in xrds.variables:
            rename_dict["thlay"] = "dz_m"
        xrds = xrds.rename(rename_dict)

    if surf_only:
        xrds = xrds.isel(z=0).expand_dims("z", axis=1)

    xrds = xrds.reset_coords()
    xrds = xrds.set_coords(["latitude", "longitude"])
    z = list(range(len(xrds["z"].values)))
    xrds["z"] = (('z',), z)

    return xrds


def list_met_3D_vars(dataset):
    """List the 3D met variables

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the variables

    Returns
    -------
    list[str]
        List containing the 3D vars
    """

    all_vars = set(list(dataset.data_vars))

    vars_3D = []

    if "thlay" in all_vars:
        vars_3D.append("thlay")
    else:
        warnings.warn(
                "thlay is not a variable. Currently, no other method to calculate the "
                "the layer thickness is implemented."
        )
    if "temp" in all_vars:
        vars_3D.append("temp")
    else:
        warnings.warn("3D temperature not available")
    if "pres" in all_vars:
        vars_3D.append("pres")
    else:
        warnings.warn("currently adding the pressure requires the pres variable. "
        "We might implement using the hybrid coordinates in the future")
    return vars_3D
