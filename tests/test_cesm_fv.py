import shutil
import warnings
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock

from monetio.sat._tropomi_l2_no2_mm import open_dataset

HERE = Path(__file__).parent


def retrieve_test_file():
    fn = "CAM_chem_merra2_FCSD_1deg_QFED_world_201909-01-09_small_sfc.nc"

    # Download to tests/data if not already present
    p = HERE / "data" / fn
    if not p.is_file():
        warnings.warn(f"Downloading test file {fn} for CESM-FV test")
        import requests

        r = requests.get(
            "https://csl.noaa.gov/groups/csl4/modeldata/melodies-monet/data/"
            + f"example_model_data/cesmfv_example/{fn}",
            stream=True,
        )
        r.raise_for_status()
        with open(p, "wb") as f:
            f.write(r.content)

    return p


@pytest.fixture(scope="session")
def test_file_path(tmp_path_factory, worker_id):
    if worker_id == "master":
        # Not executing with multiple workers;
        # let pytest's fixture caching do its job
        return retrieve_test_file()

    # Get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    # Copy to the shared test location
    p_test = root_tmp_dir / "cesm_fv_tst.nc"
    with FileLock(p_test.as_posix() + ".lock"):
        if p_test.is_file():
            return p_test
        else:
            p = retrieve_test_file()
            shutil.copy(p, p_test)
            return p_test


def _test_ds(ds):
    assert set(ds.dims) == {"time", "x", "y", "z"}

    # Test coordinates
    assert "lat" not in ds.data_vars
    assert "lon" not in ds.data_vars
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert np.all(ds.latitude.values[0, :] == ds.latitude.values[0,0])
    assert np.all(ds.longitude.values[:, 0] == ds.longitude.values[0,0])
    assert tuple(ds["O3"].dims) == ('time', 'z', 'y', 'x')
    assert ds['O3'].attrs['units'] == 'ppbv'


def test_open_mfdataset(test_file_path):
    ds = _cesm_fv.open_dataset(TEST_FP)
