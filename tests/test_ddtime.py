"""Tests for the AMDA/DDBASE time axis, decoded from the master file's UNITS.

DDBASE stores time either as DDTime character records (YYYYDDDHHMMSSMMM, with
a 0-based day-of-year) or as plain seconds since the Unix epoch. Neither is
described by any ISTP attribute, and the data files carry no usable units of
their own -- several carry a wrong one inherited from a CDF conversion -- so
the declaration in the master file is the only thing worth trusting.
"""

import numpy as np
import netCDF4
import pytest

import pyistp
from pyistp._impl import _ddtime_to_datetime64, _decode_epoch
from pyistp.support_data_variable import SupportDataVariable

# Cross-checked against the DDBASE files themselves: the first record of ACE
# imf20071231.nc is 2007364000008000 and its _info.nc says the dataset starts
# at 2007-12-31T00:00:08.000Z.
DDTIME_CASES = [
    (b"2007364000008000", "2007-12-31T00:00:08.000"),
    (b"2008000000000000", "2008-01-01T00:00:00.000"),   # DDD is 0-based
    (b"2008365235959999", "2008-12-31T23:59:59.999"),   # 2008 is a leap year
    (b"2007364235959999", "2007-12-31T23:59:59.999"),   # 2007 is not
]


def char_array(records, width=17):
    return np.stack([np.frombuffer(r.ljust(width, b"\x00"), dtype="S1")
                     for r in records])


def axis(values, units):
    return SupportDataVariable(name="Time", values=values, attributes={"UNITS": units},
                               is_nrv=False, cdf_type="CDF_DOUBLE")


@pytest.mark.parametrize("record,expected", DDTIME_CASES)
def test_ddtime_record(record, expected):
    got = _ddtime_to_datetime64(char_array([record]))
    assert got[0] == np.datetime64(expected, "ns")


def test_ddtime_without_trailing_null():
    """The CDAWeb-converted datasets store 16 characters, not 17."""
    assert (_ddtime_to_datetime64(char_array([b"2007364000008000"], width=16))
            == _ddtime_to_datetime64(char_array([b"2007364000008000"]))).all()


def test_ddtime_empty():
    got = _ddtime_to_datetime64(np.empty((0, 17), "S1"))
    assert got.shape == (0,) and got.dtype == np.dtype("datetime64[ns]")


def test_decode_ddtime_axis():
    a = _decode_epoch(axis(char_array([r for r, _ in DDTIME_CASES]), "DDTIME"))
    assert a.values.dtype == np.dtype("datetime64[ns]")
    assert a.cdf_type == "CDF_TIME_TT2000"


def test_decode_unix_seconds_axis():
    # so_pas_3d, first record of pas_20170707_V00.nc
    a = _decode_epoch(axis(np.array([1499444538.5]), "seconds since 1970-01-01"))
    assert a.values[0] == np.datetime64("2017-07-07T16:22:18.500", "ns")
    assert a.cdf_type == "CDF_TIME_TT2000"


def test_decode_leaves_everything_else_alone():
    """An energy axis in eV, or a lag axis in seconds, must not be touched."""
    for units in ("eV", "s", "ms", "nT", ""):
        values = np.array([1.0, 2.0, 3.0])
        a = _decode_epoch(axis(values, units))
        assert a.values is values, units
        assert a.cdf_type == "CDF_DOUBLE", units


def test_decode_empty_skeleton_epoch():
    """A skeleton loaded on its own has 0 records and no characters to parse."""
    values = np.empty((0,), dtype=np.float64)
    a = _decode_epoch(axis(values, "DDTIME"))
    assert a.values is values


def _write_nc(path, time_var, with_values):
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("Time", None)
    ds.createDimension("TimeLength", 17)
    ds.createDimension("IMF", 3)
    time = ds.createVariable("Time", *time_var)
    imf = ds.createVariable("IMF", "f4", ("Time", "IMF"))
    imf.VAR_TYPE = "data"
    imf.DEPEND_0 = "Time"
    imf.UNITS = "nT"
    if with_values:
        time[:] = (char_array([r for r, _ in DDTIME_CASES])
                   if time_var[0] == "S1" else np.array([1499444538.5]))
        imf[:] = np.arange(len(time[:]) * 3, dtype="f4").reshape(-1, 3)
    else:
        # The master declares the format; only it knows what the values mean.
        time.UNITS = "DDTIME" if time_var[0] == "S1" else "seconds since 1970-01-01"
        time.VAR_TYPE = "support_data"
    ds.close()


@pytest.mark.parametrize("time_var,expected_first", [
    (("S1", ("Time", "TimeLength")), "2007-12-31T00:00:08.000"),
    (("f8", ("Time",)), "2017-07-07T16:22:18.500"),
])
# netCDF4 1.7.4's own C extension reshapes a numpy array in place when writing
# an unlimited-dimension variable, which numpy 2.5 deprecates; nothing here
# triggers it, and there is no newer netCDF4 release yet that avoids it.
@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_master_mode_decodes_the_epoch(tmp_path, time_var, expected_first):
    """The whole point: metadata from the master, values from the data file."""
    master, data = tmp_path / "master.nc", tmp_path / "data.nc"
    _write_nc(master, time_var, with_values=False)
    _write_nc(data, time_var, with_values=True)

    istp = pyistp.load(master_file=str(master), file=str(data))
    assert istp.data_variables() == ["IMF"]
    var = istp.data_variable("IMF")
    epoch = var.axes[0]
    assert epoch.values.dtype == np.dtype("datetime64[ns]")
    assert epoch.values[0] == np.datetime64(expected_first, "ns")
    assert len(epoch.values) == var.values.shape[0]


def _write_nc_with_offset_axis(path, with_values):
    """COUNTS(Time, Offset): DEPEND_0=Time (DDTIME), DEPEND_1=Offset, both
    declared in 'seconds since ...' units -- the so_pas_3d Duration scenario
    named in the README, where only the DEPEND_0 axis may be converted."""
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("Time", None)
    ds.createDimension("TimeLength", 17)
    ds.createDimension("Offset", 3)
    time = ds.createVariable("Time", "S1", ("Time", "TimeLength"))
    offset = ds.createVariable("Offset", "f4", ("Offset",))
    counts = ds.createVariable("COUNTS", "f4", ("Time", "Offset"))
    counts.VAR_TYPE = "data"
    counts.DEPEND_0 = "Time"
    counts.DEPEND_1 = "Offset"
    counts.UNITS = "counts"
    if with_values:
        time[:] = char_array([r for r, _ in DDTIME_CASES])
        offset[:] = np.array([0.0, 60.0, 120.0], dtype="f4")
        counts[:] = np.arange(len(time[:]) * 3, dtype="f4").reshape(-1, 3)
    else:
        time.UNITS = "DDTIME"
        time.VAR_TYPE = "support_data"
        # Not time -- an offset that happens to use a seconds-since unit.
        offset.UNITS = "seconds since 1970-01-01"
        offset.VAR_TYPE = "support_data"
    ds.close()


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_get_axes_only_converts_depend_0(tmp_path):
    """A DEPEND_1 axis in seconds-since units must be left alone even though
    the decoder would happily convert it if asked -- only DEPEND_0 is."""
    master, data = tmp_path / "master.nc", tmp_path / "data.nc"
    _write_nc_with_offset_axis(master, with_values=False)
    _write_nc_with_offset_axis(data, with_values=True)

    istp = pyistp.load(master_file=str(master), file=str(data))
    var = istp.data_variable("COUNTS")
    time_axis, offset_axis = var.axes

    assert time_axis.name == "Time"
    assert time_axis.values.dtype == np.dtype("datetime64[ns]")

    assert offset_axis.name == "Offset"
    assert offset_axis.values.dtype != np.dtype("datetime64[ns]")
    assert list(offset_axis.values) == [0.0, 60.0, 120.0]
