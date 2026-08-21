from .drivers import current_driver, Driver


def _driver_factory(file_or_buffer):
    if isinstance(file_or_buffer, bytes):
        magic = file_or_buffer[:4]
    else:
        with open(file_or_buffer, "rb") as f:
            magic = f.read(4)
    # netCDF3 files start with "CDF" followed by a version byte, so comparing
    # the four bytes read against b'CDF' never matched and they fell through to
    # the CDF driver, which opens them and then returns None for every variable.
    if magic == b'\x89HDF' or (magic[:3] == b'CDF'
                               and magic[3:4] in (b'\x01', b'\x02', b'\x05')):
        from .drivers.netcdf import Driver as NetCDFDriver
        return NetCDFDriver(file_or_buffer)
    return current_driver(file_or_buffer)
from .data_variable import DataVariable
from .support_data_variable import SupportDataVariable
import re
import numpy as np
from typing import List, Optional
import logging

DEPEND_REGEX = re.compile("DEPEND_\\d", re.IGNORECASE)

ISTP_NOT_COMPLIANT_W = "Non compliant ISTP file"

log = logging.getLogger(__name__)

# AMDA/DDBASE stores its time axis in two formats no ISTP attribute describes:
# DDTime character records, and plain seconds since the Unix epoch. The data
# files carry no usable units of their own -- several carry a wrong one
# inherited from a CDF conversion -- so the master file is what decides here.
DDTIME_UNITS = "DDTIME"
UNIX_SECONDS_RE = re.compile(r"^seconds\s+since\s+1970-01-01", re.IGNORECASE)


def _ddtime_to_datetime64(chars):
    """YYYYDDDHHMMSSMMM character records -> datetime64[ns]. DDD is 0-based."""

    def field(start, stop):
        column = np.ascontiguousarray(chars[:, start:stop])
        return column.view(f"S{stop - start}").ravel().astype(np.int64)

    year, doy = field(0, 4), field(4, 7)
    hour, minute = field(7, 9), field(9, 11)
    second, milli = field(11, 13), field(13, 16)
    days = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]")
    days = days + doy.astype("timedelta64[D]")
    ms = ((hour * 60 + minute) * 60 + second) * 1000 + milli
    return (days.astype("datetime64[ms]")
            + ms.astype("timedelta64[ms]")).astype("datetime64[ns]")


def _decode_epoch(axis):
    """Decode a time axis whose storage format only the master file declares.

    Values are left untouched unless the declared units ask for a conversion,
    so this cannot disturb a file that already carries a real CDF epoch.
    """
    units = str(axis.attributes.get("UNITS", "")).strip()
    values = np.asarray(axis.values)
    if units.upper() == DDTIME_UNITS:
        if values.dtype.kind in ("S", "U") and values.ndim == 2 and values.shape[1] >= 16:
            axis.values = _ddtime_to_datetime64(values)
            axis.cdf_type = "CDF_TIME_TT2000"
        elif values.size:
            log.warning(
                f"{ISTP_NOT_COMPLIANT_W}: {axis.name} declares {DDTIME_UNITS} but holds "
                f"{values.dtype} of shape {values.shape}, leaving it as is")
    elif UNIX_SECONDS_RE.match(units) and values.dtype.kind in ("f", "i", "u"):
        # Scaled through microseconds: float64 cannot hold nanoseconds since
        # 1970 without losing the last couple of hundred of them.
        micro = np.round(np.asarray(values, dtype="f8") * 1e6).astype("int64")
        axis.values = micro.astype("datetime64[us]").astype("datetime64[ns]")
        axis.cdf_type = "CDF_TIME_TT2000"
    return axis


def _get_attributes(master_cdf: Driver, cdf: Driver, var: str):
    attrs = {}
    for attr in master_cdf.variable_attributes(var):
        value = master_cdf.variable_attribute_value(var, attr)
        if attr.endswith("_PTR") or attr[:-1].endswith("_PTR_"):
            if master_cdf.has_variable(value):
                value = master_cdf.values(value, is_metadata_variable=True)
                if hasattr(value, 'tolist'):
                    attrs[attr] = value.tolist()
                else:
                    attrs[attr] = value
            else:
                log.warning(
                    f"{ISTP_NOT_COMPLIANT_W}: variable {var} has {attr} attribute which points to variable {value} which does not exist")
                attrs[attr] = value
        else:
            attrs[attr] = value
    return attrs


def _get_axis(master_cdf: Driver, cdf: Driver, axis_var: str, data_var: str):
    src_cdf = cdf if cdf.has_variable(axis_var) else master_cdf if master_cdf.has_variable(axis_var) else None
    if src_cdf is not None:
        if src_cdf.is_char(axis_var):
            if 'sig_digits' in master_cdf.variable_attributes(axis_var):  # cluster CSA trick :/
                return SupportDataVariable(name=axis_var, values=np.asarray(src_cdf.values(axis_var), dtype=float),
                                           attributes=_get_attributes(master_cdf, src_cdf, axis_var),
                                           is_nrv=src_cdf.is_nrv(axis_var),
                                           cdf_type=src_cdf.cdf_type(axis_var)
                                           )
        return SupportDataVariable(name=axis_var, values=src_cdf.values(axis_var),
                                   attributes=_get_attributes(master_cdf, src_cdf, axis_var),
                                   is_nrv=src_cdf.is_nrv(axis_var),
                                   cdf_type=src_cdf.cdf_type(axis_var)
                                   )
    else:
        log.warning(
            f"{ISTP_NOT_COMPLIANT_W}: trying to load {axis_var} as support data for {data_var} but it is absent from the file")
    return None


def _get_axes(master_cdf: Driver, cdf: Driver, var: str, data_shape):
    attrs = sorted(filter(lambda attr: DEPEND_REGEX.match(attr), master_cdf.variable_attributes(var)))
    unix_time_name = master_cdf.variable_attribute_value(var, "DEPEND_TIME")
    axes = list(
        map(lambda attr: _get_axis(master_cdf, cdf, master_cdf.variable_attribute_value(var, attr), var), attrs))
    if attrs and attrs[0].upper() == "DEPEND_0" and axes[0] is not None:
        _decode_epoch(axes[0])
    if unix_time_name is not None and unix_time_name in master_cdf.variables():
        unix_time = _get_axis(master_cdf, cdf, unix_time_name, var)
        if len(unix_time) == data_shape[0] and len(axes[0].values) != data_shape[0]:
            unix_time.values = (unix_time.values * 1e9).astype('<M8[ns]')
            axes[0] = unix_time
            log.warning(
                f"{ISTP_NOT_COMPLIANT_W}: swapping DEPEND_0 with DEPEND_TIME for {var}, if you think this is a bug report it here: https://github.com/SciQLop/PyISTP/issues")
    return axes


def _get_labels(attributes) -> List[str]:
    if 'LABL_PTR_1' in attributes:
        return attributes['LABL_PTR_1']
    if 'LABLAXIS' in attributes:
        return [attributes['LABLAXIS']]


def _load_data_var(master_cdf: Driver, cdf: Driver, var: str) -> DataVariable or None:
    values = lambda: cdf.values(var)
    shape = cdf.shape(var)
    axes = _get_axes(master_cdf, cdf, var, shape)
    attributes = _get_attributes(master_cdf, cdf, var)
    cdf_type = cdf.cdf_type(var)
    labels = _get_labels(attributes)
    if len(axes) == 0:
        log.warning(f"{ISTP_NOT_COMPLIANT_W}: {var} was marked as data variable but it has 0 support variable")
        return None
    if None in axes or axes[0].values.shape[0] != shape[0]:
        return None
    return DataVariable(name=var, values=values, attributes=attributes, axes=axes, cdf_type=cdf_type, labels=labels)


class ISTPLoaderImpl:
    cdf: Optional[Driver] = None

    def __init__(self, file=None, buffer=None, master_file=None, master_buffer=None):
        if file is not None:
            log.debug(f"Loading {file}")
        self.cdf = _driver_factory(file or buffer)
        if master_file or master_buffer:
            self.master_cdf = _driver_factory(master_file or master_buffer)
        else:
            self.master_cdf = self.cdf
        self.data_variables = []
        self._update_data_vars_lis()

    def attributes(self):
        return self.master_cdf.attributes()

    def attribute(self, key):
        return self.master_cdf.attribute(key)

    def _update_data_vars_lis(self):
        if self.master_cdf:
            self.data_variables = []
            for var in self.master_cdf.variables():
                var_attrs = self.master_cdf.variable_attributes(var)
                # search for the VAR_TYPE attribute, regardless of its case
                var_type_attr = next((a for a in var_attrs if a.upper() == 'VAR_TYPE'), None)
                var_type = self.master_cdf.variable_attribute_value(var, var_type_attr) if var_type_attr else None
                param_type = (self.master_cdf.variable_attribute_value(var,
                                                                       'PARAMETER_TYPE') or "").lower()  # another cluster CSA crap
                if (var_type == 'data' or param_type == 'data') and not self.master_cdf.is_char(var):
                    self.data_variables.append(var)
            if len(self.data_variables) == 0:
                log.warning(f"{ISTP_NOT_COMPLIANT_W}: No data variable found, this is suspicious")

    def data_variable(self, var_name) -> DataVariable:
        return _load_data_var(self.master_cdf, self.cdf, var_name)

    def support_data_variable(self, var_name) -> Optional[SupportDataVariable]:
        return _get_axis(self.master_cdf, self.cdf, var_name, var_name)
