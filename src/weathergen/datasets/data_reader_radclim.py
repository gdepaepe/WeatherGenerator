# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray


from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
    str_to_timedelta,
)

_logger = logging.getLogger(__name__)


class DataReaderRadClim(DataReaderTimestep):
    "Wrapper for RadClim datasets"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct data reader for RadClim dataset

        Parameters
        ----------
        tw_handler:
            time window handler
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream

        Returns
        -------
        None
        """

        ds: Dataset = xr.open_zarr(filename)

        # If there is no overlap with the time range, the dataset will be empty
        if tw_handler.t_start >= ds.time.values[-1] or tw_handler.t_end <= ds.time.values[0]:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        # Slice with start and end time
        ds = ds.sel(time=slice(tw_handler.t_start, tw_handler.t_end))

        # Downsampling with frequency
        if "frequency" in stream_info:
            frequency = str_to_timedelta(stream_info["frequency"])
            ds = ds.resample(time=frequency).nearest()
        else:
            frequency = xr.infer_freq(ds.coords["time"])
        if "subsampling_rate" in stream_info:
            name = stream_info["name"]
            _logger.warning(
                f"subsampling_rate specified for anemoi dataset for stream {name}. "
                + "Use frequency instead."
            )
        _logger.warning(f"FREQUENCY: {frequency}")
            
        # Set attributes
        self.properties = {"stream_id": 0,} # WHICH NBR TO CHOOSE???
        period = pd.to_timedelta(frequency).to_timedelta64()
        self.variable = list(ds.data_vars.keys())[0] # RadClim dataset has only 1 variable
        self.latitudes = ds.lat.values.astype(np.float32)
        self.longitudes = ds.lon.values.astype(np.float32)
        self.datetimes = ds.time.values
        data_start_time = self.datetimes[0]
        data_end_time = self.datetimes[-1]
        super().__init__(
            tw_handler,
            stream_info,
            data_start_time,
            data_end_time,
            period,
        )
        
        # If there is no overlap with the time range, no need to keep the dataset.

        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            self.init_empty()
            return
        else:
            self.ds = ds[self.variable]
            self.len = len(ds.time)
       
        # select/filter requested source and target channels
        self.source_idx, self.source_channels = self.select_channels("source")
        self.target_idx, self.target_channels = self.select_channels("target")

        # geoinfo is empty
        self.geoinfo_channels = []
        self.geoinfo_idx = []

        ds_name = stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: geoinfo channels: {self.geoinfo_channels}")

        self.mean = np.array([self.ds.attrs["mean dBZ (Marshal Palmer with a=200, b=1.6)"]], dtype=np.float32)
        self.stdev = np.array([self.ds.attrs["std dBZ (Marshal Palmer with a=200, b=1.6)"]], dtype=np.float32)

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for window (for either source or target, through public interface)

        Parameters
        ----------
        idx : int
            Index of temporal window
        channels_idx : np.array
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        assert t_idxs[0] >= 0, "index must be non-negative"
        didx_start = t_idxs[0]
        # End is inclusive
        didx_end = t_idxs[-1] + 1

        # extract number of time steps 
        try:
            data = self.ds.values[didx_start:didx_end].astype(np.float32)
        except MissingDateError as e:
            _logger.debug(f"Date not present in anemoi dataset: {str(e)}. Skipping.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # flatten
        data = data.reshape((-1, 1))
        lat = self.latitudes.reshape((-1))
        lon = self.longitudes.reshape((-1))

        # convert rates to dBZ via Marshal Palmer
        # return 0 for any rain lighter than ~0.037mm/h and clip at 60dBZ ~205mm/h
        data = 10 * np.log10(200 * np.power(data, 1.6) + 1e-8)
        data = np.clip(data, 0.0, 60.0)
        data = np.nan_to_num(data, nan=0.0)

        # construct lat/lon coords
        latlon = np.concatenate(
            [
                np.expand_dims(lat, 0),
                np.expand_dims(lon, 0),
            ],
            axis=0,
        ).transpose()
        # repeat latlon len(t_idxs) times
        coords = np.vstack((latlon,) * len(t_idxs))

        # empty geoinfos
        geoinfos = np.zeros((len(data), 0), dtype=data.dtype)

        # date time matching #data points of data
        # assuming a fixed frequency for the dataset
        datetimes = np.repeat(self.datetimes[didx_start:didx_end], len(data) // len(t_idxs))

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd

    def select_channels(self, ch_type: str) -> Tuple[NDArray[np.int64], List]:
        """
        Select source or target channels

        Parameters
        ----------
        ds0 :
            raw anemoi dataset with available channels
        ch_type :
            "source" or "target", i.e channel type to select

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes

        """

        channels = self.stream_info.get(ch_type)
        channels_exclude = self.stream_info.get(ch_type + "_exclude", [])
        # sanity check
        is_empty = len(channels) == 0 if channels is not None else False
        if is_empty:
            _logger.warning(f"No channel for {self.stream_info["name"]} for {ch_type}.")
        if len(channels) > 1 or (len(channels) == 1 and channels[0] != self.variable):
            _logger.warning(f"Stream {self.stream_info["name"]} only has channel {self.variable}.")

        if self.variable in channels and self.variable not in channels_exclude:
            chs_idx = np.array([0])
            channels = [self.variable]
        else:
            chs_idx = np.empty(shape=[0])
            channels = []
            
        return chs_idx, channels
