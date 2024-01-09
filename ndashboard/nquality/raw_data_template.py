import xarray as xr
import numpy as np


class SessionNeuralData(object):
    def __init__(
            self,
            da_presentation: xr.DataArray,
            presentation_dim='presentation',
            neuroid_dim='neuroid',
            stimulus_id_coord='stimulus_id',
            timestamp_coord='unix_timestamp',
    ):

        """
        A class which wraps the raw spike count data, which is supplied as an xr.DataArray with dims:
            value: (presentation_dim, neuroid_dim)
        It has a mandatory coord:
            stimulus_id_coord: (presentation_dim)

        :param da_presentation:  xr.DataArray of spike counts, with dimensions (presentation_dim, neuroid_dim).
        """
        dim_set = {
            presentation_dim,
            neuroid_dim,
        }

        mandatory_coords = {
            stimulus_id_coord,
            timestamp_coord,
        }

        assert isinstance(da_presentation, xr.DataArray), f"da_presentation:{da_presentation}, Required type: xr.DataArray"

        assert set(da_presentation.dims) == dim_set, f"da_presentation.dims:{da_presentation.dims}, Required dimensions: {dim_set}"
        for coord in mandatory_coords:
            # print(coord, da_presentation.coords, coord in da_presentation.coords)
            assert coord in da_presentation.coords, f"da_presentation.coords:{da_presentation.coords}, Required coordinates: {mandatory_coords}"

        assert set(da_presentation[stimulus_id_coord].dims) == {presentation_dim}, f"da_presentation[{stimulus_id_coord}].dims:{da_presentation[stimulus_id_coord].dims}, Required dimensions: {presentation_dim}"

        # Perform basic checks
        nan_entries = np.isnan(da_presentation).sum()
        negative_entries = (da_presentation < 0).sum()
        noninteger_entries = (np.mod(da_presentation, 1) != 0).sum()

        if nan_entries > 0:
            raise ValueError(f"da_presentation contains {nan_entries} NaN entries")
        if negative_entries > 0:
            raise ValueError(f"da_presentation contains {negative_entries} negative entries")
        if noninteger_entries > 0:
            raise ValueError(f"da_presentation contains {noninteger_entries} non-integer entries")

        # Rename dims to standard names
        presentation_dim_standard = 'presentation'
        neuroid_dim_standard = 'neuroid'
        stimulus_id_standard = 'stimulus_id'
        timestamp_coord_standard = 'unix_timestamp'

        da_presentation = da_presentation.rename(
            {
                presentation_dim: presentation_dim_standard,
                neuroid_dim: neuroid_dim_standard,
                stimulus_id_coord: stimulus_id_standard,
                timestamp_coord: timestamp_coord_standard,
            }
        )

        self.da_presentation = da_presentation
        self.presentation_dim = presentation_dim_standard
        self.neuroid_dim = neuroid_dim_standard
        self.stimulus_id_coord = stimulus_id_standard
        self.timestamp_coord = timestamp_coord_standard
        self.timestamp_start = float(np.min(da_presentation[self.timestamp_coord].values))

    @property
    def stimulus_id_to_da(self):
        if not hasattr(self, '_stimulus_id_to_da'):
            self._stimulus_id_to_da = {}
            for stimulus_id, da in self.da_presentation.groupby(self.stimulus_id_coord):
                self._stimulus_id_to_da[stimulus_id] = da.transpose(self.presentation_dim, self.neuroid_dim)

        return self._stimulus_id_to_da


if __name__ == '__main__':
    path = '/Users/mjl/PycharmProjects/ndashboard/experiment/dicarlo_rsvp/da_date_example1.nc'
    da = xr.load_dataarray(path)
    da = da.dropna('neuroid')
    print(SessionNeuralData(da_presentation=da).stimulus_id_to_da)
