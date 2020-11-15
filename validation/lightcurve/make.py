#!/usr/bin/env python
"""Run Gammapy validation: CTA 1DC"""
import logging
import warnings
from pathlib import Path
import click
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.datasets import SpectrumDataset
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.maps import MapAxis
from gammapy.estimators import LightCurveEstimator
from gammapy.makers import (
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
)
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from regions import CircleSkyRegion
from astropy.coordinates import Angle



log = logging.getLogger(__name__)

@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]),)
@click.option("--show-warnings", is_flag=True, help="Show warnings?")
def cli(log_level, show_warnings):
    logging.basicConfig(level=log_level)
    log.setLevel(level=log_level)
    if not show_warnings:
        warnings.simplefilter("ignore")




data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")

t0 = Time("2006-07-29T20:30")
duration = 10 * u.min
n_time_bins = 35
times = t0 + np.arange(n_time_bins) * duration
time_intervals = [
    Time([tstart, tstop]) for tstart, tstop in zip(times[:-1], times[1:])
]
print(time_intervals[0].mjd)

#Target geometry definition
e_reco = MapAxis.from_energy_bounds(0.4, 20, 10, "TeV").edges
e_true = MapAxis.from_energy_bounds(0.1, 40, 20, "TeV").edges
target_position = SkyCoord(
    329.71693826 * u.deg, -30.2255890 * u.deg, frame="icrs"
)
on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


#data reduction makers
dataset_maker = SpectrumDatasetMaker(
    containment_correction=True, selection=["counts", "aeff", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker()
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)


def select_data():
    target_position = SkyCoord(
    329.71693826 * u.deg, -30.2255890 * u.deg, frame="icrs"
    )
    selection = dict(
    type="sky_circle",
    frame="icrs",
    lon=target_position.ra,
    lat=target_position.dec,
    radius=2 * u.deg,
    )
    obs_ids = data_store.obs_table.select_observations(selection)["OBS_ID"]
    observations = data_store.get_observations(obs_ids)
    return observations

def filter_observations(observations):
    short_observations = observations.select_time(time_intervals)
    # check that observations have been filtered
    print(
    f"Number of observations after time filtering: {len(short_observations)}\n"
    )
    print(short_observations[1].gti)
    return  short_observations




def create_datasets(short_observations):
    datasets = []

    dataset_empty = SpectrumDataset.create(
    e_reco=e_reco, e_true=e_true, region=on_region
    )
    for obs in short_observations:
        dataset = dataset_maker.run(dataset_empty.copy(), obs)
        dataset_on_off = bkg_maker.run(dataset, obs)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, obs)
        datasets.append(dataset_on_off)
    return datasets





def define_model():
    spectral_model = PowerLawSpectralModel(
        index=3.4,
        amplitude=2e-11 * u.Unit("1 / (cm2 s TeV)"),
        reference=1 * u.TeV,
    )
    spectral_model.parameters["index"].frozen = False
    sky_model = SkyModel(
           spatial_model=None, spectral_model=spectral_model, name="pks2155"
                    )
    return sky_model

def assign_model_todatasets(datasets, sky_model):
    for dataset in datasets:
        dataset.models = sky_model
    return datasets



def extract_lightcurve(datasets):
    lc_maker_1d = LightCurveEstimator(
    energy_range=[0.7, 20] * u.TeV, source="pks2155"
    )

    lc_1d = lc_maker_1d.run(datasets)
    lc_1d.plot(marker="o")









@cli.command("run-analyses", help="Run Gammapy validation: Light curve")
def run_analyses():
    print("Run")

    datasets = create_datasets(filter_observations(select_data()))
    sky_model = define_model()

    for dataset in datasets:
        dataset.models = sky_model

    lc_maker_1d = LightCurveEstimator(
        energy_edges=[0.7, 20] * u.TeV,
        source="pks2155",
        time_intervals=time_intervals
    )

    #lc_1d = lc_maker_1d.run(data)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()


