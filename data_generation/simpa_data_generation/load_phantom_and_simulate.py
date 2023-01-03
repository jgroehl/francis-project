# SPDX-FileCopyrightText: 2022 International Photoacoustic Standardisation Consortium (IPASC)
# SPDX-FileCopyrightText: 2022 Janek Gröhl
# SPDX-License-Identifier: MIT

import simpa as sp
from simpa import Tags
import numpy as np
from utils.settings import generate_base_settings
from ipasc_simpa_kwave_adapter import IpascSimpaKWaveAdapter
from scipy.ndimage import zoom

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()

settings = generate_base_settings(path_manager)

# extract information on geometry and spacing for the purposes of volume creation and device definition
dim_x_mm = settings[Tags.DIM_VOLUME_X_MM]
dim_y_mm = settings[Tags.DIM_VOLUME_Y_MM]
dim_z_mm = settings[Tags.DIM_VOLUME_Z_MM]
spacing = settings[Tags.SPACING_MM]

# ###################################################################################
# VOLUME CREATION
#
# Using the SIMPA volume creation module with the segmentation based volume creator
#
# ###################################################################################

sizes = np.round(np.asarray([dim_x_mm, dim_y_mm, dim_z_mm]) / spacing).astype(int)
label_volume = np.zeros(sizes)

label_mask = np.load(r"C:\Users\grohl01\Downloads\data\ground_truth.npz")["gt"]

input_spacing = 0.078125
label_mask = np.round(zoom(label_mask, input_spacing/spacing, order=0)).astype(int)

label_volume[:, int(sizes[1]/2), :] = label_mask

def segmentation_class_mapping():
    ret_dict = dict()
    ret_dict[1] = sp.TISSUE_LIBRARY.blood(oxygenation=0.5)
    ret_dict[0] = (sp.MolecularCompositionGenerator()
                   .append(sp.MOLECULE_LIBRARY.water(1.0))
                   .get_molecular_composition(sp.SegmentationClasses.WATER))
    return ret_dict

settings.set_volume_creation_settings({
    Tags.INPUT_SEGMENTATION_VOLUME: label_volume,
    Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(),

})
acoustic_settings = settings.get_acoustic_settings()
# For this simulation: Use the created absorption map as the input initial pressure
acoustic_settings[Tags.DATA_FIELD] = Tags.DATA_FIELD_ABSORPTION_PER_CM

pipeline = [
    sp.SegmentationBasedVolumeCreationAdapter(settings),
    IpascSimpaKWaveAdapter(settings),
    sp.TimeReversalAdapter(settings),
    sp.FieldOfViewCropping(settings, "FieldOfViewCropping")
]

# Create a device with
device = sp.PhotoacousticDevice(device_position_mm=np.array([dim_x_mm/2,
                                                             dim_y_mm/2,
                                                             0]),
                                field_of_view_extent_mm=np.asarray([-20, 20, 0, 0, 0, 40]))
device.set_detection_geometry(sp.LinearArrayDetectionGeometry(device_position_mm=device.device_position_mm,
                                                              pitch_mm=0.3,
                                                              number_detector_elements=128,
                                                              sampling_frequency_mhz=50,
                                                              field_of_view_extent_mm=np.asarray([-20, 20, 0, 0, 0, 40])))
device.add_illumination_geometry(sp.SlitIlluminationGeometry(slit_vector_mm=[100, 0, 0]))

sp.simulate(simulation_pipeline=pipeline,
            settings=settings,
            digital_device_twin=device)

sp.visualise_data(settings=settings,
                  path_manager=path_manager,
                  wavelength=800,
                  show_absorption=True,
                  show_time_series_data=True,
                  show_reconstructed_data=True)