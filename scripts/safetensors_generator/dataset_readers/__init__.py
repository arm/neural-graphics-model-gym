# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""
Set of methods which collect data for the later production of Safetensors files.
"""

from scripts.safetensors_generator.dataset_readers.crop_safetensors import (
    CropSafetensorsNSS,
)
from scripts.safetensors_generator.dataset_readers.nfru_exr_dataset_reader import (
    NFRUEXRDatasetReader,
)
from scripts.safetensors_generator.dataset_readers.nss_exr_dataset_reader import (
    NSSEXRDatasetReader,
)
from scripts.safetensors_generator.dataset_readers.safetensors_feature_iterator import (
    SafetensorsFeatureIterator,
)

Dataset_Readers = {
    "cropper": CropSafetensorsNSS,
    "NFRUv2_2": NFRUEXRDatasetReader,
    "NSSv1_0_1": NSSEXRDatasetReader,
}
