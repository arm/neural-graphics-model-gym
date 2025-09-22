# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import struct

import numpy as np

DXGI_FORMAT_R16G16B16A16_FLOAT = 10
DXGI_FORMAT_R11G11B10_FLOAT = 26
DXGI_FORMAT_R8G8B8A8_SNORM = 31
DXGI_FORMAT_R16G16_FLOAT = 34
DXGI_FORMAT_R32_FLOAT = 41
DXGI_FORMAT_R8G8_UNORM = 49
DXGI_FORMAT_R8_UNORM = 61


def encode_floating_point(x, mantissa_bits, exponent_bits, exponent_bias):
    """
    Encodes a numpy array of values into floating point values with a non-standard number of
    mantissa/exponent bits.
    Returns a numpy array of uint32s, one per encoded value.
    """
    # sanitize domain: no negatives, NaNs->0, -inf->0, +inf stays +inf for saturation:
    x = np.nan_to_num(x, nan=0.0, posinf=np.inf, neginf=0.0)
    x = np.maximum(x, 0.0)

    max_exponent_raw = (1 << exponent_bits) - 1
    max_mantissa_raw = (1 << mantissa_bits) - 1
    min_exponent = 1 - exponent_bias
    max_exponent = max_exponent_raw - 1 - exponent_bias

    out = np.zeros_like(x, dtype=np.uint32)
    nonzero = x > 0

    # Handle normals first:
    # e = floor(log2(x)), exponent field = e + exponent_bias
    e = np.floor(np.log2(np.where(nonzero, x, 1.0))).astype(np.int32)
    exponent_field = e + exponent_bias

    # normalized where 1 <= exponent_field <= max_exponent_raw-1
    norm = nonzero & (exponent_field >= 1) & (exponent_field <= max_exponent_raw - 1)
    if np.any(norm):
        xn = x[norm] / np.power(2.0, e[norm].astype(np.float32))  # in [1,2)
        mantissa = np.rint((xn - 1.0) * (1 << mantissa_bits)).astype(np.int32)
        mantissa = np.clip(mantissa, 0, max_mantissa_raw)
        out[norm] = (
            exponent_field[norm].astype(np.uint32) << mantissa_bits
        ) | mantissa.astype(np.uint32)

    # subnormals where exponent_field <= 0  (use exponent=0, scaled mantissa)
    sub = nonzero & (exponent_field <= 0)
    if np.any(sub):
        # value = mantissa / 2^mantissa_bits * 2^(min_exponent) =>
        # mantissa = round(x * 2^(mantissa_bits - min_exponent))
        mantissa = np.rint(x[sub] * np.exp2(mantissa_bits - min_exponent)).astype(
            np.int32
        )
        mantissa = np.clip(mantissa, 0, max_mantissa_raw)
        out[sub] = mantissa.astype(np.uint32)  # exponent field = 0

    # overflow (>= max representable): saturate to max finite (exp=max_exp-1, mant=all 1s)
    over = x >= (2.0 ** (max_exponent) * (2.0 - 2.0 ** (-mantissa_bits)))
    if np.any(over):
        out[over] = (
            ((max_exponent_raw - 1) << mantissa_bits) | max_mantissa_raw
        ).astype(np.uint32)

    return out


def encode_r11g11b10(x):
    """
    Encodes an array of floating point values into packed R11G11B10 floating point values.
    Takes a numpy array with the first dimension interpreted as RGB channels,
    i.e. tensor.shape must be (3, ...).
    Returns a numpy array of uint32 with a first dimension removed.
    Each element will be the packed R11G11B10 value.
    """
    if x.shape[0] != 3:
        raise ValueError(
            f"Invalid input shape. First dimension must be 3 (RGB), but got: {x.shape[0]}"
        )

    r11 = encode_floating_point(
        x[0, ...], mantissa_bits=6, exponent_bits=5, exponent_bias=15
    )
    g11 = encode_floating_point(
        x[1, ...], mantissa_bits=6, exponent_bits=5, exponent_bias=15
    )
    b10 = encode_floating_point(
        x[2, ...], mantissa_bits=5, exponent_bits=5, exponent_bias=15
    )

    packed = (r11 | (g11 << 11) | (b10 << 22)).astype(np.uint32)

    return packed


def decode_floating_point(
    encoded, mantissa_bits, exponent_bits, exponent_bias, out_type
):
    """
    Decodes floating point values with a non-standard number of mantissa/exponent bits.
    encoded: numpy array of unsigned integers, one per encoded value.
    Returns a numpy array of the given floating-point type.
    """

    if not np.issubdtype(encoded.dtype, np.unsignedinteger):
        raise TypeError("Input array must be of unsigned integer type")
    if not np.issubdtype(out_type, np.floating):
        raise TypeError("Output array must be of floating-point type")

    exponent_mask = (1 << exponent_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1

    mantissa = encoded & mantissa_mask
    exponent = (encoded >> mantissa_bits) & exponent_mask

    out = np.zeros_like(encoded, dtype=out_type)

    # subnormal (exponent == 0, mantissa != 0)
    sub = (exponent == 0) & (mantissa != 0)
    if np.any(sub):
        out[sub] = (
            mantissa[sub].astype(out_type)
            / (1 << mantissa_bits)
            * np.exp2(1 - exponent_bias)
        )

    # normal (0 < exponent < max)
    norm = (exponent > 0) & (exponent < exponent_mask)
    if np.any(norm):
        out[norm] = (
            1.0 + mantissa[norm].astype(out_type) / (1 << mantissa_bits)
        ) * np.exp2(exponent[norm].astype(np.int32) - exponent_bias)

    # overflow/Inf (exponent == exponent_mask)
    inf = exponent == exponent_mask
    if np.any(inf):
        out[inf] = np.inf

    return out


def decode_r11g11b10(words):
    """
    Decodes an array of packed R11G11B10 floating point values.
    words: numpy array of uint32 (any shape). Each element is treated as a packed R11G11B10 value.
    Returns a numpy array of float16 with a new first dimension for the RGB channels,
    i.e. shape (3, ...)
    """
    if words.dtype != np.uint32:
        raise TypeError("Input array must be of dtype uint32")

    r_bits = words & 0x7FF  # 11 bits
    g_bits = (words >> 11) & 0x7FF  # 11 bits
    b_bits = (words >> 22) & 0x3FF  # 10 bits

    r = decode_floating_point(
        r_bits, mantissa_bits=6, exponent_bits=5, exponent_bias=15, out_type=np.float16
    )
    g = decode_floating_point(
        g_bits, mantissa_bits=6, exponent_bits=5, exponent_bias=15, out_type=np.float16
    )
    b = decode_floating_point(
        b_bits, mantissa_bits=5, exponent_bits=5, exponent_bias=15, out_type=np.float16
    )

    return np.stack([r, g, b], axis=0)


def read_dds(path):
    """
    Reads a .dds file into a 3D numpy array (in CHW layout).
    The dtype of the array depends on the DXGI format in the file.
    """
    with open(path, "rb") as f:
        if f.read(4) != b"DDS ":
            raise ValueError("Not a valid DDS file")

        dds_header = f.read(124)  # DDS_HEADER struct
        height = struct.unpack_from("<I", dds_header, 8)[0]
        width = struct.unpack_from("<I", dds_header, 12)[0]
        four_cc = dds_header[80:84]

        if four_cc != b"DX10":
            raise ValueError(f"Unsupported DDS format: {four_cc.decode('utf-8')}")

        dx10_header = f.read(20)  # DDS_HEADER_DXT10 struct
        dxgi_format = struct.unpack_from("<I", dx10_header, 0)[0]

        if dxgi_format == DXGI_FORMAT_R32_FLOAT:
            result = np.fromfile(f, dtype=np.float32, count=height * width)
            result = result.reshape(1, height, width)
        elif dxgi_format == DXGI_FORMAT_R11G11B10_FLOAT:
            # For the special case of R11G11B10, numpy doesn't have a
            # native dtype for this so we decode it ourselves.
            encoded = np.fromfile(f, dtype=np.uint32, count=height * width)
            result = decode_r11g11b10(encoded)
            result = result.reshape(3, height, width)
        else:
            raise ValueError(f"Unsupported DXGI format: {dxgi_format}")

        if f.peek(1) != b"":
            raise ValueError(
                "Unexpected data at end of file! Something is probably wrong with the format "
                "logic above."
            )

    return result


def save_dds(x, path, dxgi_format):
    # pylint: disable=too-many-branches
    """
    Saves a 3D numpy array (in CHW layout) as a DDS file with the specified DXGI format.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if len(x.shape) != 3:
        raise ValueError(
            f"Invalid tensor rank. Must be 3 dimensions, but got: {x.shape}"
        )

    channels = x.shape[0]
    height = x.shape[1]
    width = x.shape[2]

    # Only certain combinations of dtype, channels and DXGI format are supported
    # (and many don't even make sense). Validate this and at the same time do any
    # conversions to get the payload that we'll write to the file.
    match (x.dtype, channels):
        case (np.float16, 4):
            if dxgi_format == DXGI_FORMAT_R16G16B16A16_FLOAT:
                payload = np.transpose(x, (1, 2, 0))  # Input is CHW, need HWC for DDS
            else:
                raise ValueError(
                    f"Invalid DXGI format for {x.dtype} and {channels} channels: {dxgi_format}"
                )
        case (np.float16, 3):
            if dxgi_format == DXGI_FORMAT_R11G11B10_FLOAT:
                # For the special case of R11G11B10, numpy doesn't have a native dtype for
                # this so we encode it ourselves.
                payload = encode_r11g11b10(x)
            else:
                raise ValueError(
                    f"Invalid DXGI format for {x.dtype} and {channels} channels: {dxgi_format}"
                )
        case (np.float16, 2):
            if dxgi_format == DXGI_FORMAT_R16G16_FLOAT:
                payload = np.transpose(x, (1, 2, 0))  # Input is CHW, need HWC for DDS
            else:
                raise ValueError(
                    f"Invalid DXGI format for {x.dtype} and {channels} channels: {dxgi_format}"
                )
        case (np.float32, 1):
            if dxgi_format == DXGI_FORMAT_R32_FLOAT:
                payload = np.transpose(x, (1, 2, 0))  # Input is CHW, need HWC for DDS
            else:
                raise ValueError(
                    f"Invalid DXGI format for {x.dtype} and {channels} channels: {dxgi_format}"
                )
        case (np.uint8, 2):
            if dxgi_format == DXGI_FORMAT_R8G8_UNORM:
                payload = np.transpose(x, (1, 2, 0))  # Input is CHW, need HWC for DDS
            else:
                raise ValueError(
                    f"Invalid DXGI format for {x.dtype} and {channels} channels: {dxgi_format}"
                )
        case (np.uint8, 1):
            if dxgi_format == DXGI_FORMAT_R8_UNORM:
                payload = np.transpose(x, (1, 2, 0))  # Input is CHW, need HWC for DDS
            else:
                raise ValueError(
                    f"Invalid DXGI format for {x.dtype} and {channels} channels: {dxgi_format}"
                )
        case (np.int8, 4):
            if dxgi_format == DXGI_FORMAT_R8G8B8A8_SNORM:
                payload = np.transpose(x, (1, 2, 0))  # Input is CHW, need HWC for DDS
            else:
                raise ValueError(
                    f"Invalid DXGI format for {x.dtype} and {channels} channels: {dxgi_format}"
                )
        case _:
            raise ValueError(
                f"Unsupported dtype and channels: {x.dtype} and {channels}"
            )

    # Write the file
    with open(path, "wb") as f:
        f.write(b"DDS ")

        # ---- DDS_HEADER struct (124 bytes) ----
        f.write(struct.pack("<I", 124))  # size of DDS_HEADER struct
        f.write(
            struct.pack(
                "<IIIIII",
                0x00001007,  # header_flags (DDSD_CAPS|DDSD_HEIGHT|DDSD_WIDTH|DDSD_PIXELFORMAT)
                height,
                width,
                0,  # pitch_or_linear_size (not used in our case)
                0,  # depth (0 for 2D)
                1,  # mipMapCount
            )
        )
        f.write(
            struct.pack("<11I", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )  # Reserved (zeroes)

        # DDS_PIXELFORMAT struct (32 bytes)
        f.write(struct.pack("<I", 32))  # size of DDS_PIXELFORMAT struct
        f.write(struct.pack("<I", 0x00000004))  # flags (DDPF_FOURCC)
        f.write(b"DX10")  # fourCC = 'DX10'
        f.write(struct.pack("<5I", 0, 0, 0, 0, 0))  # Remaining fields unused

        # Remainder of DDS_HEADER struct
        f.write(struct.pack("<I", 0x1000))  # dwCaps (DDSCAPS_TEXTURE)
        f.write(struct.pack("<4I", 0, 0, 0, 0))  # Remaining fields unused

        # ---- DDS_HEADER_DXT10 struct (20 bytes) ----
        f.write(
            struct.pack(
                "<IIIII",
                dxgi_format,  # dxgiFormat
                3,  # resourceDimension (D3D10_RESOURCE_DIMENSION_TEXTURE2D)
                0,  # miscFlag
                1,  # arraySize
                0,  # miscFlags2 (alpha mode)
            )
        )

        # ---- Payload ----
        payload.tofile(f)
