# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F


def lerp_weight(x: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Linear interpolation weight from a sample at x to xs.
    Returns the linear interpolation weight of a "query point" at coordinate `x`
    with respect to a "sample" at coordinate `xs`.
    The integer coordinates `x` are at pixel centers.
    The floating point coordinates `xs` are at pixel edges.
    (OpenGL convention).
    Args:
        x: "Query" point position.
        xs: "Sample" position.
    Returns:
        - 1 when x = xs.
        - 0 when |x - xs| > 1.
    """
    dx = x - xs
    abs_dx = torch.abs(dx)
    return torch.maximum(
        torch.tensor(1.0, dtype=x.dtype, device=x.device) - abs_dx,
        torch.tensor(0.0, dtype=x.dtype, device=x.device),
    )


def resampler(
    image: torch.Tensor,
    warp: torch.Tensor,
    clamp_to_edge: bool = False,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """Resample image values at dense warp coordinates."""

    padding = "border" if clamp_to_edge else "zeros"
    grid = warp + 0.5
    grid = grid.permute(0, 2, 3, 1)
    _, _, height, width = image.shape
    u, v = torch.split(grid, split_size_or_sections=1, dim=-1)
    v = ((v / height) * 2.0) - 1.0
    u = ((u / width) * 2.0) - 1.0
    grid = torch.concat([u, v], dim=-1)

    return F.grid_sample(
        image,
        grid,
        mode=interpolation,
        padding_mode=padding,
        align_corners=False,
    )


def dense_image_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    clamp_to_edge: bool = True,
) -> torch.Tensor:
    """Image warping using per-pixel flow vectors."""

    if len(image.shape) != 4:
        raise ValueError("Image must be 4D Tensor.")
    if len(flow.shape) != 4:
        raise ValueError("Flow must be 4D Tensor.")

    _, _, height, width = image.shape
    flow_y, flow_x = torch.split(flow, split_size_or_sections=1, dim=1)
    flow_xy = torch.concat([flow_x, flow_y], dim=1).to(dtype=image.dtype)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, height, device=flow.device),
        torch.arange(0, width, device=flow.device),
        indexing="ij",
    )
    stacked_grid = torch.stack([grid_x, grid_y], dim=0).to(dtype=image.dtype)
    query_points_on_grid = stacked_grid.unsqueeze(0) - flow_xy

    return resampler(
        image,
        query_points_on_grid,
        interpolation=interpolation,
        clamp_to_edge=clamp_to_edge,
    )


def bilinear_oob_zero(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Image warping using per-pixel flow vectors with zero OOB padding."""

    return dense_image_warp(
        image,
        flow,
        interpolation="bilinear",
        clamp_to_edge=False,
    )


def backward_warp_nearest(
    image: torch.Tensor, flow: torch.Tensor, oob_zero: bool = False
) -> torch.Tensor:
    """Image warping technique."""
    flow = torch.round(flow)

    batch_size, ch, h, w = (
        image.size(0),
        image.size(1),
        image.size(2),
        image.size(3),
    )

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    stacked_grid = (torch.stack([grid_y, grid_x], dim=0)).to(flow.dtype).to(flow.device)
    batched_grid = torch.unsqueeze(stacked_grid, dim=0)

    query_points = batched_grid - flow

    # Bound
    gif, gjf = torch.split(query_points, split_size_or_sections=[1, 1], dim=1)
    gi0c = (torch.clamp(gif, 0, float(h - 1))).to(torch.int32)
    gj0c = (torch.clamp(gjf, 0, float(w - 1))).to(torch.int32)

    # Flatten Grid
    # We convert image to NHWC
    # TODO should be able to do in NCHW format
    image = image.permute(0, 2, 3, 1)
    flattened_grid = torch.reshape(image, [batch_size * h * w, ch])
    batch_offsets = torch.reshape(
        torch.arange(batch_size) * h * w, [batch_size, 1, 1, 1]
    ).to(flow.device)

    # We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using gather_nd.
    def gather(y_coords, x_coords):
        linear_coordinates = batch_offsets + y_coords * w + x_coords
        gathered_values = flattened_grid[linear_coordinates]
        return torch.reshape(gathered_values, [batch_size, h, w, ch])

    result = gather(gi0c, gj0c)

    # Convert back to NCHW.
    result = result.permute(0, 3, 1, 2)
    if oob_zero:
        h = float(h)
        w = float(w)
        out_of_bounds = (
            (
                (gif < h).to(torch.float32)
                * (gif >= 0).to(torch.float32)
                * (gjf < w).to(torch.float32)
                * (gjf >= 0).to(torch.float32)
            )
            .to(result.dtype)
            .to(result.device)
        )
        result *= out_of_bounds

    return result


def sample_bilinear(
    grid: torch.Tensor,
    query_points: torch.Tensor,
    h_upper_limit: int = None,
    w_upper_limit: int = None,
) -> torch.Tensor:
    """Given a `grid` and `query_points` will bilinearly sample the queries from the `grid`"""

    # Grid Dimensions
    batch_size, h, w, ch = (
        grid.size(0),
        grid.size(1),
        grid.size(2),
        grid.size(3),
    )

    # Queries - split into II, JJ
    gif, gjf = torch.split(query_points, split_size_or_sections=[1, 1], dim=-1)

    # Compute linear interpolation weights without clamping.
    gi0 = torch.floor(gif - 0.5)
    gj0 = torch.floor(gjf - 0.5)
    gi1 = gi0 + 1
    gj1 = gj0 + 1

    wi0 = lerp_weight(gi0 + 0.5, gif)
    wi1 = lerp_weight(gi1 + 0.5, gif)
    wj0 = lerp_weight(gj0 + 0.5, gjf)
    wj1 = lerp_weight(gj1 + 0.5, gjf)

    w_00 = wi0 * wj0
    w_01 = wi0 * wj1
    w_10 = wi1 * wj0
    w_11 = wi1 * wj1

    _h = h
    _w = w
    if h_upper_limit is not None:
        _h = h_upper_limit
    if w_upper_limit is not None:
        _w = w_upper_limit

    # But clip when indexing into `grid`.
    gi0c = (torch.clamp(gi0, 0, float(_h - 1))).to(torch.int32)
    gj0c = (torch.clamp(gj0, 0, float(_w - 1))).to(torch.int32)

    gi1c = (torch.clamp(gi0 + 1, 0, float(_h - 1))).to(torch.int32)
    gj1c = (torch.clamp(gj0 + 1, 0, float(_w - 1))).to(torch.int32)

    # Flatten Grid
    flattened_grid = torch.reshape(grid, [batch_size * h * w, ch])
    batch_offsets = torch.reshape(
        torch.arange(batch_size) * h * w, [batch_size, 1, 1, 1]
    ).to(grid.device)

    # We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back.
    def gather(y_coords, x_coords):
        linear_coordinates = batch_offsets + y_coords * w + x_coords
        gathered_values = flattened_grid[linear_coordinates]
        return torch.reshape(gathered_values, [batch_size, h, w, ch])

    # Grab 4 corners
    grid_val_00 = gather(gi0c, gj0c)
    grid_val_01 = gather(gi0c, gj1c)
    grid_val_10 = gather(gi1c, gj0c)
    grid_val_11 = gather(gi1c, gj1c)

    sliced_grid = (
        torch.multiply(w_00, grid_val_00)
        + torch.multiply(w_01, grid_val_01)
        + torch.multiply(w_10, grid_val_10)
        + torch.multiply(w_11, grid_val_11)
    )

    return sliced_grid


def catmull_rom_params(xy: torch.tensor) -> torch.tensor:
    """Catmull Rom Spline Formula"""
    tc = torch.floor(xy - 0.5) + 0.5
    f = xy - tc
    f2 = f * f
    f3 = f2 * f

    w0 = f2 - 0.5 * (f3 + f)
    w1 = 1.5 * f3 - 2.5 * f2 + 1
    w3 = 0.5 * (f3 - f2)
    w2 = 1 - w0 - w1 - w3

    ww0 = w0
    ww1 = w1 + w2
    ww2 = w3

    s0 = tc - 1
    s1 = tc + w2 / ww1
    s2 = tc + 2

    return s0, s1, s2, ww0, ww1, ww2


def catmull_rom_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
    h_upper_limit: int = None,
    w_upper_limit: int = None,
    strict_mode: bool = False,
) -> torch.Tensor:
    """Catmull Rom Resampling"""

    # This assertion (len(image.shape) == 4) to protect
    # from passing something out of range to size()
    if len(image.shape) != 4:
        raise ValueError("Image must be 4D Tensor.")
    if len(flow.shape) != 4:
        raise ValueError("Flow must be 4D Tensor.")
    batch_size, channels, height, width = (
        image.size(0),
        image.size(1),
        image.size(2),
        image.size(3),
    )

    # We convert image grid to NHWC
    # TODO should be able to do in NCHW format
    image = torch.permute(image, [0, 2, 3, 1])
    flow = torch.permute(flow, [0, 2, 3, 1])

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width), torch.arange(height), indexing="xy"
    )
    stacked_grid = (torch.stack([grid_y, grid_x], dim=2)).to(flow.dtype).to(flow.device)
    batched_grid = torch.unsqueeze(stacked_grid, dim=0) + 0.5

    query_points_on_grid = batched_grid - flow

    s0, s1, s2, w0, w1, w2 = catmull_rom_params(query_points_on_grid)

    sample_uv0 = torch.stack([s1[..., 0], s0[..., 1]], dim=-1)
    sample_uv1 = torch.stack([s0[..., 0], s1[..., 1]], dim=-1)
    sample_uv2 = torch.stack([s1[..., 0], s1[..., 1]], dim=-1)
    sample_uv3 = torch.stack([s2[..., 0], s1[..., 1]], dim=-1)
    sample_uv4 = torch.stack([s1[..., 0], s2[..., 1]], dim=-1)

    sample_w0 = torch.unsqueeze(w1[..., 0] * w0[..., 1], dim=-1)
    sample_w1 = torch.unsqueeze(w0[..., 0] * w1[..., 1], dim=-1)
    sample_w2 = torch.unsqueeze(w1[..., 0] * w1[..., 1], dim=-1)
    sample_w3 = torch.unsqueeze(w2[..., 0] * w1[..., 1], dim=-1)
    sample_w4 = torch.unsqueeze(w1[..., 0] * w2[..., 1], dim=-1)

    out_color_0 = sample_bilinear(image, sample_uv0, h_upper_limit, w_upper_limit)
    out_color_1 = sample_bilinear(image, sample_uv1, h_upper_limit, w_upper_limit)
    out_color_2 = sample_bilinear(image, sample_uv2, h_upper_limit, w_upper_limit)
    out_color_3 = sample_bilinear(image, sample_uv3, h_upper_limit, w_upper_limit)
    out_color_4 = sample_bilinear(image, sample_uv4, h_upper_limit, w_upper_limit)

    # Out-of-bounds
    if strict_mode:
        _h = height
        _w = width
        if h_upper_limit is not None:
            _h = h_upper_limit
        if w_upper_limit is not None:
            _w = w_upper_limit

        def weight_oob(w, uv):
            gif, gjf = torch.split(uv, split_size_or_sections=[1, 1], dim=-1)
            oob_i = (
                (gif >= 0.0).to(torch.float32)
                * (gif <= float(_h - 1)).to(torch.float32)
            ).to(torch.float32)
            oob_j = (
                (gjf >= 0.0).to(torch.float32)
                * (gjf <= float(_w - 1)).to(torch.float32)
            ).to(torch.float32)
            oob_region = oob_j * oob_i
            return w * oob_region

        sample_w0 = weight_oob(sample_w0, sample_uv0)
        sample_w1 = weight_oob(sample_w1, sample_uv1)
        sample_w2 = weight_oob(sample_w2, sample_uv2)
        sample_w3 = weight_oob(sample_w3, sample_uv3)
        sample_w4 = weight_oob(sample_w4, sample_uv4)

    corner_weights = sample_w0 + sample_w1 + sample_w2 + sample_w3 + sample_w4
    final_multiplier = torch.reciprocal(corner_weights + 1e-7)  # Add small epsilon.

    out_color = (
        out_color_0 * sample_w0
        + out_color_1 * sample_w1
        + out_color_2 * sample_w2
        + out_color_3 * sample_w3
        + out_color_4 * sample_w4
    ) * final_multiplier

    out_color = torch.maximum(out_color, torch.tensor(0))
    out_color = torch.reshape(out_color, [batch_size, height, width, channels])

    out_color = torch.permute(out_color, [0, 3, 1, 2])

    return out_color
