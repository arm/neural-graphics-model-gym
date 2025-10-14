# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Tuple

import torch
import torchao
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.arm.vgf.compile_spec import VgfCompileSpec
from executorch.backends.arm.vgf.partitioner import VgfPartitioner
from executorch.exir import to_edge_transform_and_lower
from rich.console import Console
from torch.utils._pytree import tree_map
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from ng_model_gym.core.data.dataloader import DataLoaderMode, get_dataloader
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.base_ng_model_wrapper import BaseNGModelWrapper
from ng_model_gym.core.model.model import get_model_key
from ng_model_gym.core.model.model_tracer import model_tracer
from ng_model_gym.core.utils.checkpoint_utils import load_checkpoint
from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.core.utils.general_utils import is_invoked_cli
from ng_model_gym.core.utils.types import ExportSpec, ExportType, TrainEvalMode

logger = logging.getLogger(__name__)


def _calculate_snorm_params(details: tuple[float, int]) -> tuple[float, float]:
    """Returns new signed normalized scale and zero point"""
    scale, zero_point = details

    # SNORM Constants
    qmin = -127.0  # SNORM negates -128 value
    qmax = 127.0
    min_val, max_val = -1.0, 1.0
    sn_scale = (max_val - min_val) / (qmax - qmin)
    sn_zero_point = qmin - (min_val / sn_scale)  # Note: evals to 0.0 as symmetric

    # Calculate New Scale and Zero Pt.
    new_scale = scale / sn_scale
    new_zero_point = zero_point * sn_scale - sn_zero_point

    return new_scale, new_zero_point


def _get_scale_zero_point(observer_module: torch.nn.Module):
    """
    Safely extract scale/zero_point, handling both Python floats and Tensors.
    Returns (scale, zero_point) as Python floats or lists (if they’re Tensor).
    """
    scale = None
    zero_point = None

    if hasattr(observer_module, "scale"):
        s = observer_module.scale
        if isinstance(s, torch.Tensor):
            s = (
                s.detach().cpu().numpy().tolist().pop()
            )  # convert to Python float or list
        scale = s

    if hasattr(observer_module, "zero_point"):
        zp = observer_module.zero_point
        if isinstance(zp, torch.Tensor):
            zp = zp.detach().cpu().numpy().tolist().pop()
        zero_point = zp

    return scale, zero_point


def _flatten_args(args):
    """
    Recursively flatten nested tuples/lists from node.args.
    Returns a list of items (which can be fx.Nodes or other data).
    """
    items = []

    def _flatten(item):
        if isinstance(item, (tuple, list)):
            for x in item:
                _flatten(x)
        else:
            items.append(item)

    _flatten(args)
    return items


def _find_single_observer_for_node(node: torch.fx.Node):
    """
    Looks at all the 'user' nodes that consume `node`.
    If exactly one user is an observer, returns that module name.
    Otherwise returns None.
    """
    for user_node in node.users:
        if (
            user_node.op == "call_module"
            and "activation_post_process" in user_node.target
        ):
            return user_node.target
    return None


def _extract_input_output_scales(fx_model: torch.fx.GraphModule) -> dict[str, float]:
    """
    Given an FX GraphModule (ExecuTorch model):
    1. Finds all input placeholders (model inputs).
    2. Finds the final output node and flattens its arguments (model outputs).
    3. Tries to locate associated observer modules for each input/output.
    4. Extracts the scale/zero_point for each such observer.
    5. Returns a dict suitable for JSON serialization.
    """

    # Turn named_modules into a dict for easy lookup
    name_to_module = dict(fx_model.named_modules())

    results = {"inputs": {}, "outputs": {}}

    # Gather input placeholders
    input_nodes = [n for n in fx_model.graph.nodes if n.op == "placeholder"]
    for n in input_nodes:
        # n.name is typically "foo" or "bar"
        placeholder_name = n.name
        observer_name = _find_single_observer_for_node(n)

        scale, zero_point = None, None
        if observer_name is not None:
            observer_mod = name_to_module[observer_name]
            scale, zero_point = _get_scale_zero_point(observer_mod)

        snorm_scale, snorm_zero_point = _calculate_snorm_params((scale, zero_point))
        results["inputs"][placeholder_name] = {
            "SINT": {"scale": scale, "zero_point": zero_point},
            "SNORM": {"scale": snorm_scale, "zero_point": snorm_zero_point},
        }

    # Find the final output node
    # Typically only one output node in a standard FX graph
    output_node = [n for n in fx_model.graph.nodes if n.op == "output"][0]

    # The output node usually has node.args which might be a tuple-of-tuples
    out_args = _flatten_args(output_node.args)

    # For each node feeding the output, attempt to read observer info
    for out_arg in out_args:
        if not isinstance(out_arg, torch.fx.Node):
            # It's a raw value or something else so skip
            continue

        # This node might be e.g. activation_post_process_16
        out_name = out_arg.name
        scale, zero_point = None, None

        # If it's a call_module referencing the observer, we can read directly
        # Or if it's a ReLU node, we might look for the observer that feeds it
        if out_arg.op == "call_module" and "activation_post_process" in out_arg.target:
            observer_mod = name_to_module[out_arg.target]
            scale, zero_point = _get_scale_zero_point(observer_mod)
        else:
            # Possibly the real observer is behind this node
            # so we might do a quick search among its inputs
            observer_name = _find_single_observer_for_node(out_arg)
            if observer_name is not None:
                observer_mod = name_to_module[observer_name]
                scale, zero_point = _get_scale_zero_point(observer_mod)

        snorm_scale, snorm_zero_point = _calculate_snorm_params((scale, zero_point))
        results["outputs"][out_name] = {
            "SINT": {"scale": scale, "zero_point": zero_point},
            "SNORM": {"scale": snorm_scale, "zero_point": snorm_zero_point},
        }

    return results


def _write_input_output_scales(filepath: Path, fx_model: torch.fx.GraphModule) -> None:
    zp_and_scales = _extract_input_output_scales(fx_model)
    logging.info(f"Found input/output `zero_point`'s and `scale`'s: {zp_and_scales}")
    _update_metadata_file(filepath, zp_and_scales)


def _check_cuda():
    """Check if CUDA GPU is available"""
    if not torch.cuda.is_available():
        err_msg = (
            "CUDA build of PyTorch with GPU is currently required for model exporting."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


def _input_shape_constraints(is_dynamic: bool):
    """Provide dynamic shape information"""

    batch_size = torch.export.Dim("batch")  # Batch size can be anything
    # Input width/height must be a multiple of 8 (because of the resizing layers)
    input_height_over_8 = torch.export.Dim("input_height_over_8", min=1)
    input_width_over_8 = torch.export.Dim("input_width_over_8", min=1)
    input_height = 8 * input_height_over_8
    input_width = 8 * input_width_over_8

    # NCHW - tuple contents match forward tensor input
    dynamic_shape = (
        ({0: batch_size, 2: input_height, 3: input_width},) if is_dynamic else None
    )
    return dynamic_shape


@contextmanager
def _loader_context(label: str, dump_dir: str):
    """Context manager shows spinner if program was invoked from the CLI or logs if from library"""
    cli = is_invoked_cli()
    if cli:
        console = Console()
        message = f"[bold green]Exporting to {label}"
        with console.status(message, spinner="dots"):
            yield
    else:
        message = f"Exporting to {label}"
        logger.info(message)
        yield
        logger.info(f"{label} export complete: {dump_dir}")


def _vgf_partition_and_lower(dialect, spec: str, dump_dir: str):
    """Use the VGF partitioner to export to a VGF file.

    We also export to TOSA as an artifact, but this is not used anywhere.
    This will be deprecated in the future.
    """
    # TOSA partition and export.
    with _loader_context("TOSA", dump_dir):
        tosa_partitioner = TOSAPartitioner(
            TosaCompileSpec(
                tosa_spec=TosaSpecification.create_from_string(spec)
            ).dump_intermediate_artifacts_to(dump_dir)
        )
        to_edge_transform_and_lower(dialect, partitioner=[tosa_partitioner])

    # VGF partition and export.
    with _loader_context("VGF", dump_dir):
        vgf_partitioner = VgfPartitioner(
            VgfCompileSpec(
                tosa_spec=TosaSpecification.create_from_string(spec)
            ).dump_intermediate_artifacts_to(dump_dir)
        )
        to_edge_transform_and_lower(dialect, partitioner=[vgf_partitioner])


def _export_module_to_vgf(
    params,
    neural_network,
    model_forward_input: Tuple[Any, ...],
    export_type,
    metadata_path,
):
    """Use ExecuTorch to perform exporting to a VGF file."""

    tosa_spec = (
        ExportSpec.TOSA_FP if export_type == ExportType.FP32 else ExportSpec.TOSA_INT
    )

    if export_type != ExportType.QAT_INT8:
        neural_network.eval()

    if export_type == ExportType.PTQ_INT8:
        # Perform post-training quantization before writing model to output file
        neural_network = torch.export.export_for_training(
            neural_network, model_forward_input
        ).module()
        quantizer = TOSAQuantizer(TosaSpecification.create_from_string(tosa_spec))
        quantizer.set_global(get_symmetric_quantization_config(is_qat=False))
        neural_network = prepare_pt2e(neural_network, quantizer)
    elif export_type == ExportType.QAT_INT8:
        torchao.quantization.pt2e.move_exported_model_to_eval(neural_network)

    # Do a forward pass of the model (unpack tuple from trace).
    neural_network(*model_forward_input)

    # Extract input/output scales and zero points for quantized models and dump to JSON.
    # Currently only for QAT INT8 export.
    if export_type == ExportType.QAT_INT8:
        Path(params.output.export.vgf_output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Exporting input/output scale and zero points to {params.output.export.vgf_output_dir}"
        )
        _write_input_output_scales(
            metadata_path,
            neural_network,
        )

    # Get the quantized module ready for export.
    if export_type != ExportType.FP32:
        neural_network = convert_pt2e(neural_network)

    aten_dialect = torch.export.export(
        neural_network,
        args=model_forward_input,
        strict=True,
        dynamic_shapes=_input_shape_constraints(params.output.export.dynamic_shape),
    )

    _vgf_partition_and_lower(
        aten_dialect, tosa_spec, str(params.output.export.vgf_output_dir)
    )


def _update_metadata_file(metadata_path: Path, object_to_add: dict):
    """Update the metadata file with additional information.

    Creates the file first if it doesn't exist.
    """
    metadata = {}
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata.update(object_to_add)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def executorch_vgf_export(
    params: ConfigModel, export_type: ExportType, model_path: Path
):
    """Export PyTorch model to a VGF file, based on the params & export type provided.

    This uses ExecuTorch and goes through the following steps:
        * create model from checkpoint
        * add quantization nodes if loading QAT checkpoint
        * do PTQ if needed
        * do a forward inference to trace the model
        * export to Aten dialect
        * partition and lower to a VGF file.
    """

    _check_cuda()

    if export_type == ExportType.QAT_INT8:
        params.model_train_eval_mode = TrainEvalMode.QAT_INT8  #

        if params.dataset.path.train is None:
            raise ValueError(
                "Config error: No path specified for the train dataset."
                "This is required for exporting a QAT model to a VGF file."
            )

        loader_mode = DataLoaderMode.TRAIN
    else:
        params.model_train_eval_mode = TrainEvalMode.FP32

        if params.dataset.path.test is None:
            raise ValueError(
                "Config error: No path specified for the test dataset."
                "This is required for exporting an FP32 model to a VGF file."
            )

        loader_mode = DataLoaderMode.TEST

    # Resolve weights and load the model.
    model = load_checkpoint(model_path, params, torch.device("cpu"))

    dataloader = get_dataloader(
        params,
        num_workers=params.dataset.num_workers,
        prefetch_factor=params.dataset.prefetch_factor,
        loader_mode=loader_mode,
        trace_mode=export_type != ExportType.QAT_INT8,
    )

    # Get sample data to trace graph.
    preprocess_trace_input = next(iter(dataloader))[0]
    model_forward_input = model_tracer(model, preprocess_trace_input)

    # Move model forward inputs to CPU
    to_cpu = lambda x: x.to("cpu") if isinstance(x, torch.Tensor) else x
    # tree_map is an internal torch util to traverse containers with tensors
    model_forward_input = tree_map(to_cpu, model_forward_input)

    if isinstance(model, BaseNGModelWrapper):
        model = model.get_ng_model()

    elif not isinstance(model, BaseNGModel):
        raise ValueError(f"model type: {type(model)} , is not valid")

    neural_network = model.get_neural_network()

    model_key = get_model_key(params.model.name, params.model.version)

    metadata_path = (
        Path(params.output.export.vgf_output_dir)
        / f"{model_key}-{export_type}-metadata.json"
    )

    _update_metadata_file(metadata_path, model.get_additional_constants())

    _export_module_to_vgf(
        params, neural_network, model_forward_input, export_type, metadata_path
    )

    logger.info(f"Export complete!: {str(params.output.export.vgf_output_dir)}\n")
