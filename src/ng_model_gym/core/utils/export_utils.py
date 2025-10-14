# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torchao
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.arm.vgf.compile_spec import VgfCompileSpec
from executorch.backends.arm.vgf.partitioner import VgfPartitioner
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.quantize_io_pass import extract_io_quant_params
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


def _get_quantizable_input_indices(
    exported_program: EdgeProgramManager,
) -> Sequence[int]:
    """Finds input indices that are quantized in the graph."""
    graph = exported_program.graph_module.graph
    user_inputs = exported_program.graph_signature.user_inputs

    inputs_to_quantization = []

    for input_index, user_input in enumerate(user_inputs):
        placeholders = [
            n for n in graph.nodes if n.op == "placeholder" and n.name == user_input
        ]
        if not placeholders:
            continue
        target_placeholder = placeholders[0]

        if len(target_placeholder.users) != 1:
            raise ValueError(f"Input {input_index} has more than one users")

        quantize = next(iter(target_placeholder.users))
        if (
            quantize.target
            != exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        ):
            continue

        inputs_to_quantization.append(input_index)

    return inputs_to_quantization


def _get_quantizable_output_indices(
    exported_program: EdgeProgramManager,
) -> Sequence[int]:
    """Finds output indices that are quantized in the graph."""
    graph = exported_program.graph_module.graph
    outputs = [n for n in graph.nodes if n.op == "output"]
    if len(outputs) != 1:
        raise NotImplementedError("Only 1 output node is supported.")

    outputs_to_quantization = []

    user_outputs = list(outputs[0].args[0])
    for output_index, user_output in enumerate(user_outputs):
        if (
            user_output.target
            != exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        ):
            continue

        outputs_to_quantization.append(output_index)

    return outputs_to_quantization


def _calculate_snorm_params(details: tuple[float, int]) -> tuple[float, float]:
    """Returns new signed normalized scale and zero point"""
    scale, zero_point = details

    # SNORM constants
    qmin = -127.0  # SNORM negates -128 value
    qmax = 127.0
    min_val, max_val = -1.0, 1.0
    sn_scale = (max_val - min_val) / (qmax - qmin)
    sn_zero_point = qmin - (min_val / sn_scale)  # Note: evals to 0.0 as symmetric

    new_scale = scale / sn_scale
    new_zero_point = zero_point * sn_scale - sn_zero_point

    return new_scale, new_zero_point


def _write_input_output_scales(io_quant_params: Dict[str, Any], filepath: Path) -> None:
    """
    Update metadata structure with extracted input/output quantization parameters.
    SNORM scale and zero points are also calculated.

    io_quant_params is a nested dict from extract_io_quant_params(), e.g.:
        {"inputs": {"x": ...}, "outputs": {"y": ...}}
    """
    metadata = {}
    for input_output in io_quant_params.keys():
        metadata[input_output] = {}
        for key, val in io_quant_params[input_output].items():
            scale = float(val["scale"])
            zero_point = int(val["zero_point"])
            # qmin/qmax not needed here since _calculate_snorm_params handles it internally
            snorm_scale, snorm_zero_point = _calculate_snorm_params((scale, zero_point))
            metadata[input_output][key] = {
                "SINT": {"scale": scale, "zero_point": zero_point},
                "SNORM": {"scale": snorm_scale, "zero_point": snorm_zero_point},
            }

    _update_metadata_file(filepath, metadata)


def _check_cuda():
    """Check if CUDA GPU is available"""
    if not torch.cuda.is_available():
        err_msg = (
            "CUDA build of PyTorch with GPU is currently required for model exporting."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


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


def _export_module_to_vgf(
    params: ConfigModel,
    ng_model: BaseNGModel,
    model_forward_input: Tuple[Any, ...],
    export_type: ExportType,
    metadata_path,
):
    """Use ExecuTorch to perform exporting to a VGF file."""

    tosa_spec = (
        ExportSpec.TOSA_FP if export_type == ExportType.FP32 else ExportSpec.TOSA_INT
    )
    compile_spec = VgfCompileSpec(TosaSpecification.create_from_string(tosa_spec))

    dynamic_input_spec = (
        ng_model.define_dynamic_export_model_input()
        if params.output.export.dynamic_shape
        else None
    )

    neural_network = ng_model.get_neural_network()

    if export_type == ExportType.QAT_INT8:
        torchao.quantization.pt2e.move_exported_model_to_eval(neural_network)
    else:
        neural_network.eval()

    if export_type == ExportType.PTQ_INT8:
        # Perform post-training quantization before writing model to output file
        neural_network = torch.export.export_for_training(
            neural_network, model_forward_input, strict=True
        ).module()
        quantizer = TOSAQuantizer(TosaSpecification.create_from_string(tosa_spec))
        quantizer.set_global(get_symmetric_quantization_config(is_qat=False))
        neural_network = prepare_pt2e(neural_network, quantizer)

    if (
        not params.output.export.dynamic_shape
        and params.output.export.vgf_static_input_shape
    ):
        model_forward_input = tuple(
            torch.rand(n, c, h, w)
            for n, c, h, w in params.output.export.vgf_static_input_shape
        )
        logger.info("Tracing and exporting model with config-provided static shape")

    # Do a forward pass of the model (unpack tuple from trace).
    neural_network(*model_forward_input)

    # Get the quantized module ready for export.
    if export_type != ExportType.FP32:
        neural_network = convert_pt2e(neural_network)

        # Need to use a static shaped graph for extracting input/output scales and zero points.
        to_edge = to_edge_transform_and_lower(
            torch.export.export(neural_network, args=model_forward_input, strict=True),
            partitioner=[VgfPartitioner(compile_spec)],
        )
        edge_program = to_edge._edge_programs["forward"]
        input_idxs = _get_quantizable_input_indices(edge_program)
        output_idxs = _get_quantizable_output_indices(edge_program)
        raw_io_quant_params = extract_io_quant_params(
            to_edge, input_idxs=input_idxs, output_idxs=output_idxs
        )

        Path(params.output.export.vgf_output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Exporting IO scale and zero points to: {params.output.export.vgf_output_dir}"
        )
        _write_input_output_scales(raw_io_quant_params, metadata_path)

    aten_dialect = torch.export.export(
        neural_network,
        args=model_forward_input,
        strict=True,
        dynamic_shapes=dynamic_input_spec,
    )

    # VGF partition and export.
    with _loader_context("VGF", str(params.output.export.vgf_output_dir)):
        vgf_partitioner = VgfPartitioner(
            compile_spec.dump_intermediate_artifacts_to(
                str(params.output.export.vgf_output_dir)
            )
        )
        to_edge_transform_and_lower(aten_dialect, partitioner=[vgf_partitioner])


def _update_metadata_file(metadata_path: Path, object_to_add: Dict):
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

    is_dynamic_input = params.output.export.dynamic_shape
    static_input_shape = params.output.export.vgf_static_input_shape

    if not is_dynamic_input and static_input_shape is None:
        raise ValueError(
            "Dynamic export shape is set to false but no static VGF"
            " input shape is specified in config"
        )
    if is_dynamic_input and static_input_shape is not None:
        logger.warning(
            "Dynamic export is enabled and provided static input shape will be ignored"
        )

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

    model_key = get_model_key(params.model.name, params.model.version)

    metadata_path = (
        Path(params.output.export.vgf_output_dir)
        / f"{model_key}-{export_type.name}-metadata.json"
    )

    _update_metadata_file(metadata_path, model.get_additional_constants())

    _export_module_to_vgf(
        params, model, model_forward_input, export_type, metadata_path
    )

    logger.info(
        f"Export complete, exported to: {str(params.output.export.vgf_output_dir)}\n"
    )
