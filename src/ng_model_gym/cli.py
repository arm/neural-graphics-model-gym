# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, TYPE_CHECKING, TypedDict

import typer
from rich.console import Console

from ng_model_gym.core.utils.types import ExportType, ProfilerType, TrainEvalMode

if TYPE_CHECKING:
    from ng_model_gym.core.utils import ConfigModel

# pylint: disable=import-outside-toplevel, line-too-long

# Create a logger using the root module name 'src', to be used throughout project.
ROOT_MODULE_NAME = "ng_model_gym"
logger = logging.getLogger(ROOT_MODULE_NAME)

# Initialise CLI app
app = typer.Typer(
    no_args_is_help=True, pretty_exceptions_enable=False, add_completion=False
)


class CLIState(TypedDict):
    """Store state from CLI"""

    params: ConfigModel
    config_path: Path
    profiler: ProfilerType


cli_state = CLIState()


@app.command(name="init")
def generate_config(
    out_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Output directory to place config file template & schema",
            show_default=str(Path.cwd()),
        ),
    ] = None
):
    """Generate configuration file & schema"""

    console = Console()
    with console.status("[bold green]Generating config…", spinner="dots") as _:
        from ng_model_gym.core.utils.config_utils import generate_config_file

    config_output_path, schema_path = generate_config_file(out_dir)

    console.print(
        f"""
    [bold magenta]Config[/bold magenta] file written to [bold bright_green]{config_output_path}[/bold bright_green]
    [bold magenta]Schema[/bold magenta] file copied to [bold bright_green]{schema_path}[/bold bright_green]

    [bold red]→[/bold red] Open the config file now and fill in the <…> placeholders.
    [bold red]→[/bold red] Refer to [bold bright_green]schema_config.json[/bold bright_green] for a complete description of available configuration fields.
    [bold red]→[/bold red] Use the [bold bright_cyan]--config-path[/bold bright_cyan] or [bold bright_cyan]-c[/bold bright_cyan] option to pass your config file when running commands.

    [dim]Run [bold]ng-model-gym --help[/bold] to explore all CLI options.[/dim]
    """
    )


@app.command(name="train")
def train_cli(
    resume: Annotated[
        bool,
        typer.Option(
            "--resume", "-r", help="Restore training from most recent checkpoint"
        ),
    ] = False,
    finetune: Annotated[
        bool,
        typer.Option(
            help="Fine-tune from provided pre-trained model (specified in config)"
        ),
    ] = False,
    evaluate: Annotated[
        bool, typer.Option(help="Run evaluation metrics on trained model")
    ] = True,
):
    """Perform training"""
    from ng_model_gym import do_evaluate, do_training

    params = cli_state["params"]

    # Fine-tuning and resuming are different, because
    # fine-tuning - we take the pretrained model from the included checkpoints;
    # resuming - we resume our training as it has been stopped, so
    # we could resume training for the fine-tuned model.
    if finetune:
        # Overrides the finetune flag set in the config files.
        params.train.finetune = True

    if resume:
        # Overrides the resume flag set in the config files.
        params.train.resume = True

    if evaluate and params.dataset.path.test is None:
        raise ValueError(
            "Config error: Evaluation is specified but no test dataset path is provided"
        )

    model_path = do_training(params, TrainEvalMode.FP32, cli_state["profiler"])

    if evaluate:
        do_evaluate(params, model_path, model_type=TrainEvalMode.FP32)


@app.command(name="qat")
def qat_cli(
    resume: Annotated[
        bool,
        typer.Option(
            "--resume", "-r", help="Restore QAT training from most recent checkpoint"
        ),
    ] = False,
    finetune: Annotated[
        bool,
        typer.Option(
            help="Fine-tune from provided pre-trained model (specified in config)"
        ),
    ] = False,
    evaluate: Annotated[
        bool, typer.Option(help="Run evaluation metrics on trained QAT model")
    ] = True,
):
    """Perform quantization-aware training"""
    from ng_model_gym import do_evaluate, do_training

    params = cli_state["params"]

    # Fine-tuning and resuming are different, because
    # fine-tuning - we take the pretrained model from the included checkpoints;
    # resuming - we resume our training as it has been stopped, so
    # we could resume training for the fine-tuned model.
    if finetune:
        # Overrides the finetune flag set in the config files.
        params.train.finetune = True

    if resume:
        # Overrides the resume flag set in the config files.
        params.train.resume = True

    if evaluate and params.dataset.path.test is None:
        raise ValueError(
            "Config error: Evaluation is specified but no test dataset path is provided"
        )

    model_path = do_training(params, TrainEvalMode.QAT_INT8, cli_state["profiler"])

    if evaluate:
        do_evaluate(params, model_path, model_type=TrainEvalMode.QAT_INT8)


@app.command(name="evaluate")
def eval_cli(
    model_path: Annotated[
        Path,
        typer.Option(
            help="Path to model .pt file", exists=True, dir_okay=False, readable=True
        ),
    ],
    model_type: Annotated[
        TrainEvalMode,
        typer.Option(
            help="The type of model to evaluate",
        ),
    ],
):
    """Run evaluation on a chosen model"""
    from ng_model_gym import do_evaluate

    params = cli_state["params"]

    do_evaluate(
        params, model_path, model_type=model_type, profile_setting=cli_state["profiler"]
    )


@app.command(name="export")
def export_cli(
    model_path: Annotated[
        Path,
        typer.Option(
            help="Path to model .pt file for VGF file export",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    export_type: Annotated[
        ExportType,
        typer.Option(
            help="Model export type. Note only QAT trained models can be exported to qat_int8",
            exists=True,
            readable=True,
        ),
    ],
):
    """Export a model to an executable format"""
    from ng_model_gym import do_export

    params = cli_state["params"]
    do_export(params, model_path, export_type)


@app.command(name="config-options")
def config_options_cli():
    """View all the configurations available"""
    from ng_model_gym import print_config_options

    print_config_options()


class AppLogLevel(str, Enum):
    """Log levels"""

    INFO = "info"
    DEBUG = "debug"
    QUIET = "quiet"


def version_callback(value: bool):
    """Get current version"""

    if value:
        from importlib.metadata import (  # pylint: disable=import-outside-toplevel
            version,
        )

        __version__ = version("ng_model_gym")

        print(f"Version: {__version__}")
        raise typer.Exit()


@app.callback(
    epilog="More options available with [COMMAND] --help" " , e.g. train --help",
    help=None,
)
# pylint: disable=unused-argument
def cli_root(
    ctx: typer.Context,
    config_path: Annotated[
        Optional[Path],
        typer.Option(
            "--config-path",
            "-c",
            help="Path to JSON config file",
            exists=True,
            dir_okay=False,
            readable=True,
            show_default=False,
        ),
    ] = None,
    log_level: Annotated[AppLogLevel, typer.Option()] = AppLogLevel.INFO,
    deterministic_cuda: Annotated[
        bool,
        typer.Option(
            help="Enable PyTorch deterministic CUDA algorithms "
            "(may impact training performance)"
        ),
    ] = False,
    profiler: Annotated[
        ProfilerType, typer.Option(help="Profile training")
    ] = ProfilerType.DISABLED,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version", help="Show the project version", callback=version_callback
        ),
    ] = None,
):
    """Flags for the root CLI command and common to all the commands"""

    if ctx.resilient_parsing:
        return

    # Skip this function if --help flag is passed to a command
    if (
        "--help" in sys.argv
        or "config-options" in sys.argv
        or "init" in sys.argv
        or "--out_dir" in sys.argv
    ):
        return

    # IMPORTANT - add commands requiring config to this list
    cmds_requiring_cfg = {"train", "qat", "evaluate", "export"}

    if ctx.invoked_subcommand not in cmds_requiring_cfg:
        return

    if config_path is None:
        raise typer.BadParameter(
            "Config file is required to run this command: --config-path or -c "
        )

    console = Console()

    with console.status("[bold green] Loading modules...", spinner="dots") as _:
        from ng_model_gym.core.utils.config_utils import load_config_file
        from ng_model_gym.core.utils.general_utils import fix_randomness
        from ng_model_gym.core.utils.logging import log_machine_info, logging_config

    # Create a globally accessible "Click" ctx to check if program was invoked from the CLI
    ctx.ensure_object(dict)
    ctx.obj["ng-model-gym-cli-active"] = True

    params = load_config_file(config_path)
    cli_state["params"] = params
    cli_state["config_path"] = config_path
    cli_state["profiler"] = profiler

    log_level = log_level.value
    if log_level == AppLogLevel.DEBUG:
        log_level = logging.DEBUG
    elif log_level == AppLogLevel.QUIET:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO

    # Setup logging
    logging_config(params, ROOT_MODULE_NAME, log_level)

    fix_randomness(params.train.seed, deterministic_cuda)

    log_machine_info()


# pylint: enable=unused-argument


def main():
    """Invoke typer CLI"""
    app()


if __name__ == "__main__":
    main()
