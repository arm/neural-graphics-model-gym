# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
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


@app.command(name="list-models")
def list_models_cli():
    """List pretrained models available from configured repositories."""
    from ng_model_gym.core.model.repos.remote_model_manager import (
        list_pretrained_models,
    )

    console = Console()
    try:
        available = list_pretrained_models()  # Dict[str, List[ModelRepository]]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        console.print(f"[red]Failed to list models: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not available:
        console.print("[yellow]No downloadable models found.[/yellow]")
        return

    for server_name, repos in available.items():
        if not repos:
            continue

        console.print(f"\n[bold]{server_name}[/bold]")

        for repo in repos:
            repo_id = repo.repository.name
            revision = repo.repository.revision
            repo_url = repo.repository.url

            revision_suffix = f" @ {revision[:7]}" if revision else ""

            repo_line = f"  [bold]{repo_id}[/bold]{revision_suffix}"
            repo_line = f"{repo_line} ([dim]{repo_url}[/dim])"
            console.print(repo_line)

            if not repo.models:
                console.print("    [yellow](no .pt models found)[/yellow]")
                continue

            for model in repo.models:
                console.print(f"    * {model.file_name}")

        console.print()

    console.print(
        "[dim]See [bold]ng-model-gym download --help[/bold] for downloading models[/dim]\n"
    )


@app.command(name="download")
def download_cli(
    model_name: Annotated[
        str,
        typer.Argument(
            help=(
                "Model filename to download (e.g. nss_v0.1.0_fp32.pt) or a "
                "model reference (e.g. hf://Org/Repo/filename.pt)"
            )
        ),
    ],
    destination: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Directory to save the downloaded model",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
):
    """Download pretrained model checkpoint from configured repositories"""
    from ng_model_gym.core.model.repos.remote_model_manager import (
        download_pretrained_model,
    )

    console = Console()
    try:
        downloaded_path = download_pretrained_model(
            model_name=model_name,
            destination=destination,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        console.print(f"[red]Download failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold bright_green]Downloaded[/bold bright_green] "
        f"{model_name} to [bold]{downloaded_path}[/bold]"
    )


@app.command(name="train")
def train_cli(
    resume: Annotated[
        Optional[Path],
        typer.Option(
            "--resume",
            "-r",
            help="Path to checkpoint file or directory to resume training from",
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=True,
        ),
    ] = None,
    finetune: Annotated[
        Optional[Path],
        typer.Option(
            "--finetune",
            "-f",
            help="Path to local pre-trained model weights (.pt) or remote model"
            " identifier @<repo_name>/<file_name> to fine-tune from",
        ),
    ] = None,
    evaluate: Annotated[
        bool, typer.Option(help="Run evaluation metrics on trained model")
    ] = True,
):
    """Perform training"""
    from ng_model_gym import do_evaluate, do_training

    params = cli_state["params"]

    if evaluate and params.dataset.path.test is None:
        raise ValueError(
            "Config error: Evaluation is specified but no test dataset path is provided"
        )
    if resume and finetune:
        raise typer.BadParameter(
            "Cannot specify both --resume and --finetune",
            param_hint="--resume/--finetune",
        )

    model_path = do_training(
        params,
        TrainEvalMode.FP32,
        cli_state["profiler"],
        finetune_model_path=finetune,
        resume_model_path=resume,
    )

    if evaluate:
        do_evaluate(params, model_path, model_type=TrainEvalMode.FP32)


@app.command(name="qat")
def qat_cli(
    resume: Annotated[
        Optional[Path],
        typer.Option(
            "--resume",
            "-r",
            help="Path to checkpoint file or directory to resume QAT training from",
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=True,
        ),
    ] = None,
    finetune: Annotated[
        Optional[Path],
        typer.Option(
            "--finetune",
            "-f",
            help="Path to pre-trained model weights (.pt) to fine-tune from",
        ),
    ] = None,
    evaluate: Annotated[
        bool, typer.Option(help="Run evaluation metrics on trained QAT model")
    ] = True,
):
    """Perform quantization-aware training"""
    from ng_model_gym import do_evaluate, do_training

    params = cli_state["params"]

    if evaluate and params.dataset.path.test is None:
        raise ValueError(
            "Config error: Evaluation is specified but no test dataset path is provided"
        )
    if resume and finetune:
        raise typer.BadParameter(
            "Cannot specify both --resume and --finetune",
            param_hint="--resume/--finetune",
        )

    model_path = do_training(
        params,
        TrainEvalMode.QAT_INT8,
        cli_state["profiler"],
        resume_model_path=resume,
        finetune_model_path=finetune,
    )

    if evaluate:
        do_evaluate(params, model_path, model_type=TrainEvalMode.QAT_INT8)


@app.command(name="evaluate")
def eval_cli(
    model_path: Annotated[
        Path,
        typer.Option(
            help="Path to local model .pt file or remote model identifier @<repo_name>/<file_name> "
            "to evaluate",
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
            help="Path to model .pt file or remote model identifier @<repo_name>/<file_name> "
            "for VGF file export",
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
