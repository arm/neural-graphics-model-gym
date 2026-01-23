# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import string
from pathlib import Path
from typing import Dict, List

from huggingface_hub.errors import RepositoryNotFoundError

from ng_model_gym.core.model.repos.base_model_server import ModelRepository
from ng_model_gym.core.model.repos.huggingface_model_server import (
    HuggingfaceModelServer,
)

MODEL_SERVERS = [HuggingfaceModelServer()]


def parse_model_identifier(model_identifier: str) -> tuple[str, str]:
    """Return repo and filename from [@]<repo_name>/<filename> identifier"""
    model_identifier = model_identifier.strip()
    if model_identifier.startswith("@"):
        model_identifier = model_identifier[len("@") :]

    if model_identifier.startswith(string.punctuation):
        raise ValueError(
            "Symbols not allowed for model identifier unless it starts with '@'"
        )

    if "/" not in model_identifier:
        raise ValueError(
            f"Model identifier must be <repo_name>/<filename>, not {model_identifier}"
        )

    repo_name, file_name = model_identifier.split("/", 1)
    if not repo_name or not file_name:
        raise ValueError(
            f"Model identifier must be <repo_name>/<filename>, not {model_identifier}"
        )

    return repo_name, file_name


def list_pretrained_models() -> Dict[str, List[ModelRepository]]:
    """
    List downloadable pretrained model files from model servers.

    Returns:
        Dictionary of model server names and a list of its repositories containing models
    """
    found: Dict[str, List[ModelRepository]] = {}

    for server in MODEL_SERVERS:
        repos = server.list_repositories()  # Fail fast as exceptions not caught
        found[server.server_display_name] = repos

    return found


def download_pretrained_model(model_name: str, destination: Path | None = None) -> Path:
    """
    Download pretrained weight from model servers.

    Args:
        model_name: model identifier in the form <repo_name>/<filename>.
            Also supports @<repo_name>/<filename> identifier for clarity over a file path.
        destination: directory save path for the downloaded file. If None, uses a cache dir

    Returns:
        Path to the downloaded model file
    """
    user_repo_name, user_file_name = parse_model_identifier(model_name)

    if destination is not None:
        destination = Path(destination).expanduser()
        if destination.exists() and not destination.is_dir():
            raise ValueError(f"Destination must be a directory: {destination}")
        destination.mkdir(parents=True, exist_ok=True)

    found_repo = False
    available_repos: set[str] = set()
    available_models: set[str] = set()

    for server in MODEL_SERVERS:
        remote_repo_list: List[ModelRepository] = server.list_repositories()

        # Check all remote repos for a match against user's specified repo
        for remote_repo in remote_repo_list:
            available_repos.add(remote_repo.repository.name)

            if remote_repo.repository.name != user_repo_name:
                continue

            found_repo = True
            for model in remote_repo.models:
                available_models.add(model.file_name)
                if user_file_name in available_models:
                    return server.download(remote_repo.repository, model, destination)

    if not found_repo:
        available = ", ".join(sorted(available_repos)) or "(none)"
        raise RepositoryNotFoundError(
            f"Unknown repository '{user_repo_name}'. Available repositories: {available}"
        )

    available = ", ".join(sorted(available_models)) or "(none)"
    raise ValueError(
        f"Model '{user_file_name}' not found in repository '{user_repo_name}'. "
        f"\nAvailable models: \n{available}"
    )
