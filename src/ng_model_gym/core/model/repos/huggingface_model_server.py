# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import shutil
from fnmatch import fnmatch
from pathlib import Path
from typing import List

from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from ng_model_gym.core.model.repos.base_model_server import (
    BaseModelServer,
    ModelMetadata,
    ModelRepository,
    RepositoryMetadata,
)


class HuggingfaceModelServer(BaseModelServer):
    """HuggingFace model server"""

    server_display_name = "HuggingFace"

    def __init__(self):
        # Add new HF model repositories here or update hash
        self.repos = [
            RepositoryMetadata(
                namespace="Arm",
                name="neural-super-sampling",
                url="https://huggingface.co/Arm/neural-super-sampling",
                revision="2e9b606acd9fa25071825a12f0764f1c3bef9480",
            )
        ]

    def list_repositories(
        self,
    ) -> list[ModelRepository]:
        """
        Returns a dictionary of the model_server, repository and its models
        """

        api = HfApi()
        model_repositories: List[ModelRepository] = []

        for repo in self.repos:
            try:
                repo_files = api.list_repo_files(
                    repo_id=f"{repo.namespace}/{repo.name}",
                    repo_type="model",
                    revision=repo.revision,
                )
            except RepositoryNotFoundError as exc:
                raise RepositoryNotFoundError(
                    f"{HuggingfaceModelServer.server_display_name} "
                    f"Unable to find repository ({repo.name}@{repo.revision}): {exc}"
                ) from exc

            models: List[ModelMetadata] = []
            for file_path in repo_files:
                filename = Path(file_path).name
                if not fnmatch(filename, "*.pt"):
                    continue

                models.append(
                    ModelMetadata(
                        file_name=filename,
                        relative_path=file_path,
                    )
                )

            model_repositories.append(
                ModelRepository(
                    repository=repo, models=sorted(models, key=lambda m: m.file_name)
                )
            )

        return model_repositories

    def download(
        self,
        repository: RepositoryMetadata,
        model: ModelMetadata,
        destination: Path | None,
    ) -> Path:
        """Downloads HF model"""
        repo_id = f"{repository.namespace}/{repository.name}"
        try:
            cache_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=repository.revision,
                filename=model.relative_path,
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"{self.server_display_name}: failed to download {model.file_name} "
                f"from {repository.name}@{repository.revision}: {exc}"
            ) from exc

        cached_path = Path(cache_path)

        # If user hasn't specified download destination, keep in cache
        if destination is None:
            return cached_path

        # Copy to destination
        model_path = Path(destination) / model.relative_path
        model_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(cached_path, model_path)

        return model_path
