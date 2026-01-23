# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass()
class ModelMetadata:
    """Information about a downloadable pretrained model"""

    file_name: str
    relative_path: str


@dataclass()
class RepositoryMetadata:
    """Information about a repository"""

    namespace: str  # E.g "Arm"
    name: str  # E.g "neural-super-sampling"
    url: str  # E.g "https://huggingface.co/Arm/neural-super-sampling"
    revision: str  # repository hash


@dataclass()
class ModelRepository:
    """Repository metadata and the list of models in the repository"""

    repository: RepositoryMetadata
    models: list[ModelMetadata]


class BaseModelServer(ABC):
    """Listing and downloading from model server"""

    @property
    @abstractmethod
    def server_display_name(self) -> str:
        """Returns model server name"""
        raise NotImplementedError

    @abstractmethod
    def list_repositories(self) -> List[RepositoryMetadata]:
        """Returns available repositories"""
        raise NotImplementedError

    @abstractmethod
    def download(
        self,
        repository: RepositoryMetadata,
        model: ModelMetadata,
        destination: Path | None,
    ) -> Path:
        """
        Download model from server repositories. If destination is None, download to default
        location e.g. cache folder
        """
        raise NotImplementedError
