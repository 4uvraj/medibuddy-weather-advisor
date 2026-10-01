from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from policy_engine.exceptions import PolicyConfigurationError
from policy_engine.models import PolicyDocument, PolicyManifest, PolicySet


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise yaml.YAMLError("YAML mapping keys must be hashable") from error
            if duplicate:
                raise yaml.YAMLError(f"Duplicate YAML mapping key: {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _load_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as policy_file:
            return yaml.load(policy_file, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise PolicyConfigurationError(f"Malformed YAML in {path.name}: {error}") from error
    except OSError as error:
        raise PolicyConfigurationError(f"Could not read {path}: {error}") from error


def load_policy_set(manifest_path: str | Path) -> PolicySet:
    """Load and validate the manifest and its external SOP document."""
    manifest_file = Path(manifest_path)
    try:
        manifest = PolicyManifest.model_validate(_load_yaml(manifest_file))
        policy_file = (manifest_file.resolve().parent / manifest.policies_file).resolve()
        if policy_file.parent != manifest_file.resolve().parent:
            raise PolicyConfigurationError("The policies_file must stay in the manifest directory")
        document = PolicyDocument.model_validate(_load_yaml(policy_file))
        if manifest.policy_set_version != document.policy_set_version:
            raise PolicyConfigurationError(
                "Policy-set version mismatch: manifest declares "
                f"{manifest.policy_set_version}, policy document declares {document.policy_set_version}"
            )
        return PolicySet(
            manifest=manifest,
            policy_set_version=document.policy_set_version,
            policies=document.sops,
        )
    except ValidationError as error:
        raise PolicyConfigurationError(f"Invalid policy configuration: {error}") from error