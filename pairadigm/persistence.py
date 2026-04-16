"""
Persistence helpers for Pairadigm objects.

Replaces the old pickle-based save/load with a structured format:
  <save_dir>/
    metadata.json   — all scalar attributes + model_names + providers
    data.parquet    — the original item DataFrame
    pairwise_df.parquet  — (if set) pairwise comparisons
    scored_df.parquet    — (if set) scored items

Old .pkl files are detected and rejected with a clear migration message.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from .core import Pairadigm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _metadata_from_obj(obj: "Pairadigm") -> Dict[str, Any]:
    """Extract JSON-serialisable metadata from a Pairadigm instance."""
    return {
        "version": "1.0",
        # Client config
        "model_names": obj.model_names,
        "providers": [c.provider for c in obj.clients],
        # Column names
        "item_id_name": obj.item_id_name,
        "text_name": obj.text_name,
        "item_id_cols": obj.item_id_cols,
        "item_text_cols": obj.item_text_cols,
        # Flags
        "paired": obj.paired,
        "annotated": obj.annotated,
        "llm_annotated": obj.llm_annotated,
        # Column lists
        "annotator_cols": obj.annotator_cols,
        "llm_annotator_cols": obj.llm_annotator_cols,
        "prior_breakdown_cols": obj.prior_breakdown_cols,
        # Concept config
        "target_concept": obj.target_concept,
        "cgcot_prompts": obj.cgcot_prompts,
        # Persistence
        "save_dir": obj.save_dir,
        # Transparency
        "column_renames": getattr(obj, "column_renames", {}),
    }


def _restore_metadata(obj: "Pairadigm", meta: Dict[str, Any]) -> None:
    """Push stored metadata back onto reconstructed Pairadigm instance."""
    obj.item_id_name      = meta.get("item_id_name")
    obj.text_name         = meta.get("text_name")
    obj.item_id_cols      = meta.get("item_id_cols")
    obj.item_text_cols    = meta.get("item_text_cols")
    obj.paired            = meta.get("paired", False)
    obj.annotated         = meta.get("annotated", False)
    obj.llm_annotated     = meta.get("llm_annotated", False)
    obj.annotator_cols    = meta.get("annotator_cols") or []
    obj.llm_annotator_cols = meta.get("llm_annotator_cols") or []
    obj.prior_breakdown_cols = meta.get("prior_breakdown_cols")
    obj.target_concept    = meta.get("target_concept")
    obj.cgcot_prompts     = meta.get("cgcot_prompts")
    obj.column_renames    = meta.get("column_renames", {})
    obj.save_dir          = meta.get("save_dir")
    obj.validation_results = None


# ---------------------------------------------------------------------------
# Public save / load
# ---------------------------------------------------------------------------

def save_pairadigm(obj: "Pairadigm", save_dir: str) -> None:
    """
    Save a Pairadigm instance to a directory using structured files.

    Saves:
      - ``metadata.json``     — scalar attributes, model names & providers
      - ``data.parquet``      — the item DataFrame (always present)
      - ``pairwise_df.parquet`` — pairwise comparison DataFrame (if present)
      - ``scored_df.parquet``   — scored item DataFrame (if present)

    Parameters
    ----------
    obj : Pairadigm
        The instance to save.
    save_dir : str
        Directory path. Created automatically if it does not exist.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Ensure object is sync'd with the current save_dir before serialising metadata
    obj.save_dir = str(save_path)

    # Save metadata
    meta = _metadata_from_obj(obj)
    (save_path / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save DataFrames
    obj.data.to_parquet(save_path / "data.parquet", index=True)

    if obj.pairwise_df is not None:
        obj.pairwise_df.to_parquet(save_path / "pairwise_df.parquet", index=True)

    if obj.scored_df is not None:
        obj.scored_df.to_parquet(save_path / "scored_df.parquet", index=True)

    print(f"Pairadigm saved to: {save_path}/")
    print(f"  Files written: metadata.json, data.parquet"
          + (", pairwise_df.parquet" if obj.pairwise_df is not None else "")
          + (", scored_df.parquet"   if obj.scored_df   is not None else ""))
    if any(c.provider not in ("ollama",) for c in obj.clients):
        print(
            "  Note: API keys and base URLs are NOT saved. You will need to supply them again "
            "when loading (via environment variables or the api_key and base_url parameters)."
        )


def load_pairadigm(
    save_dir: str, 
    api_keys: Optional[Union[str, List[str]]] = None,
    base_urls: Optional[Union[str, List[str]]] = None
) -> "Pairadigm":
    """
    Load a Pairadigm instance from a structured save directory.

    Parameters
    ----------
    save_dir : str
        Path to the directory created by :func:`save_pairadigm` or
        ``Pairadigm.save()``.
    api_keys : str or list of str, optional
        API key or list of API keys for the loaded models. Must match the
        number of models.
    base_urls : str or list of str, optional
        Base URL or list of Base URLs for the loaded models. Must match
        the number of models.

    Returns
    -------
    Pairadigm
        Fully reconstructed instance (clients are re-initialised from stored
        provider/model info; cloud-provider API keys are re-read from
        environment variables).

    Raises
    ------
    FileNotFoundError
        If ``save_dir`` or ``metadata.json`` are not found.
    ValueError
        If the path points to a legacy ``.pkl`` file (unsupported).
    """
    # Detect legacy pickle files
    path = Path(save_dir)
    if path.suffix == ".pkl":
        raise ValueError(
            "Legacy pickle files (.pkl) are not supported in Pairadigm v1.0.\n"
            "To read old saves, install pairadigm==0.5.4 (the last pickle-based "
            "release), load the object, then call .save(<new_dir>) with v1.0 "
            "to convert it to the new structured format."
        )

    if not path.exists():
        raise FileNotFoundError(f"Save directory not found: {path}")

    meta_file = path / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {path}. "
            "Is this a valid Pairadigm save directory?"
        )

    # Load metadata
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    # Load DataFrames
    data_path = path / "data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"data.parquet not found in {path}.")
    data = pd.read_parquet(data_path)

    pairwise_df = None
    if (path / "pairwise_df.parquet").exists():
        pairwise_df = pd.read_parquet(path / "pairwise_df.parquet")

    scored_df = None
    if (path / "scored_df.parquet").exists():
        scored_df = pd.read_parquet(path / "scored_df.parquet")

    # Reconstruct LLM clients
    from .client import LLMClient

    model_names = meta["model_names"]
    providers   = meta.get("providers", [None] * len(model_names))

    if api_keys is not None:
        if isinstance(api_keys, str):
            api_keys_list = [api_keys]
        else:
            api_keys_list = list(api_keys)
            
        if len(api_keys_list) != len(model_names):
            raise ValueError(
                f"Number of api_keys ({len(api_keys_list)}) must match "
                f"the number of model_names ({len(model_names)}). "
                f"Expected keys for the following models: {model_names}"
            )
    else:
        api_keys_list = [None] * len(model_names)

    if base_urls is not None:
        if isinstance(base_urls, str):
            base_urls_list = [base_urls]
        else:
            base_urls_list = list(base_urls)
            
        if len(base_urls_list) != len(model_names):
            raise ValueError(
                f"Number of base_urls ({len(base_urls_list)}) must match "
                f"the number of model_names ({len(model_names)}), even if passing 'None'."
            )
    else:
        base_urls_list = [None] * len(model_names)

    clients = []
    for model_name, provider, api_key_val, base_url_val in zip(model_names, providers, api_keys_list, base_urls_list):
        try:
            client = LLMClient(model_name=model_name, provider=provider, api_key=api_key_val, base_url=base_url_val)
        except ValueError as exc:
            # API key missing — create a shell client that will fail at use-time
            warnings.warn(
                f"Could not initialise client for '{model_name}' ({provider}): {exc}\n"
                "Set the appropriate environment variable before calling LLM methods.",
                UserWarning,
                stacklevel=2,
            )
            client = LLMClient.__new__(LLMClient)
            client.model_name = model_name
            client.provider   = provider
            client.api_key    = None
            client.base_url   = None
            client.client     = None
        clients.append(client)

    # Build a minimal Pairadigm shell and restore state
    from .core import Pairadigm

    obj = Pairadigm.__new__(Pairadigm)
    obj.clients      = clients
    obj.model_names  = model_names
    obj.data         = data
    obj.pairwise_df  = pairwise_df
    obj.scored_df    = scored_df

    _restore_metadata(obj, meta)

    # Ensure the loaded object knows where it was just loaded from,
    # overwriting whatever was in the metadata (in case the folder moved).
    obj.save_dir = str(path)

    print(f"Pairadigm loaded from: {path}/")
    return obj
