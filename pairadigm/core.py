"""
Pairadigm v1.0 — core module.

Contains the Pairadigm class (thin orchestrator) and the build_pairadigm
convenience function.  All heavy logic has been extracted to:
  - breakdowns.py   (CGCoT breakdown generation)
  - scoring.py      (BT/Davidson scoring)
  - validation.py   (alt_test, dawid_skene, irr, check_transitivity)
  - visualization.py (plot functions)
  - persistence.py  (save/load)
  - _stats.py       (pure statistical helpers)
  - client.py       (LLMClient)
"""

from __future__ import annotations

import itertools
import random
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .client import LLMClient
from . import breakdowns as _bd
from . import persistence as _persist
from . import scoring as _sc
from . import validation as _val
from . import visualization as _viz


_MODEL_COSTS_PER_1M_TOKENS = {
    # Format: model prefix string match: (input_cost_per_1m, output_cost_per_1m)
    
    # --- OpenAI Models ---
    "gpt-5.4": (2.50, 15.00),          # Latest flagship reasoning model
    "gpt-5.4-mini": (0.75, 4.50),     # High-performance efficient model
    "gpt-5.4-nano": (0.20, 1.25),     # Most cost-efficient GPT-5 class
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),              # Specialized reasoning (Standard)
    "o1-mini": (3.00, 12.00),          # Specialized reasoning (Efficient)
    "o3-mini": (1.10, 4.40),           # Optimized reasoning throughput
    
    # --- Google Gemini Models ---
    "gemini-3.1-pro": (2.00, 12.00),   # Flagship Gemini (<=200k context)
    "gemini-3-flash": (0.50, 3.00),    # Balanced speed/intelligence
    "gemini-2.5-flash": (0.30, 2.50),  # Improved 2.5 series
    "gemini-2.5-flash-lite": (0.10, 0.40), # Direct successor to 2.0 Flash
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    
    # --- Anthropic Claude Models ---
    "claude-4.6-opus": (5.00, 25.00),  # Peak intelligence frontier model
    "claude-4.6-sonnet": (3.00, 15.00), # Leading agentic/coding model
    "claude-4.5-haiku": (1.00, 5.00),   # Newest high-speed model
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-haiku": (0.25, 1.25),    # Legacy support
}

def _estimate_token_count(text: str) -> int:
    """Helper to estimate token count. Tries tiktoken, then heuristic fallbacks."""
    if not isinstance(text, str):
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback heuristic: word count * 1.33
        return int(len(text.split()) * 1.33)


################################
# Pairadigm class
################################

class Pairadigm:
    """
    Main class for Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation.

    ``Pairadigm`` orchestrates the full pipeline for measuring a latent concept
    (e.g. "persuasiveness", "clarity", "toxicity") across a corpus of text items
    using LLM-powered pairwise comparisons and Bradley-Terry scoring.

    Supports three flexible starting points:

    1. **Unpaired items** — provide a flat list of items and let Pairadigm
       generate breakdowns, create pairs, annotate them, and score everything.
    2. **Pre-paired items** — supply an existing pair DataFrame and skip the
       pairing step.
    3. **Human-annotated pairs** — bring your own annotations and use Pairadigm
       only for LLM annotation, IRR checking, and scoring.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.  Each row is one item (unpaired mode) or one pair
        (paired mode).
    item_id_name : str
        Name of the column that uniquely identifies each item (e.g. ``'essay_id'``).
        Required for unpaired data.
    text_name : str, optional
        Name of the column holding the raw text of each item (e.g. ``'essay_text'``).
        Used during breakdown generation.
    paired : bool, default False
        Set to ``True`` when ``data`` is already a pair-level DataFrame (one row
        per comparison pair).
    item_id_cols : list of str, optional
        **Paired mode only.** Two-element list naming the columns that hold the
        IDs of the left and right items in each pair (e.g. ``['id_a', 'id_b']``).
        They are internally renamed to ``'item1'`` / ``'item2'``.
    item_text_cols : list of str, optional
        **Paired mode only.** Two-element list naming the columns that hold the
        text of the left and right items.
    annotated : bool, default False
        Set to ``True`` when ``data`` already contains human annotation columns.
        Requires ``paired=True``.
    annotator_cols : list of str, optional
        Names of columns containing existing **human** annotation decisions
        (values should be ``'Text1'``, ``'Text2'``, or ``'Tie'``).
    llm_annotator_cols : list of str, optional
        Names of columns containing existing **LLM** annotation decisions.
    prior_breakdown_cols : list of str, optional
        Names of column(s) that already hold CGCoT breakdowns so that
        ``generate_breakdowns()`` can be skipped.  One column for unpaired data;
        two columns for paired data.
    cgcot_prompts : list of str
        **Required.** List of prompt templates used to generate CGCoT breakdowns.
        Every prompt must include a ``{text}`` placeholder.  Prompts after the
        first may also reference ``{previous_answers}`` to chain responses.
    model_name : str or list of str, default ``'gemini-2.0-flash-exp'``
        Name(s) of the LLM model(s) to use.  Pass a list to run multiple models
        in parallel (e.g. for ensemble annotation).
    api_key : str or list of str, optional
        API key(s) corresponding to each model.  Can be ``None`` if your
        environment already has the key set (e.g. via ``GOOGLE_API_KEY``).
    base_url : str or list of str, optional
        Base URL(s) for the LLM provider(s).  Useful for OpenAI-compatible
        local servers or proxies.
    target_concept : str
        **Required.** The concept being measured — used in comparison prompts and
        score column names (e.g. ``'persuasiveness'``, ``'argumentative quality'``).
    llm_clients : LLMClient or list of LLMClient, optional
        Pre-built :class:`~pairadigm.client.LLMClient` instances.  When supplied,
        ``model_name``, ``api_key``, and ``base_url`` are ignored.
    save_dir : str, optional
        Path to a directory for auto-saving after each major pipeline step.
        The directory is created if it does not exist.

    Examples
    --------
    **Minimal unpaired setup:**

    >>> import pandas as pd
    >>> from pairadigm import Pairadigm
    >>> df = pd.DataFrame({'essay_id': [1, 2, 3], 'text': ['...', '...', '...']})
    >>> prompts = [
    ...     "Describe the persuasive features of this essay: {text}",
    ...     "Given these features:\n{previous_answers}\nRate the argument quality.",
    ... ]
    >>> p = Pairadigm(
    ...     data=df,
    ...     item_id_name='essay_id',
    ...     text_name='text',
    ...     cgcot_prompts=prompts,
    ...     target_concept='persuasiveness',
    ...     api_key='YOUR_API_KEY',
    ... )

    **Paired data with existing human annotations:**

    >>> pairs_df = pd.DataFrame({
    ...     'id_a': [1, 2], 'id_b': [3, 1],
    ...     'text_a': ['...', '...'], 'text_b': ['...', '...'],
    ...     'human_judge': ['Text1', 'Text2'],
    ... })
    >>> p = Pairadigm(
    ...     data=pairs_df,
    ...     item_id_name='essay_id',
    ...     paired=True,
    ...     item_id_cols=['id_a', 'id_b'],
    ...     item_text_cols=['text_a', 'text_b'],
    ...     annotated=True,
    ...     annotator_cols=['human_judge'],
    ...     cgcot_prompts=prompts,
    ...     target_concept='persuasiveness',
    ...     api_key='YOUR_API_KEY',
    ... )
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        data: pd.DataFrame,
        item_id_name: Optional[str] = None,
        text_name: Optional[str] = None,
        paired: bool = False,
        item_id_cols: Optional[List[str]] = None,
        item_text_cols: Optional[List[str]] = None,
        annotated: bool = False,
        annotator_cols: Optional[List[str]] = None,
        llm_annotator_cols: Optional[List[str]] = None,
        prior_breakdown_cols: Optional[List[str]] = None,
        cgcot_prompts: Optional[List[str]] = None,
        model_name: Optional[Union[str, List[str]]] = "gemini-2.0-flash-exp",
        api_key: Optional[Union[str, List[str]]] = None,
        base_url: Optional[Union[str, List[str]]] = None,
        target_concept: Optional[str] = None,
        llm_clients: Optional[Union[LLMClient, List[LLMClient]]] = None,
        save_dir: Optional[str] = None,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # target_concept is required
        if target_concept is None:
            raise ValueError("target_concept must be specified.")

        # cgcot_prompts is required (changed from UserWarning → ValueError)
        if cgcot_prompts is None or not isinstance(cgcot_prompts, list) or len(cgcot_prompts) == 0:
            raise ValueError(
                "cgcot_prompts must be a non-empty list of prompt templates. "
                "Set them using .set_cgcot_prompts() if you do not have them ready at init time."
            )

        self.data = data.copy()
        self.target_concept = target_concept
        self.cgcot_prompts = cgcot_prompts
        self.paired = paired
        self.annotated = annotated
        self.column_renames: Dict[str, str] = {}  # 7f: transparency

        # Validate and set up unpaired inputs
        self._validate_unpaired_input(item_id_name, text_name, paired)
        # Validate and set up paired inputs
        item_id_cols, item_text_cols = self._validate_paired_input(
            item_id_cols, item_text_cols, paired
        )
        # Validate annotation inputs
        self._validate_annotations(annotated, annotator_cols, llm_annotator_cols, paired)

        self.item_id_name = item_id_name
        self.text_name = text_name
        self.item_id_cols = item_id_cols
        self.item_text_cols = item_text_cols
        self.annotator_cols = annotator_cols or []
        self.llm_annotator_cols = llm_annotator_cols or []
        self.llm_annotated = bool(llm_annotator_cols)
        self.prior_breakdown_cols = prior_breakdown_cols

        # Enforce standard prefixes for annotator columns:
        #   - Human / manual annotator columns → 'annotator_' prefix
        #   - LLM decision columns → 'decision_' prefix
        self._normalise_annotator_prefixes()

        # Apply column renames
        self._apply_column_renames(prior_breakdown_cols, paired)

        # Initialise clients
        self._init_clients(llm_clients, model_name, api_key, base_url)

        # Auto-save directory (9a-autosave)
        self.save_dir: Optional[str] = save_dir

        # Result storage
        self.pairwise_df: Optional[pd.DataFrame] = None
        if paired:
            self.pairwise_df = self.data.copy()
            if item_id_name is None:
                self.item_id_name = "item_id_DEFAULT"
        self.scored_df: Optional[pd.DataFrame] = None
        self.validation_results: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Private init helpers (fix 3f)
    # ------------------------------------------------------------------

    def _validate_unpaired_input(self, item_id_name, text_name, paired):
        if not paired:
            if item_id_name not in self.data.columns:
                raise ValueError(f"Column '{item_id_name}' not found in DataFrame.")
            if text_name and text_name not in self.data.columns:
                raise ValueError(f"Column '{text_name}' not found in DataFrame.")
            # Quality check: warn on NA values in text column
            if text_name and self.data[text_name].isna().any():
                n_na = self.data[text_name].isna().sum()
                warnings.warn(
                    f"Text column '{text_name}' contains {n_na} NA value(s). "
                    "These items may cause errors during breakdown generation.",
                    UserWarning,
                )

    def _validate_paired_input(self, item_id_cols, item_text_cols, paired):
        if not paired:
            return item_id_cols, item_text_cols
        if item_id_cols is None or len(item_id_cols) != 2:
            raise ValueError(
                "For paired data, item_id_cols must be a list of two column names."
            )
        for col in item_id_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        if item_text_cols is None or len(item_text_cols) != 2:
            raise ValueError(
                "For paired data, item_text_cols must be a list of two column names."
            )
        for col in item_text_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        # Rename to canonical item1/item2
        if item_id_cols[0] != "item1" or item_id_cols[1] != "item2":
            self.data = self.data.rename(
                columns={item_id_cols[0]: "item1", item_id_cols[1]: "item2"}
            )
            self.column_renames[item_id_cols[0]] = "item1"
            self.column_renames[item_id_cols[1]] = "item2"
            item_id_cols = ["item1", "item2"]
        return item_id_cols, item_text_cols

    def _validate_annotations(self, annotated, annotator_cols, llm_annotator_cols, paired):
        if not annotated:
            return
        num_llm = len(llm_annotator_cols) if llm_annotator_cols else 0
        if not annotator_cols and num_llm < 1:
            raise ValueError(
                "For annotated data, annotator_cols must contain human annotation column names."
            )
        for col in (annotator_cols or []):
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        if annotated and not paired:
            raise ValueError(
                "If data is annotated, it must also be paired (paired=True)."
            )

    def _normalise_annotator_prefixes(self) -> None:
        """Enforce standard column prefixes: 'annotator_' for human, 'decision_' for LLM."""
        # --- Human / manual annotator columns: should NOT start with 'decision_' ---
        new_annotator_cols: List[str] = []
        for col in self.annotator_cols:
            if col.startswith("decision_"):
                new_name = "annotator_" + col[len("decision_"):]
                if col in self.data.columns:
                    self.data = self.data.rename(columns={col: new_name})
                self.column_renames[col] = new_name
                new_annotator_cols.append(new_name)
                warnings.warn(
                    f"Manual annotator column '{col}' renamed to '{new_name}' "
                    "(the 'decision_' prefix is reserved for LLM annotations).",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                new_annotator_cols.append(col)
        self.annotator_cols = new_annotator_cols

        # --- LLM annotator columns: should start with 'decision_' ---
        new_llm_cols: List[str] = []
        for col in self.llm_annotator_cols:
            if not col.startswith("decision_") and col != "decision":
                new_name = f"decision_{col}"
                if col in self.data.columns:
                    self.data = self.data.rename(columns={col: new_name})
                self.column_renames[col] = new_name
                new_llm_cols.append(new_name)
                warnings.warn(
                    f"LLM annotator column '{col}' renamed to '{new_name}' "
                    "(LLM decision columns should use the 'decision_' prefix).",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                new_llm_cols.append(col)
        self.llm_annotator_cols = new_llm_cols

    def _apply_column_renames(self, prior_breakdown_cols, paired):
        if prior_breakdown_cols is None:
            return
        if paired:
            if len(prior_breakdown_cols) != 2:
                raise ValueError(
                    "For paired data, prior_breakdown_cols must contain exactly 2 column names."
                )
            renames = {}
            if prior_breakdown_cols[0] != "breakdown1":
                renames[prior_breakdown_cols[0]] = "breakdown1"
            if prior_breakdown_cols[1] != "breakdown2":
                renames[prior_breakdown_cols[1]] = "breakdown2"
            if renames:
                self.data = self.data.rename(columns=renames)
                self.column_renames.update(renames)
                self.prior_breakdown_cols = ["breakdown1", "breakdown2"]
        else:
            if len(prior_breakdown_cols) != 1:
                raise ValueError(
                    "For unpaired data, prior_breakdown_cols must contain exactly 1 column name."
                )
            if prior_breakdown_cols[0] != "breakdown1":
                self.data = self.data.rename(
                    columns={prior_breakdown_cols[0]: "breakdown1"}
                )
                self.column_renames[prior_breakdown_cols[0]] = "breakdown1"
                self.prior_breakdown_cols = ["breakdown1"]

    def _init_clients(self, llm_clients, model_name, api_key, base_url):
        if llm_clients is not None:
            if isinstance(llm_clients, LLMClient):
                self.clients = [llm_clients]
            elif isinstance(llm_clients, list):
                self.clients = llm_clients
            else:
                raise TypeError("llm_clients must be LLMClient or List[LLMClient].")
            self.model_names = [c.model_name for c in self.clients]
            return

        if isinstance(model_name, str):
            model_name = [model_name]
        elif not isinstance(model_name, list):
            raise TypeError("model_name must be str or List[str].")

        def _normalise(param, name, length):
            if param is None:
                return [None] * length
            if isinstance(param, str):
                return [param] * length
            if isinstance(param, list):
                if len(param) != length:
                    raise ValueError(
                        f"If {name} is a list, it must have the same length as model_name."
                    )
                return param
            raise TypeError(f"{name} must be str, list of str, or None.")

        base_url = _normalise(base_url, "base_url", len(model_name))
        api_key  = _normalise(api_key,  "api_key",  len(model_name))

        self.clients = [
            LLMClient(api_key=k, model_name=m, base_url=u)
            for m, k, u in zip(model_name, api_key, base_url)
        ]
        self.model_names = model_name

    # ------------------------------------------------------------------
    # @property client (fix 3i)
    # ------------------------------------------------------------------

    @property
    def client(self) -> LLMClient:
        """Convenience accessor — returns the first (primary) LLM client."""
        return self.clients[0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_clients(
        self, client_indices: Optional[Union[int, List[int]]]
    ) -> List[Tuple[int, LLMClient]]:
        """Return [(index, client), ...] from an optional index selector. (fix 7d)"""
        if client_indices is None:
            return list(enumerate(self.clients))
        if isinstance(client_indices, int):
            if client_indices >= len(self.clients):
                raise ValueError(
                    f"client_indices {client_indices} out of range "
                    f"(only {len(self.clients)} client(s) available)."
                )
            return [(client_indices, self.clients[client_indices])]
        if isinstance(client_indices, list):
            out = []
            for idx in client_indices:
                if idx >= len(self.clients):
                    raise ValueError(
                        f"client_indices {idx} out of range "
                        f"(only {len(self.clients)} client(s) available)."
                    )
                out.append((idx, self.clients[idx]))
            return out
        raise TypeError("client_indices must be None, int, or List[int].")

    def get_score_col_name(
        self,
        decision_col: str = "decision",
        split: Optional[str] = None,
        model_name: str = "Bradley_Terry",
    ) -> str:
        """Return the canonical score column name. (fix 7b)"""
        return _sc.get_score_col_name(decision_col, split, model_name)

    # ------------------------------------------------------------------
    # Public setup helpers
    # ------------------------------------------------------------------

    def set_cgcot_prompts(self, prompts: Union[List[str], str]) -> None:
        """
        Replace the CGCoT prompt templates used for breakdown generation.

        Accepts either a Python list of prompt strings or a path to a plain-text
        file.  In the file format, prompts may be separated by blank lines
        (``\\n\\n``), ``---`` delimiters, or one prompt per line.

        Every prompt **must** contain a ``{text}`` placeholder.  Prompts after
        the first may additionally reference ``{previous_answers}`` to build
        chain-of-thought responses.

        Parameters
        ----------
        prompts : list of str or str
            Either a list of prompt template strings, or a file-system path to a
            ``.txt`` file containing the prompts.

        Examples
        --------
        **From a Python list:**

        >>> p.set_cgcot_prompts([
        ...     "Summarise the argument made in this essay: {text}",
        ...     "Given the summary:\n{previous_answers}\nHow persuasive is this?",
        ... ])

        **From a file:**

        >>> p.set_cgcot_prompts('my_prompts.txt')
        """
        if isinstance(prompts, str):
            fp = Path(prompts)
            if not fp.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompts}")
            content = fp.read_text(encoding="utf-8").strip()
            if "\n\n" in content:
                prompt_list = [p.strip() for p in content.split("\n\n") if p.strip()]
            elif "---" in content:
                prompt_list = [p.strip() for p in content.split("---") if p.strip()]
            else:
                prompt_list = [ln.strip() for ln in content.split("\n") if ln.strip()]
            if not prompt_list:
                raise ValueError("No valid prompts found in file.")
            if self._validate_prompts(prompt_list):
                self.cgcot_prompts = prompt_list
        elif isinstance(prompts, list):
            if len(prompts) == 0:
                raise ValueError("prompts must be a non-empty list.")
            if self._validate_prompts(prompts):
                self.cgcot_prompts = prompts
        else:
            raise TypeError("prompts must be a list of strings or a file path string.")

    def get_clients_info(self) -> pd.DataFrame:
        """
        Return a summary of all registered LLM clients.

        Returns
        -------
        pd.DataFrame
            One row per client with columns ``index``, ``model_name``, and
            ``provider``.

        Examples
        --------
        >>> p.get_clients_info()
           index           model_name  provider
        0      0  gemini-2.0-flash-exp    google
        1      1               gpt-4o    openai
        """
        return pd.DataFrame(
            [{"index": i, "model_name": c.model_name, "provider": c.provider}
             for i, c in enumerate(self.clients)]
        )

    def test_clients_connection(
        self,
        test_prompt: str = "What is the best restaurant in Detroit, MI?",
        return_responses: bool = False
    ) -> Union[Dict[str, bool], Dict[str, str]]:
        """
        Send a test prompt to every registered LLM client and check connectivity.

        Useful for verifying that API keys are valid and the models are reachable
        before kicking off a long annotation job.

        Parameters
        ----------
        test_prompt : str, optional
            The prompt sent to each model.  Defaults to a simple factual question.
        return_responses : bool, default False
            If ``False`` (default), returns a dict of ``{model_name: bool}``
            where ``True`` means the model returned a non-empty response.
            If ``True``, returns the raw response strings instead of booleans.

        Returns
        -------
        dict
            ``{model_name: bool}`` or ``{model_name: str}`` depending on
            ``return_responses``.

        Examples
        --------
        >>> p.test_clients_connection()
        Testing LLM client connections using: 'What is the best restaurant...'...
          gemini-2.0-flash-exp: MODEL OK
        {'gemini-2.0-flash-exp': True}

        >>> p.test_clients_connection(return_responses=True)
        {'gemini-2.0-flash-exp': 'Supino Pizzeria on East ...'}
        """
        results: Dict[str, Union[bool, str]] = {}
        print(f"Testing LLM client connections using: '{test_prompt}'...")
        for c in self.clients:
            try:
                resp = c.generate(prompt=test_prompt, max_tokens=50)
                if return_responses:
                    results[c.model_name] = resp
                else:
                    results[c.model_name] = bool(resp)
                print(f"  {c.model_name}: {'MODEL OK' if resp else 'EMPTY RESPONSE'}")
            except Exception as exc:
                results[c.model_name] = False
                print(f"  {c.model_name}: FAILED ({exc})")
        return results

    def _validate_prompts(self, prompts: List[str]) -> bool:
        """Validate CGCoT prompt templates. (Fix 7g: check {previous_answers} for chained prompts.)"""
        if not prompts or not isinstance(prompts, list):
            raise ValueError("Prompts must be a non-empty list.")
        for i, p in enumerate(prompts):
            if "{text}" not in p:
                raise ValueError(
                    f"Prompt {i + 1} is missing {{text}} placeholder: {p[:60]}..."
                )
            # 7g: warn if a chained prompt (index > 0) doesn't reference {previous_answers}
            # The CGCoT engine always passes `previous_answers` to the format call, so
            # omitting it is not an error per se, but likely an oversight.
            if i > 0 and "{previous_answers}" not in p:
                warnings.warn(
                    f"Prompt {i + 1} does not contain {{previous_answers}}. "
                    "In a multi-prompt chain, each prompt after the first has access "
                    "to previous responses via {previous_answers}. If this is intentional "
                    "(e.g., each prompt is fully independent) you can ignore this warning.",
                    UserWarning,
                    stacklevel=3,
                )
        return True

    # ------------------------------------------------------------------
    # Cost Estimation
    # ------------------------------------------------------------------

    def estimate_costs(
        self,
        stage: str = "both",
        custom_cost_per_1m_input: Optional[float] = None,
        custom_cost_per_1m_output: Optional[float] = None,
        client_indices: Optional[Union[int, List[int]]] = None,
        expected_breakdown_output_tokens: int = 200,
        expected_pairwise_output_tokens: int = 50,
        system_message: str = _bd._DEFAULT_BREAKDOWN_SYSTEM_MSG,
        comparison_prompt: Optional[str] = None
    ) -> None:
        """
        Estimate the number of tokens and API cost for running the pipeline.
        
        Parameters
        ----------
        stage : str, default 'both'
            Which pipeline stage to estimate. Options: 'breakdowns', 'pairwise', 'both'.
        custom_cost_per_1m_input : float, optional
            Override the input cost per 1M tokens.
        custom_cost_per_1m_output : float, optional
            Override the output cost per 1M tokens.
        client_indices : int or list of int, optional
            Which client(s) to use by index for the estimation. None = all.
        expected_breakdown_output_tokens: int, default 200
            Expected average output tokens for a breakdown stage.
        expected_pairwise_output_tokens: int, default 50
            Expected average output tokens for a pairwise comparison stage.
        system_message : str, optional
            The system message to include in calculations.
        comparison_prompt : str, optional
            The pairwise comparison prompt to include in calculations.
        """
        clients_to_use = self._resolve_clients(client_indices)
        
        print("\n" + "="*60)
        print("          LLM API COST ESTIMATION")
        print("="*60)
        print("DISCLAIMER: These are rough heuristics for token counting")
        print("and pricing based on general models. They do not reflect")
        print("real-time constraints, retries, or precise tokenizers.")
        print("-" * 60)

        for client_idx, client in clients_to_use:
            model_name = self.model_names[client_idx]
            
            if client.provider in ['ollama', 'huggingface']:
                if custom_cost_per_1m_input is None or custom_cost_per_1m_output is None:
                    raise ValueError(
                        f"Provider '{client.provider}' detected for model '{model_name}'. "
                        "You must pass in values for custom_cost_per_1m_input and custom_cost_per_1m_output. "
                        "If you are using local models you have downloaded and are running on your own hardware, these values should just be 0."
                    )

            in_cost, out_cost = 0.0, 0.0
            
            if custom_cost_per_1m_input is not None and custom_cost_per_1m_output is not None:
                in_cost = custom_cost_per_1m_input
                out_cost = custom_cost_per_1m_output
            else:
                # Match longest prefix to handle names like gpt-4o-mini correctly before gpt-4o
                matched_prefix = None
                for prefix, (ic, oc) in sorted(_MODEL_COSTS_PER_1M_TOKENS.items(), key=lambda x: -len(x[0])):
                    if prefix in model_name.lower():
                        matched_prefix = prefix
                        in_cost, out_cost = ic, oc
                        break
            
            print(f"Client [{client_idx}]: {model_name}")
            if in_cost > 0 or out_cost > 0:
                print(f"Pricing used: ${in_cost:.3f} per 1M input | ${out_cost:.3f} per 1M output")
            else:
                print("Pricing used: Unknown or $0 per 1M tokens")
            
            total_input = 0
            total_output = 0
            sys_tokens = _estimate_token_count(system_message)
            
            if stage in ["breakdowns", "both"]:
                if self.data is not None and (self.text_name in self.data.columns or self.paired):
                    source_df = self.pairwise_df if self.pairwise_df is not None and self.paired else self.data
                    if self.paired and self.item_id_cols and self.item_text_cols:
                        items_col1 = source_df[[self.item_id_cols[0], self.item_text_cols[0]]].rename(columns={self.item_id_cols[0]: 'id', self.item_text_cols[0]: 'text'})
                        items_col2 = source_df[[self.item_id_cols[1], self.item_text_cols[1]]].rename(columns={self.item_id_cols[1]: 'id', self.item_text_cols[1]: 'text'})
                        unique_items = pd.concat([items_col1, items_col2]).drop_duplicates(subset=['id']).dropna(subset=['text'])
                        texts = unique_items['text'].tolist()
                    elif self.text_name in self.data.columns:
                        texts = self.data[self.text_name].dropna().tolist()
                    else:
                        texts = []
                        
                    num_items = len(texts)
                    if num_items > 0:
                        prompt_len = sum(_estimate_token_count(p) for p in self.cgcot_prompts)
                        text_lens = sum(_estimate_token_count(str(t)) for t in texts)
                        
                        b_input = (sys_tokens * num_items * len(self.cgcot_prompts)) + (prompt_len * num_items) + (text_lens * len(self.cgcot_prompts))
                        b_out = expected_breakdown_output_tokens * num_items * len(self.cgcot_prompts)
                        
                        total_input += b_input
                        total_output += b_out
                        
                        print(f"  [Breakdowns] Items: {num_items}, Expected Input: ~{b_input:,}, Expected Output: ~{b_out:,}")
                    else:
                        print(f"  [Breakdowns] Could not find text items for calculation.")
            
            if stage in ["pairwise", "both"]:
                if self.pairwise_df is not None:
                    num_pairs = len(self.pairwise_df)
                    multi = len(self.clients) > 1
                    bd1 = f"breakdown1_{model_name}" if multi else "breakdown1"
                    bd2 = f"breakdown2_{model_name}" if multi else "breakdown2"
                    
                    avg_bd_len = expected_breakdown_output_tokens
                    if bd1 in self.pairwise_df.columns and bd2 in self.pairwise_df.columns:
                        sample_bd = self.pairwise_df[bd1].dropna().head(10).tolist()
                        if sample_bd:
                            avg_bd_len = int(np.mean([_estimate_token_count(str(x)) for x in sample_bd]))
                    
                    base_prompt_len = 100 if comparison_prompt is None else _estimate_token_count(comparison_prompt)
                    
                    pw_input = (sys_tokens * num_pairs) + (base_prompt_len * num_pairs) + (avg_bd_len * 2 * num_pairs)
                    pw_out = expected_pairwise_output_tokens * num_pairs
                    
                    total_input += pw_input
                    total_output += pw_out
                    
                    print(f"  [Pairwise] Pairs: {num_pairs}, Expected Input: ~{pw_input:,}, Expected Output: ~{pw_out:,}")
            
            cost_in = (total_input / 1_000_000) * in_cost
            cost_out = (total_output / 1_000_000) * out_cost
            total_cost = cost_in + cost_out

            print(f"  >>> Estimated Cost: ${total_cost:.4f}")
            print("-" * 60)

    # ------------------------------------------------------------------
    # Breakdown generation (delegates to breakdowns.py)
    # ------------------------------------------------------------------

    def generate_breakdowns(
        self,
        max_workers: int = 8,
        rate_limit_per_minute: Optional[int] = None,
        update_dataframe: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        client_indices: Optional[Union[int, List[int]]] = None,
        show_progress: bool = True,
        system_message: str = _bd._DEFAULT_BREAKDOWN_SYSTEM_MSG,
        debug_mode: bool = False,
    ) -> Optional[Dict]:
        """
        Generate CGCoT (Concept-Guided Chain-of-Thought) breakdowns for every item.

        Works for both **unpaired** and **paired** data — the mode is determined
        automatically from ``self.paired``.

        - **Unpaired**: one breakdown per row in ``self.data``; results are written
          as a ``CGCoT_Breakdown`` column.
        - **Paired**: unique items are extracted from both pair columns, one
          breakdown per unique item; results are written as ``breakdown1`` /
          ``breakdown2`` columns in ``self.pairwise_df``.

        Parameters
        ----------
        max_workers : int, default 8
            Number of parallel threads used for LLM calls.  Increase for faster
            throughput; decrease if you hit rate limits.
        rate_limit_per_minute : int or None, optional
            Hard cap on API calls per minute.  Set this if your LLM tier has a
            strict rate limit (e.g. ``60`` for many free-tier accounts).
        update_dataframe : bool, default True
            If ``True`` (default), writes breakdown columns directly into
            ``self.data`` (unpaired) or ``self.pairwise_df`` (paired) and returns
            ``None``.  If ``False``, returns the raw result dict without modifying
            the object.
        max_tokens : int, default 1000
            Maximum tokens the LLM may generate per prompt step.
        temperature : float, default 0.0
            LLM sampling temperature.  ``0.0`` gives deterministic outputs.
        client_indices : int or list of int, optional
            Which client(s) to use (by index).  ``None`` uses all registered
            clients.  Pass ``0`` to use only the first client.
        show_progress : bool, default True
            Whether to display a ``tqdm`` progress bar.
        system_message : str, optional
            System prompt prepended to every breakdown call.  Defaults to a
            generic research-assistant instruction.
        debug_mode : bool, default False
            When ``True``, preserves ``"Prompt N response:"`` section headers in
            stored breakdowns — useful when iterating on prompt templates.

        Returns
        -------
        None or dict
            ``None`` when ``update_dataframe=True``.  A result dict when
            ``update_dataframe=False``:

            - Unpaired: ``{item_id: breakdown_text}`` (single client) or
              ``{client_index: {item_id: breakdown_text}}`` (multiple clients).
            - Paired: always ``{client_index: {item_id: breakdown_text}}``.

        Examples
        --------
        >>> p.generate_breakdowns()                       # unpaired
        >>> p.generate_breakdowns()                       # paired — same call!
        >>> p.generate_breakdowns(max_workers=4, rate_limit_per_minute=60)
        """
        # Run cost estimation before execution
        try:
            self.estimate_costs(stage="breakdowns", client_indices=client_indices, system_message=system_message)
        except ValueError as e:
            print(f"\nCost estimation skipped: {e}")
        
        shared_kwargs = dict(
            cgcot_prompts=self.cgcot_prompts,
            clients=self.clients,
            model_names=self.model_names,
            client_indices=client_indices,
            max_workers=max_workers,
            rate_limit_per_minute=rate_limit_per_minute,
            max_tokens=max_tokens,
            temperature=temperature,
            show_progress=show_progress,
            system_message=system_message,
            debug_mode=debug_mode,
        )

        if self.paired:
            source_df = self.pairwise_df if self.pairwise_df is not None else self.data
            all_results = _bd.generate_breakdowns(
                data=source_df,
                item_id_name=self.item_id_name,
                item_id_cols=self.item_id_cols,
                item_text_cols=self.item_text_cols,
                **shared_kwargs,
            )
            if update_dataframe and self.pairwise_df is not None:
                clients_to_use = self._resolve_clients(client_indices)
                item1_id_col, item2_id_col = self.item_id_cols
                for idx, _ in clients_to_use:
                    model_name = self.model_names[idx]
                    bd1 = f"breakdown1_{model_name}" if len(self.clients) > 1 else "breakdown1"
                    bd2 = f"breakdown2_{model_name}" if len(self.clients) > 1 else "breakdown2"
                    res = all_results[idx]
                    self.pairwise_df[bd1] = self.pairwise_df[item1_id_col].map(res)
                    self.pairwise_df[bd2] = self.pairwise_df[item2_id_col].map(res)
                print(
                    "\nBreakdowns added to [object].pairwise_df — column(s): "
                    + ", ".join(
                        f"breakdown1_{self.model_names[idx]}, breakdown2_{self.model_names[idx]}"
                        if len(self.clients) > 1
                        else "breakdown1, breakdown2"
                        for idx, _ in clients_to_use
                    )
                )
                if self.save_dir:
                    self.save(self.save_dir)
                    print(f"Auto-saved to: {self.save_dir}")
                return None
            return all_results
        else:
            all_results = _bd.generate_breakdowns(
                data=self.data,
                item_id_name=self.item_id_name,
                text_name=self.text_name,
                **shared_kwargs,
            )
            if update_dataframe:
                clients_to_use = self._resolve_clients(client_indices)
                if len(clients_to_use) > 1:
                    for idx, _ in clients_to_use:
                        col = (
                            f"CGCoT_Breakdown_{self.model_names[idx]}"
                            if len(self.clients) > 1
                            else "CGCoT_Breakdown"
                        )
                        self.data[col] = self.data[self.item_id_name].map(all_results[idx])
                else:
                    idx = clients_to_use[0][0]
                    col = (
                        f"CGCoT_Breakdown_{self.model_names[idx]}"
                        if len(self.clients) > 1
                        else "CGCoT_Breakdown"
                    )
                    self.data[col] = self.data[self.item_id_name].map(all_results[idx])
                print(
                    "\nBreakdowns added to [object].data — column(s): "
                    + ", ".join(
                        f"CGCoT_Breakdown_{self.model_names[idx]}" if len(self.clients) > 1
                        else "CGCoT_Breakdown"
                        for idx, _ in clients_to_use
                    )
                )
                if self.save_dir:
                    self.save(self.save_dir)
                    print(f"Auto-saved to: {self.save_dir}")
                return None
            return all_results

    def generate_breakdowns_from_paired(
        self,
        max_workers: int = 8,
        rate_limit_per_minute: Optional[int] = None,
        update_pairwise_df: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        client_indices: Optional[Union[int, List[int]]] = None,
        show_progress: bool = True,
        system_message: str = _bd._DEFAULT_BREAKDOWN_SYSTEM_MSG,
        debug_mode: bool = False,
    ) -> Dict:
        """
        Alias for :meth:`generate_breakdowns` on paired data.

        .. deprecated::
            Call :meth:`generate_breakdowns` directly — it now handles both
            paired and unpaired data automatically.
        """
        import warnings
        warnings.warn(
            "generate_breakdowns_from_paired() is deprecated. "
            "Call generate_breakdowns() instead — it handles both paired and "
            "unpaired data automatically.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.generate_breakdowns(
            max_workers=max_workers,
            rate_limit_per_minute=rate_limit_per_minute,
            update_dataframe=update_pairwise_df,
            max_tokens=max_tokens,
            temperature=temperature,
            client_indices=client_indices,
            show_progress=show_progress,
            system_message=system_message,
            debug_mode=debug_mode,
        )

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------

    @staticmethod
    def pair_items(items, num_pairs_per_item=10, random_seed=42) -> pd.DataFrame:
        """
        Generate a sparse but connected set of pairwise comparison pairs.

        Ensures every item appears in at least ``num_pairs_per_item`` pairs,
        while keeping the total pair count manageable (not all N*(N-1)/2
        combinations).  A sequential backbone guarantees connectivity.

        This is a static method — it can be called without an instance:
        ``Pairadigm.pair_items(my_list)`` — and is also available at the
        module level as ``pairadigm.pair_items(...)``.

        Parameters
        ----------
        items : list
            List of item IDs to pair.  Any hashable type is accepted
            (strings, integers, etc.).
        num_pairs_per_item : int, default 10
            Target number of comparisons each item should appear in.  A higher
            value increases coverage at the cost of more LLM calls.
        random_seed : int, default 42
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Two-column DataFrame with ``'item1'`` and ``'item2'`` columns;
            each row is one unique pair.

        Examples
        --------
        >>> ids = ['essay_1', 'essay_2', 'essay_3', 'essay_4']
        >>> Pairadigm.pair_items(ids, num_pairs_per_item=2)
             item1    item2
        0  essay_1  essay_2
        1  essay_2  essay_3
        ....
        """
        if random_seed is not None:
            random.seed(random_seed)
        n = len(items)
        if n < 2:
            return pd.DataFrame(columns=["item1", "item2"])
        min_pairs = num_pairs_per_item or max(3, min(6, int(n ** 0.5)))
        all_pairs = set(itertools.combinations(items, 2))
        chosen: set = set()
        covered = {item: set() for item in items}
        for i in range(n - 1):
            pair = tuple(sorted((items[i], items[i + 1])))
            chosen.add(pair)
            covered[items[i]].add(items[i + 1])
            covered[items[i + 1]].add(items[i])
        extra = list(all_pairs - chosen)
        random.shuffle(extra)
        for a, b in extra:
            if len(covered[a]) < min_pairs or len(covered[b]) < min_pairs:
                chosen.add((a, b))
                covered[a].add(b)
                covered[b].add(a)
        return pd.DataFrame(list(chosen), columns=["item1", "item2"])

    def generate_pairings(
        self,
        num_pairs_per_item: int = 10,
        random_seed: int = 42,
        breakdowns: bool = False,
        update_classObject: bool = True,
        make_splits: bool = False,
        test_size: float = 0.15,
        eval_size: float = 0.15,
        include_mixed_pairs: bool = False,
        num_mixed_pairs: int = 10,
    ) -> pd.DataFrame:
        """
        Generate pairwise comparison pairs from the items in ``self.data``.

        Internally calls :meth:`pair_items` and (optionally) stratifies items
        into train / eval / test splits to prevent data leakage when the
        resulting annotations will be used for model training.

        Parameters
        ----------
        num_pairs_per_item : int, default 10
            How many comparison partners each item should be paired with.
            Higher values yield more stable Bradley-Terry scores but require
            more LLM calls.
        random_seed : int, default 42
            Random seed passed to the pairing algorithm and split generator.
        breakdowns : bool, default False
            If ``True``, breakdown text (from ``CGCoT_Breakdown`` columns) is
            automatically joined onto the pair DataFrame as ``breakdown1`` and
            ``breakdown2`` columns.  Requires :meth:`generate_breakdowns` to
            have been run first.
        update_classObject : bool, default True
            If ``True``, stores the resulting pair DataFrame in
            ``self.pairwise_df`` and updates ``self.item_id_cols``.
        make_splits : bool, default False
            If ``True``, splits items into train / eval / test groups *before*
            pairing, so that pairs never span split boundaries (preventing
            data leakage).  Strongly recommended if annotations will train a
            model.
        test_size : float, default 0.15
            Fraction of items assigned to the test split (only when
            ``make_splits=True``).
        eval_size : float, default 0.15
            Fraction of items assigned to the eval split (only when
            ``make_splits=True``).
        include_mixed_pairs : bool, default False
            If ``True``, adds a small number of cross-split pairs
            (train-eval, train-test, eval-test).  Requires ``make_splits=True``.
        num_mixed_pairs : int, default 10
            Total number of cross-split pairs to add when
            ``include_mixed_pairs=True``.

        Returns
        -------
        pd.DataFrame
            Pair DataFrame with at minimum ``item1`` and ``item2`` columns.
            When ``make_splits=True``, also includes ``item1_split`` and
            ``item2_split``.  When ``breakdowns=True``, also includes
            ``breakdown1`` and ``breakdown2``.

        Examples
        --------
        >>> pairs = p.generate_pairings(num_pairs_per_item=8)
        >>> pairs.head()
             item1    item2
        0  essay_1  essay_3
        ...

        >>> # With train/eval/test splits and breakdowns already generated:
        >>> pairs = p.generate_pairings(
        ...     num_pairs_per_item=10,
        ...     make_splits=True,
        ...     breakdowns=True,
        ... )
        """
        _DEFAULT_TEST = 0.15
        _DEFAULT_EVAL = 0.15

        if not make_splits and (test_size != _DEFAULT_TEST or eval_size != _DEFAULT_EVAL):
            warnings.warn(
                "Non-default test_size or eval_size passed without make_splits=True. "
                "Setting make_splits=True automatically.",
                UserWarning,
            )
            make_splits = True

        if include_mixed_pairs and not make_splits:
            raise ValueError(
                "include_mixed_pairs=True requires make_splits=True."
            )
        if make_splits:
            if test_size <= 0 or eval_size <= 0:
                raise ValueError("test_size and eval_size must be > 0.")
            if test_size + eval_size >= 1.0:
                raise ValueError(
                    f"test_size ({test_size}) + eval_size ({eval_size}) must be < 1.0."
                )

        if self.paired:
            raise ValueError(
                "Data is already in paired format. Cannot generate pairings. "
                "Use generate_pairwise_annotations() directly."
            )

        all_items = self.data[self.item_id_name].tolist()

        if not make_splits:
            uuid_pairings = self.pair_items(all_items, num_pairs_per_item, random_seed)
            warnings.warn(
                "No train/eval/test splits applied. Consider make_splits=True to prevent "
                "data leakage during model training.",
                UserWarning,
            )
        else:
            train_eval, test_items = train_test_split(
                all_items, test_size=test_size, random_state=random_seed
            )
            adjusted_eval = eval_size / (1.0 - test_size)
            train_items, eval_items = train_test_split(
                train_eval, test_size=adjusted_eval, random_state=random_seed
            )
            print(
                f"Item-level splits — train: {len(train_items)}, "
                f"eval: {len(eval_items)}, test: {len(test_items)}"
            )
            warnings.warn(
                "Item-level splits applied (train/eval/test). Inspect splits before model training.",
                UserWarning,
            )
            item_to_split = (
                {i: "train" for i in train_items}
                | {i: "eval"  for i in eval_items}
                | {i: "test"  for i in test_items}
            )
            split_dfs = []
            for sname, sitems in [("train", train_items), ("eval", eval_items), ("test", test_items)]:
                if len(sitems) >= 2:
                    split_dfs.append(self.pair_items(sitems, num_pairs_per_item, random_seed))
                else:
                    warnings.warn(f"Split '{sname}' has fewer than 2 items; no pairs generated.", UserWarning)
            uuid_pairings = pd.concat(split_dfs, ignore_index=True)
            uuid_pairings["item1_split"] = uuid_pairings["item1"].map(item_to_split)
            uuid_pairings["item2_split"] = uuid_pairings["item2"].map(item_to_split)

            if include_mixed_pairs:
                existing = (
                    set(map(tuple, uuid_pairings[["item1", "item2"]].values))
                    | set(map(tuple, uuid_pairings[["item2", "item1"]].values))
                )
                rng = random.Random(random_seed)
                cross = [
                    ("train", train_items, "eval", eval_items),
                    ("train", train_items, "test", test_items),
                    ("eval",  eval_items,  "test", test_items),
                ]
                per_combo = max(1, num_mixed_pairs // len(cross))
                mixed_rows = []
                for sa, ia, sb, ib in cross:
                    candidates = [(a, b) for a in ia for b in ib
                                  if (a, b) not in existing and (b, a) not in existing]
                    rng.shuffle(candidates)
                    for a, b in candidates[:per_combo]:
                        mixed_rows.append({"item1": a, "item2": b,
                                           "item1_split": sa, "item2_split": sb})
                        existing.add((a, b))
                if mixed_rows:
                    uuid_pairings = pd.concat(
                        [uuid_pairings, pd.DataFrame(mixed_rows)], ignore_index=True
                    )
                    print(f"Added {len(mixed_rows)} cross-split (mixed) pairs.")

        self.item_id_cols = ["item1", "item2"]

        if breakdowns:
            bd_cols = [c for c in self.data.columns if c.startswith("CGCoT_Breakdown")]
            if not bd_cols:
                raise ValueError(
                    "No 'CGCoT_Breakdown' columns found. Run generate_breakdowns() first."
                )
            for col in bd_cols:
                uuid_to_desc = dict(zip(self.data[self.item_id_name], self.data[col]))
                if col == "CGCoT_Breakdown":
                    uuid_pairings["breakdown1"] = uuid_pairings["item1"].map(uuid_to_desc)
                    uuid_pairings["breakdown2"] = uuid_pairings["item2"].map(uuid_to_desc)
                else:
                    suffix = col[len("CGCoT_Breakdown_"):]
                    uuid_pairings[f"breakdown1_{suffix}"] = uuid_pairings["item1"].map(uuid_to_desc)
                    uuid_pairings[f"breakdown2_{suffix}"] = uuid_pairings["item2"].map(uuid_to_desc)

        if update_classObject:
            self.pairwise_df = uuid_pairings
            msg = "Pairwise DataFrame created and stored in self.pairwise_df"
            if make_splits:
                msg += " (item-level splits applied)"
                # 9a-splits-data: also stamp item-level split labels onto self.data
                self.data["split"] = self.data[self.item_id_name].map(item_to_split)
                print("Split labels added to self.data['split'].")
            print(msg)

        return uuid_pairings

    # ------------------------------------------------------------------
    # Pairwise comparison
    # ------------------------------------------------------------------

    @staticmethod
    def pairwise_compare(
        text1_breakdown: str,
        text2_breakdown: str,
        target_concept: str,
        client: LLMClient,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        allow_ties: bool = False,
        comparison_prompt=None,
        system_message: str = (
            "You are a precise and detail-oriented assistant working to compare "
            "two descriptions based on a specific concept."
        ),
    ) -> Tuple[str, str]:
        """
        Compare two CGCoT breakdowns and return a decision and the raw LLM response.

        This is the core comparison step: it constructs a prompt that shows the
        LLM two concept-focused descriptions side-by-side and asks which one
        better expresses the ``target_concept``.  If the response cannot be
        parsed automatically, a follow-up extraction call is made.

        This is a static method called internally by
        :meth:`generate_pairwise_annotations` but can also be invoked directly
        for debugging individual pairs.

        Parameters
        ----------
        text1_breakdown : str
            The CGCoT breakdown (structured description) for item 1.
        text2_breakdown : str
            The CGCoT breakdown (structured description) for item 2.
        target_concept : str
            The concept being evaluated (e.g. ``'persuasiveness'``).
        client : LLMClient
            The LLM client to use for the comparison call.
        max_tokens : int, default 1000
            Maximum tokens for the LLM response.
        temperature : float, default 0.0
            Sampling temperature.  ``0.0`` = deterministic.
        allow_ties : bool, default False
            If ``True``, ``'Tie'`` is presented as a valid answer option.
        comparison_prompt : str or None, optional
            A custom prompt template overriding the built-in default.  Must
            include ``{text1_breakdown}``, ``{text2_breakdown}``,
            ``'FINAL ANSWER:'``, ``'Description 1'``, and ``'Description 2'``
            placeholders.  If ``allow_ties=True``, must also mention ``'Tie'``.
        system_message : str, optional
            System prompt sent alongside the comparison request.

        Returns
        -------
        tuple of (str, str)
            ``(decision, full_response)`` where ``decision`` is one of
            ``'Text1'``, ``'Text2'``, ``'Tie'``, or an error string prefixed
            with ``'ERROR from pairadigm:'``.  ``full_response`` is the
            complete raw text returned by the LLM.

        Examples
        --------
        >>> decision, response = Pairadigm.pairwise_compare(
        ...     text1_breakdown="Essay 1 uses emotional appeals...",
        ...     text2_breakdown="Essay 2 relies on statistics...",
        ...     target_concept="persuasiveness",
        ...     client=p.client,
        ... )
        >>> print(decision)  # 'Text1' or 'Text2'
        """
        if not allow_ties:
            default_prompt = (
                f"\nDescription 1: {text1_breakdown}\n"
                f"Description 2: {text2_breakdown}\n"
                f"Which expresses greater {target_concept}: Description 1 or Description 2? "
                "You must choose one.\n\n"
                'FINAL ANSWER: <"Description 1" or "Description 2">\n'
                "JUSTIFICATION: <Your CONCISE reasoning>"
            )
        else:
            default_prompt = (
                f"\nDescription 1: {text1_breakdown}\n"
                f"Description 2: {text2_breakdown}\n"
                f"Which expresses greater {target_concept}: Description 1, Description 2, or Tie?\n\n"
                'FINAL ANSWER: <"Description 1", "Description 2", or "Tie">\n'
                "JUSTIFICATION: <Your CONCISE reasoning>"
            )

        if comparison_prompt is None:
            prompt = default_prompt
        else:
            for ph in ("{text1_breakdown}", "{text2_breakdown}"):
                if ph not in comparison_prompt:
                    raise ValueError(
                        f"Custom comparison_prompt must include '{ph}' placeholder."
                    )
            if "FINAL ANSWER:" not in comparison_prompt:
                raise ValueError("Custom comparison_prompt must include 'FINAL ANSWER:' formatting.")
            if "Description 1" not in comparison_prompt or "Description 2" not in comparison_prompt:
                raise ValueError("Custom comparison_prompt must reference both 'Description 1' and 'Description 2'.")
            if allow_ties and "Tie" not in comparison_prompt:
                raise ValueError("When allow_ties=True, prompt must include 'Tie' as an option.")
            prompt = comparison_prompt.format(
                text1_breakdown=text1_breakdown,
                text2_breakdown=text2_breakdown,
                target_concept=target_concept,
            )

        response = client.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        def _extract(resp, is_fallback=False):
            # 1. Robust pattern handling markdown ("**FINAL ANSWER:**"), missing colons, etc.
            pat1 = r"FINAL\s+ANSWER.*?(Description 1|Description 2|Tie)"
            m = re.search(pat1, resp, re.IGNORECASE | re.DOTALL)
            
            # 2. Try the end of the text if no prefix is found
            if not m:
                pat2 = r"(Description 1|Description 2|Tie)[^a-zA-Z0-9]*\Z"
                m = re.search(pat2, resp, re.IGNORECASE)

            # 3. For the fallback response, allow the terms to appear anywhere
            if not m and is_fallback:
                pat3 = r"(Description 1|Description 2|Tie)"
                m = re.search(pat3, resp, re.IGNORECASE)

            if m:
                ans = m.group(1).lower()
                if ans == "description 1":   return "Text1"
                if ans == "description 2":   return "Text2"
                if ans == "tie" and allow_ties: return "Tie"
            return None

        final = _extract(response)
        if final is None:
            ext_prompt = (
                f"In the following response, what was the FINAL ANSWER they gave? "
                f"ONLY REPLY WITH \"Description 1\" or \"Description 2\""
                f"{', or \"Tie\"' if allow_ties else ''}. Response: {response}"
            )
            ext_resp = client.generate(
                prompt=ext_prompt, system_message=system_message,
                temperature=temperature, max_tokens=max_tokens,
            )
            final = _extract(ext_resp, is_fallback=True)
            if final is None:
                final = (
                    f"ERROR from pairadigm: Could not extract answer. "
                    f"Model response: {response}"
                )
        return final, response

    def generate_pairwise_annotations(
        self,
        max_workers: int = 8,
        update_classObject: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        allow_ties: bool = False,
        client_indices: Optional[Union[int, List[int]]] = None,
        comparison_prompt=None,
        system_message: str = (
            "You are a precise and detail-oriented assistant working to compare "
            "two descriptions based on a specific concept."
        ),
    ) -> pd.DataFrame:
        """
        Run pairwise LLM comparisons on every pair in ``self.pairwise_df``.

        For each pair, the LLM is shown the two CGCoT breakdowns and asked
        which item better expresses ``self.target_concept``.  Results are
        written to ``decision`` / ``justification`` columns (or
        ``decision_<model_name>`` / ``justification_<model_name>`` when
        multiple clients are registered).

        Call this after :meth:`generate_pairings` (or after providing a
        paired DataFrame) and :meth:`generate_breakdowns`.

        Parameters
        ----------
        max_workers : int, default 8
            Number of parallel threads.  Increase for speed; decrease if you
            hit API rate limits.
        update_classObject : bool, default True
            If ``True``, overwrites ``self.pairwise_df`` with the annotated
            DataFrame and sets ``self.llm_annotated = True``.
        max_tokens : int, default 1000
            Maximum LLM output tokens per comparison.
        temperature : float, default 0.0
            Sampling temperature.  ``0.0`` = deterministic.
        allow_ties : bool, default False
            If ``True``, the LLM may choose ``'Tie'`` in addition to
            ``'Text1'`` or ``'Text2'``.
        client_indices : int or list of int, optional
            Which client(s) to use by index.  ``None`` = all registered clients.
        comparison_prompt : str or None, optional
            Custom comparison prompt template (see :meth:`pairwise_compare` for
            required placeholders).  ``None`` uses the built-in default.
        system_message : str, optional
            System prompt used for all comparison calls.

        Returns
        -------
        pd.DataFrame
            A copy of ``pairwise_df`` with ``decision`` and ``justification``
            columns added (or ``decision_<model>`` / ``justification_<model>``
            for multi-client runs).

        Examples
        --------
        >>> annotated = p.generate_pairwise_annotations()
        >>> annotated[['item1', 'item2', 'decision']].head()
             item1    item2 decision
        0  essay_1  essay_3    Text1
        ...

        >>> # Using only the second registered client:
        >>> p.generate_pairwise_annotations(client_indices=1)
        """
        if self.pairwise_df is None:
            raise ValueError(
                "No pairwise_df found. Generate pairings with breakdowns first."
            )
            
        # Run cost estimation before execution
        try:
            self.estimate_costs(
                stage="pairwise",
                client_indices=client_indices,
                system_message=system_message,
                comparison_prompt=comparison_prompt,
                expected_pairwise_output_tokens=(max_tokens if max_tokens < 100 else 50)
            )
        except ValueError as e:
            print(f"\nCost estimation skipped: {e}")
            
        clients_to_use = self._resolve_clients(client_indices)
        result_df = self.pairwise_df.copy()

        for client_idx, client in clients_to_use:
            multi = len(self.clients) > 1
            bd1  = f"breakdown1_{self.model_names[client_idx]}" if multi else "breakdown1"
            bd2  = f"breakdown2_{self.model_names[client_idx]}" if multi else "breakdown2"
            dcol = f"decision_{self.model_names[client_idx]}"   if multi else "decision"
            jcol = f"justification_{self.model_names[client_idx]}" if multi else "justification"

            if bd1 not in result_df.columns or bd2 not in result_df.columns:
                raise ValueError(
                    f"Breakdown columns '{bd1}' / '{bd2}' not found. "
                    f"Run generate_breakdowns_from_paired(client_index={client_idx}) first."
                )

            results: Dict = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.pairwise_compare,
                        row[bd1], row[bd2], self.target_concept, client,
                        max_tokens, temperature, allow_ties, comparison_prompt, system_message,
                    ): idx
                    for idx, row in result_df.iterrows()
                }
                mn = self.model_names[client_idx] if multi else "default"
                desc = f"[{mn}] Pairwise comparisons"
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    idx = futures[future]
                    try:
                        dec, just = future.result()
                    except Exception as exc:
                        dec, just = "ERROR", str(exc)
                    results[idx] = (dec, just)

            result_df[dcol] = result_df.index.map(lambda i: results[i][0])
            result_df[jcol] = result_df.index.map(lambda i: results[i][1])
            if dcol not in self.llm_annotator_cols:
                self.llm_annotator_cols.append(dcol)

        if update_classObject:
            self.pairwise_df = result_df
            self.llm_annotated = True

            # Check for any ERROR values in the decision columns and print a message to the user if any are found
            for col in self.llm_annotator_cols:
                if self.pairwise_df[col].astype(str).str.contains("ERROR").any():
                    print(f"\nWARNING: Found ERROR values in column '{col}'. Please review and regenerate annotations.")    

            # 9a-autosave
            if self.save_dir:
                self.save(self.save_dir)
                print(f"Auto-saved to: {self.save_dir}")

        return result_df

    # ------------------------------------------------------------------
    # Human annotations
    # ------------------------------------------------------------------

    def append_human_annotations(
        self,
        annotations: Union[pd.DataFrame, str],
        annotator_names: Union[str, List[str], None] = None,
        item1_col: str = "item1",
        item2_col: str = "item2",
        decision_cols: Optional[Union[str, List[str]]] = None,
        validate_items: bool = True,
        overwrite: bool = False,
    ) -> None:
        """
        Merge human annotation decisions into ``self.pairwise_df``.

        Accepts annotations from a DataFrame or a CSV / Excel file.  The method
        handles both orientations of a pair (item1-item2 and item2-item1) so
        the order in the annotation file does not need to match ``pairwise_df``.

        Decision values are automatically normalised to ``'Text1'`` / ``'Text2'``
        / ``'Tie'``.  Pass ``0`` for ``'Text1'`` and ``1`` for ``'Text2'`` if
        your annotations use integer coding.

        Parameters
        ----------
        annotations : pd.DataFrame or str
            Annotation data.  Either a DataFrame or a file path to a ``.csv``
            or ``.xlsx`` file.
        annotator_names : str or list of str, optional
            Display name(s) for each annotator used as the output column
            name(s) in ``pairwise_df``.  If ``None``, uses the column names
            from ``decision_cols``.
        item1_col : str, default ``'item1'``
            Column in the annotation source that holds the first item ID.
        item2_col : str, default ``'item2'``
            Column in the annotation source that holds the second item ID.
        decision_cols : str or list of str, optional
            Column(s) in the annotations source containing the decisions.
            Auto-detected if ``None`` (looks for columns starting with
            ``'decision'``, ``'annotator'``, or ``'human'``).
        validate_items : bool, default True
            Reserved for future validation logic; currently unused.
        overwrite : bool, default False
            If ``True``, replaces an existing column with the same annotator
            name.  Raises an error otherwise to prevent accidental overwrites.

        Examples
        --------
        **From a DataFrame:**

        >>> human_df = pd.DataFrame({
        ...     'item1': ['essay_1', 'essay_2'],
        ...     'item2': ['essay_3', 'essay_4'],
        ...     'judge_a': ['Text1', 'Text2'],
        ... })
        >>> p.append_human_annotations(
        ...     annotations=human_df,
        ...     decision_cols='judge_a',
        ...     annotator_names='Judge A',
        ... )

        **From a CSV file:**

        >>> p.append_human_annotations('annotations.csv')
        # Auto-detects decision columns
        """
        # Load from file if needed
        if isinstance(annotations, str):
            fp = Path(annotations)
            if not fp.exists():
                raise FileNotFoundError(f"Annotation file not found: {fp}")
            if fp.suffix == ".csv":
                annotations_df = pd.read_csv(fp)
            elif fp.suffix in (".xlsx", ".xls"):
                annotations_df = pd.read_excel(fp)
            else:
                raise ValueError(f"Unsupported file format: {fp.suffix}")
        elif isinstance(annotations, pd.DataFrame):
            annotations_df = annotations.copy()
        else:
            raise TypeError("annotations must be a DataFrame or a filepath string.")

        if self.pairwise_df is None:
            raise ValueError(
                "No pairwise_df found. Generate pairings first."
            )

        # Auto-detect decision columns
        if decision_cols is None:
            cands = [
                c for c in annotations_df.columns
                if c not in (item1_col, item2_col)
                and (c.startswith("decision") or c.startswith("annotator") or c.startswith("human"))
            ]
            if not cands:
                raise ValueError("No decision columns found. Specify decision_cols.")
            # Warn about columns using reserved prefixes
            reserved_prefix_cols = [
                c for c in cands
                if c.startswith("decision") or c.startswith("annotator")
            ]
            if reserved_prefix_cols:
                warnings.warn(
                    f"Auto-detected columns with reserved prefixes: {reserved_prefix_cols}. "
                    "In pairadigm, 'decision_' is reserved for LLM annotations and "
                    "'annotator_' for manual annotations. Verify these columns are "
                    "assigned to the correct annotator type.",
                    UserWarning,
                    stacklevel=2,
                )
            decision_cols = cands
            print(f"Auto-detected decision columns: {decision_cols}")

        if isinstance(decision_cols, str):
            decision_cols = [decision_cols]

        # Normalise annotator_names
        if annotator_names is None:
            annotator_names = decision_cols
        elif isinstance(annotator_names, str):
            annotator_names = [annotator_names]

        if len(annotator_names) != len(decision_cols):
            raise ValueError(
                f"annotator_names ({len(annotator_names)}) and decision_cols "
                f"({len(decision_cols)}) must have the same length."
            )

        for dcol, aname in zip(decision_cols, annotator_names):
            if dcol not in annotations_df.columns:
                raise ValueError(f"Column '{dcol}' not found in annotations.")
            if aname in self.pairwise_df.columns and not overwrite:
                raise ValueError(
                    f"Annotator '{aname}' already exists. Set overwrite=True to replace."
                )

            # Build forward mapping using vectorised merge (fix 5b)
            ann_sub = annotations_df[[item1_col, item2_col, dcol]].copy()
            ann_sub = ann_sub.rename(columns={item1_col: "item1", item2_col: "item2", dcol: aname})

            # Fix 1f: standardise decision type — normalise to string 'Text1'/'Text2'
            def _normalise_decision(val):
                if pd.isna(val):
                    return None
                if val in ("Text1", 0):
                    return "Text1"
                if val in ("Text2", 1):
                    return "Text2"
                return val  # Tie or other

            ann_sub[aname] = ann_sub[aname].apply(_normalise_decision)

            # Also build reversed rows so both orientations are captured
            rev = ann_sub.copy()
            rev["item1"], rev["item2"] = ann_sub["item2"].copy(), ann_sub["item1"].copy()
            rev[aname] = ann_sub[aname].map(
                {"Text1": "Text2", "Text2": "Text1"}
            ).fillna(ann_sub[aname])

            combined = pd.concat([ann_sub, rev], ignore_index=True)

            lookup = combined.set_index(["item1", "item2"])[aname].to_dict()
            self.pairwise_df[aname] = self.pairwise_df.apply(
                lambda r: lookup.get((r["item1"], r["item2"])), axis=1
            )

            if not self.annotated:
                self.annotated = True
                self.annotator_cols = [aname]
            elif aname not in self.annotator_cols:
                self.annotator_cols.append(aname)

            non_null = self.pairwise_df[aname].notna().sum()
            total    = len(self.pairwise_df)
            print(
                f"Uploaded annotations for '{aname}': "
                f"{non_null}/{total} pairs ({non_null / total * 100:.1f}%)"
            )

        if self.item_id_cols is None:
            self.item_id_cols = ["item1", "item2"]

        print(f"\nHuman-annotated status: {self.annotated}")
        print(f"Total annotators: {len(self.annotator_cols)}")

    # ------------------------------------------------------------------
    # Scoring (delegates to scoring.py)
    # ------------------------------------------------------------------

    def score_items(
        self,
        normalization_scale: Union[str, Tuple] = "zero-to-one",
        update_classObject: bool = True,
        summarize: bool = True,
        decision_col: str = "decision",
        use_davidson: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Compute Bradley-Terry (or Davidson) scores from pairwise comparison results.

        Fits a Bradley-Terry model to the win/loss records in ``pairwise_df``
        and returns a scored DataFrame where each item has a numerical score
        representing its relative strength on the target concept.  Scores are
        normalised so they are easy to interpret.

        Call this after :meth:`generate_pairwise_annotations`.

        Parameters
        ----------
        normalization_scale : str or tuple, default ``'zero-to-one'``
            How to normalise raw BT scores.  Options:

            * ``'zero-to-one'`` — rescales scores to the [0, 1] interval.
            * ``'z-score'``     — standardises to mean 0, standard deviation 1.
            * A tuple ``(min, max)`` — rescales to a custom interval.
        update_classObject : bool, default True
            If ``True``, stores the result in ``self.scored_df``.
        summarize : bool, default True
            If ``True``, prints a brief score summary table to the console.
        decision_col : str, default ``'decision'``
            Name of the column in ``pairwise_df`` containing LLM decisions
            (``'Text1'`` / ``'Text2'`` / ``'Tie'``).
        use_davidson : bool or None, optional
            If ``True``, fits the Davidson model (which handles ties
            explicitly).  If ``None`` (default), auto-detects by checking
            whether any ``'Tie'`` labels are present.

        Returns
        -------
        pd.DataFrame
            Item-level DataFrame with at least a ``Bradley_Terry_Score`` column
            (and ``Davidson_Score`` if ties are present).

        Examples
        --------
        >>> scored = p.score_items()
        >>> scored[['essay_id', 'Bradley_Terry_Score']].sort_values(
        ...     'Bradley_Terry_Score', ascending=False
        ... ).head()
        """
        result = _sc.score_items(
            pairwise_df=self.pairwise_df,
            data=self.data,
            item_id_name=self.item_id_name,
            target_concept=self.target_concept,
            text_name=self.text_name,
            paired=self.paired,
            item_id_cols=self.item_id_cols,
            scored_df=self.scored_df,
            normalization_scale=normalization_scale,
            summarize=summarize,
            decision_col=decision_col,
            use_davidson=use_davidson,
        )
        if update_classObject:
            self.scored_df = result

        # Save scored dataframe to parquet
        if self.save_dir:
            self.save(self.save_dir)
            print(f"Auto-saved to: {self.save_dir}")   

        return result

    def summarize_scores(
        self,
        df=None,
        text_col=None,
        score_col: str = "Bradley_Terry_Score",
    ) -> dict:
        """
        Print and return descriptive statistics for a score column.

        Parameters
        ----------
        df : pd.DataFrame or None, optional
            DataFrame to summarise.  Defaults to ``self.scored_df``.
        text_col : str or None, optional
            Column with item text, used to show top/bottom examples.  Defaults
            to ``self.text_name``.
        score_col : str, default ``'Bradley_Terry_Score'``
            The score column to summarise.

        Returns
        -------
        dict
            Summary statistics dictionary (mean, std, min, max, quartiles).

        Examples
        --------
        >>> p.summarize_scores()
        >>> p.summarize_scores(score_col='Davidson_Score')
        """
        if df is None:
            if self.scored_df is None:
                raise ValueError("No scored DataFrame. Run score_items() first.")
            df = self.scored_df
        if text_col is None:
            if self.text_name is None:
                raise ValueError("Provide text_col or set text_name in the constructor.")
            text_col = self.text_name
        return _sc.summarize_scores(
            df=df, target_concept=self.target_concept,
            score_col=score_col, text_col=text_col,
        )

    # ------------------------------------------------------------------
    # Validation (delegates to validation.py)
    # ------------------------------------------------------------------

    def prep_for_alt_test(
        self,
        decision_col: Optional[str] = None,
        # deprecated alias
        llm_decision_col: Optional[str] = None,
    ) -> Tuple[Dict, Dict]:
        if llm_decision_col is not None and decision_col is None:
            warnings.warn("llm_decision_col is deprecated; use decision_col.", DeprecationWarning, 2)
            decision_col = llm_decision_col
        if not self.annotated:
            raise ValueError("Data must have human annotations to run the alt_test.")
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison data found.")
        return _val.prep_for_alt_test(
            pairwise_df=self.pairwise_df,
            annotator_cols=self.annotator_cols,
            item_id_cols=self.item_id_cols,
            decision_col=decision_col,
        )

    def alt_test(self, **kwargs) -> Union[Tuple, Dict]:
        """
        Perform the Alternative Annotator Test (AltTest).

        The AltTest checks whether the LLM annotator(s) perform at least as
        well as a human annotator when predicting the decisions of other human
        annotators.  A "win" means the LLM achieves a higher agreement score
        than the comparison threshold (epsilon).

        Requires at least one human annotation column (``annotator_cols``) and
        at least one LLM annotation column (``llm_annotator_cols``) to be
        present in ``pairwise_df``.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`pairadigm.validation.alt_test`.  Commonly used
            keyword arguments include:

            * ``scoring_function`` (str, default ``'accuracy'``) — metric used
              to compare annotators (``'accuracy'``, ``'kappa'``, etc.).
            * ``epsilon`` (float, default ``0.0``) — tolerance margin; the LLM
              wins if its score ≥ human score − epsilon.
            * ``q_fdr`` (float, default ``0.05``) — FDR threshold for multiple
              comparisons.
            * ``test_all_llms`` (bool, default ``True``) — whether to test
              every registered LLM client.

        Returns
        -------
        tuple or dict
            Test results including win rates and p-values.

        Examples
        --------
        >>> p.alt_test(scoring_function='accuracy', epsilon=0.1, q_fdr=0.05)
        """
        pairwise_df = kwargs.pop("pairwise_df", self.pairwise_df)
        annotator_cols = kwargs.pop("annotator_cols", self.annotator_cols)
        item_id_cols = kwargs.pop("item_id_cols", self.item_id_cols)
        annotated = kwargs.pop("annotated", self.annotated)

        return _val.alt_test(
            pairwise_df=pairwise_df,
            annotator_cols=annotator_cols,
            item_id_cols=item_id_cols,
            annotated=annotated,
            **kwargs,
        )

    def dawid_skene_alt_test(self, **kwargs) -> Union[Dict, Dict]:
        """
        AltTest variant that uses Dawid-Skene latent-class agreement scores.

        Instead of raw pairwise agreement, this method estimates annotator
        error rates via the Dawid-Skene probabilistic model and uses those
        modelled scores for the alternative annotator test.  This can be more
        robust when annotators have variable reliability or sparse overlap.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`pairadigm.validation.dawid_skene_alt_test`.

        Returns
        -------
        dict
            Test results mirroring the standard AltTest output.

        Examples
        --------
        >>> p.dawid_skene_alt_test()
        """
        pairwise_df = kwargs.pop("pairwise_df", self.pairwise_df)
        annotator_cols = kwargs.pop("annotator_cols", self.annotator_cols)
        annotated = kwargs.pop("annotated", self.annotated)

        return _val.dawid_skene_alt_test(
            pairwise_df=pairwise_df,
            annotator_cols=annotator_cols,
            annotated=annotated,
            **kwargs,
        )

    def dawid_skene_annotator_ranking(self, **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, np.ndarray]]]:
        """
        Rank annotators by Dawid-Skene reliability.

        Parameters
        ----------
        pairwise_df : pd.DataFrame, optional
            DataFrame containing pairwise comparison annotations. Defaults to ``self.pairwise_df``.
        llm_annotated : bool, optional
            Whether the data contains LLM annotations. Defaults to ``self.llm_annotated``.
        human_annotated : bool, optional
            Whether the data contains human annotations. Defaults to ``self.annotated``.
        annotator_cols : Optional[List[str]], optional
            List of column names containing human annotations. Defaults to ``self.annotator_cols``.
        llm_annotator_cols : Optional[List[str]], optional
            List of column names containing LLM annotations. Defaults to ``self.llm_annotator_cols``.
        **kwargs
            Additional keyword arguments to pass to the Dawid-Skene algorithm.

        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, Dict[str, np.ndarray]]
            DataFrame containing annotator reliability rankings. If 
            ``return_confusion_matrices`` is True, also returns a dictionary 
            mapping annotator names to their confusion matrices.
        """
        # Pop potential duplicates to avoid TypeError
        pairwise_df = kwargs.pop("pairwise_df", self.pairwise_df)
        llm_annotated = kwargs.pop("llm_annotated", self.llm_annotated)
        human_annotated = kwargs.pop("human_annotated", self.annotated)
        annotator_cols = kwargs.pop("annotator_cols", self.annotator_cols)
        llm_annotator_cols = kwargs.pop("llm_annotator_cols", self.llm_annotator_cols)

        return _val.dawid_skene_annotator_ranking(
            pairwise_df=pairwise_df,
            llm_annotated=llm_annotated,
            human_annotated=human_annotated,
            annotator_cols=annotator_cols,
            llm_annotator_cols=llm_annotator_cols,
            **kwargs,
        )

    def check_transitivity(self, annotator_cols=None) -> Dict:
        """
        Check for transitivity violations across all annotators.

        A transitivity violation occurs when annotator decisions are
        inconsistent — e.g. A > B, B > C, but C > A.  High violation rates
        may indicate that the target concept is hard to rank, that the prompts
        need refinement, or that a particular annotator is unreliable.

        Parameters
        ----------
        annotator_cols : list of str or None, optional
            Specific annotator columns to check.  If ``None``, checks all
            human and LLM annotator columns registered on the object.

        Returns
        -------
        dict
            ``{annotator_name: {'n_violations': int, 'rate': float, ...}}``
            for each annotator.

        Examples
        --------
        >>> p.check_transitivity()
        {'decision': {'n_violations': 3, 'rate': 0.04, ...}}
        """
        return _val.check_transitivity(
            pairwise_df=self.pairwise_df,
            annotator_cols=self.annotator_cols,
            llm_annotator_cols=self.llm_annotator_cols,
            annotated=self.annotated,
            llm_annotated=self.llm_annotated,
            all_annotator_cols=annotator_cols,
        )

    def irr(
        self,
        method: str = "auto",
        alpha_level: str = "nominal",
        min_overlap: int = 2,
    ) -> pd.DataFrame:
        """
        Compute pairwise inter-rater reliability (IRR) between all annotators.

        Compares every pair of annotators (human and/or LLM) and returns a
        reliability metric such as Cohen's Kappa or Krippendorff's Alpha.

        Parameters
        ----------
        method : str, default ``'auto'``
            IRR metric to compute:

            * ``'auto'``        — selects Kappa for two annotators, Alpha otherwise.
            * ``'kappa'``       — Cohen's Kappa (pairwise).
            * ``'alpha'``       — Krippendorff's Alpha (multi-annotator).
            * ``'percentage'``  — simple percentage agreement.
        alpha_level : str, default ``'nominal'``
            Measurement level for Krippendorff's Alpha: ``'nominal'``,
            ``'ordinal'``, or ``'ratio'``.
        min_overlap : int, default 2
            Minimum number of pairs that two annotators must both have rated
            to be included in the IRR calculation.

        Returns
        -------
        pd.DataFrame
            Matrix of IRR scores between annotator pairs.

        Examples
        --------
        >>> p.irr()
        >>> p.irr(method='kappa')
        """
        return _val.irr(
            pairwise_df=self.pairwise_df,
            annotator_cols=self.annotator_cols,
            llm_annotator_cols=self.llm_annotator_cols,
            annotated=self.annotated,
            llm_annotated=self.llm_annotated,
            method=method,
            alpha_level=alpha_level,
            min_overlap=min_overlap,
        )

    # ------------------------------------------------------------------
    # Visualisation (delegates to visualization.py)
    # ------------------------------------------------------------------

    def plot_score_distribution(self, score_col="Bradley_Terry_Score", **kwargs):
        """
        Plot an interactive histogram of item scores.

        Displays the distribution of Bradley-Terry (or Davidson) scores across
        all items.  Rendered as an interactive Plotly chart.

        Parameters
        ----------
        score_col : str, default ``'Bradley_Terry_Score'``
            The score column in ``self.scored_df`` to visualise.
        **kwargs
            Extra arguments forwarded to the underlying Plotly figure builder
            (e.g. ``nbins``, ``title``).

        Examples
        --------
        >>> p.plot_score_distribution()
        >>> p.plot_score_distribution(score_col='Davidson_Score')
        """
        return _viz.plot_score_distribution(
            scored_df=self.scored_df,
            target_concept=self.target_concept,
            score_col=score_col,
            **kwargs,
        )

    def plot_comparison_network(
        self,
        centrality_measure: str = "pagerank",
        decision_col: str = "decision",
        return_fig: bool = False,
        **kwargs,
    ):
        """
        Plot the directed pairwise-comparison network.

        Each node is an item; directed edges point from the winner to the loser
        of each comparison.  Node size and colour reflect the chosen centrality
        measure (or BT score).  Hover over nodes to see item text.
        Rendered as an interactive Plotly chart.

        Parameters
        ----------
        centrality_measure : str, default ``'pagerank'``
            Graph-theoretic measure used to size nodes.  Options:

            * ``'pagerank'``    — PageRank centrality.
            * ``'out_degree'``  — number of wins (recommended for ranking).
            * ``'in_degree'``   — number of losses.
            * ``'betweenness'`` — betweenness centrality.
        decision_col : str, default ``'decision'``
            Column in ``pairwise_df`` containing comparison decisions.
        return_fig : bool, default False
            If ``True``, returns the Plotly Figure object instead of displaying
            it directly (useful for embedding in notebooks or dashboards).
        **kwargs
            Additional arguments passed to the underlying plotting function,
            e.g. ``scored_df``, ``text_col``, ``item_id_name``.

        Examples
        --------
        >>> p.plot_comparison_network()
        >>> fig = p.plot_comparison_network(
        ...     centrality_measure='out_degree',
        ...     return_fig=True,
        ... )
        """
        kwargs.setdefault("scored_df", self.scored_df)
        kwargs.setdefault("item_id_name", self.item_id_name)
        kwargs.setdefault("data_df", self.data)
        kwargs.setdefault("text_col", self.text_name)

        return _viz.plot_comparison_network(
            pairwise_df=self.pairwise_df,
            target_concept=self.target_concept,
            centrality_measure=centrality_measure,
            decision_col=decision_col,
            return_fig=return_fig,
            **kwargs,
        )

    def plot_epsilon_sensitivity(self, **kwargs):
        """
        Plot the AltTest win rate across a range of epsilon values.

        Epsilon controls the tolerance margin in the Alternative Annotator Test.
        This plot helps researchers choose an appropriate epsilon by showing how
        the LLM win rate changes as the threshold becomes more or less strict.
        Rendered as an interactive Plotly chart.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`pairadigm.visualization.plot_epsilon_sensitivity`.
            Commonly used arguments:

            * ``epsilon_range`` (tuple, default ``(-0.1, 0.25)``) — range of
              epsilon values to sweep.
            * ``epsilon_step`` (float, default ``0.02``) — step between values.
            * ``test_all_llms`` (bool, default ``True``) — plot a line for
              each registered LLM client.
            * ``return_data`` (bool, default ``False``) — if ``True``, also
              returns the underlying data table.

        Examples
        --------
        >>> p.plot_epsilon_sensitivity()
        >>> p.plot_epsilon_sensitivity(
        ...     epsilon_range=(-0.05, 0.3),
        ...     epsilon_step=0.01,
        ... )
        """
        pairwise_df = kwargs.pop("pairwise_df", self.pairwise_df)
        annotator_cols = kwargs.pop("annotator_cols", self.annotator_cols)
        item_id_cols = kwargs.pop("item_id_cols", self.item_id_cols)
        annotated = kwargs.pop("annotated", self.annotated)
        llm_annotator_cols = kwargs.pop("llm_annotator_cols", self.llm_annotator_cols)

        return _viz.plot_epsilon_sensitivity(
            pairwise_df=pairwise_df,
            annotator_cols=annotator_cols,
            item_id_cols=item_id_cols,
            annotated=annotated,
            llm_annotator_cols=llm_annotator_cols,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Classification (9c)
    # ------------------------------------------------------------------

    def classify(
        self,
        score_col: Optional[str] = None,
        method: str = "kmeans",
        n_clusters: int = 3,
        output_col: Optional[str] = "kmeans_clusters",
        random_state: int = 42,
        update_classObject: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Classify items into discrete categories based on BT scores.

        Parameters
        ----------
        score_col : str or None
            Score column to cluster.  Auto-detected if ``None``.
        method : str, default 'kmeans'
            Clustering method: ``'kmeans'``, ``'gmm'``, or ``'hdbscan'``.
        n_clusters : int, default 3
            Number of clusters (ignored by hdbscan).
        output_col : str or None
            Column name for the cluster labels.  Defaults to ``'kmeans_clusters'`` to match the deault method.
        random_state : int
            Random seed.
        update_classObject : bool, default True
            Whether to update the class object with the new cluster labels.
            If True, the class object will be updated with the new cluster labels.
            If False, the class object will not be updated with the new cluster labels.
        **kwargs
            Extra arguments forwarded to the underlying clusterer.

        Returns
        -------
        pd.DataFrame
            A copy of ``scored_df`` with a ``cluster`` column added.
        """
        if self.scored_df is None:
            raise ValueError("No scored_df. Run score_items() first.")

        # Auto-detect score column
        if score_col is None:
            full_cols  = [c for c in self.scored_df.columns if c.endswith("_Score_full")]
            plain_cols = [c for c in self.scored_df.columns
                         if c.endswith("_Score") and not c.endswith("_Score_split")
                         and not c.endswith("_Score_full")]
            score_col = (full_cols or plain_cols or [None])[0]
            if score_col is None:
                raise ValueError("Could not auto-detect a score column.")

        if score_col not in self.scored_df.columns:
            raise ValueError(f"Column '{score_col}' not found in scored_df.")

        X = self.scored_df[[score_col]].dropna().values
        valid_idx = self.scored_df[score_col].notna()

        if method == "kmeans":
            out_col = "kmeans_clusters"
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto", **kwargs)
            labels = model.fit_predict(X)
        elif method == "gmm":
            out_col = "gmm_clusters"
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=n_clusters, random_state=random_state, **kwargs)
            labels = model.fit_predict(X)
        elif method == "hdbscan":
            try:
                from hdbscan import HDBSCAN
            except ImportError:
                raise ImportError(
                    "hdbscan is not installed. Install it with: pip install hdbscan"
                )
            out_col = "hdbscan_clusters"
            model = HDBSCAN(**kwargs)
            labels = model.fit_predict(X)
        else:
            raise ValueError(f"Unknown method: '{method}'. Choose from 'kmeans', 'gmm', 'hdbscan'.")

        out_col = output_col or out_col

        result = self.scored_df.copy()
        result.loc[valid_idx, out_col] = labels.astype(int)
        result.loc[~valid_idx, out_col] = -1  # NaN score items -> noise

        # Sort cluster labels by mean score (low score = cluster 0)
        cluster_means = result.dropna(subset=[score_col]).groupby(out_col)[score_col].mean()
        rank_map = {c: rank for rank, c in enumerate(cluster_means.sort_values().index)}
        result[out_col] = result[out_col].map(lambda x: rank_map.get(x, x))

        n_clusters_found = result[out_col].nunique()
        print(f"Classified {len(result)} items into {n_clusters_found} clusters "
              f"using {method} on '{score_col}'.")
        print(result.groupby(out_col)[score_col].describe().round(3).to_string())

        # Update scored_df if requested
        if update_classObject:
            self.scored_df = result

            # Save updated scored dataframe to parquet
            if self.save_dir:
                self.save(self.save_dir)
                print(f"Auto-saved updated scored dataframe with cluster assignments to: {self.save_dir}")   

        return result

    # ------------------------------------------------------------------
    # ICC validation (9d)
    # ------------------------------------------------------------------

    def icc(self, **kwargs) -> pd.DataFrame:
        """
        Compute Intraclass Correlation Coefficients (ICC) between annotators.

        ICC is especially useful when annotation decisions can be treated as
        continuous or ordinal values.  It quantifies both the consistency
        (relative ordering) and absolute agreement between raters.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`pairadigm.validation.icc`.  Common options
            include ``icc_type`` (e.g. ``'ICC(2,1)'``, ``'ICC(3,1)'``).

        Returns
        -------
        pd.DataFrame
            ICC estimates and confidence intervals for each annotator pair.

        Examples
        --------
        >>> p.icc()
        """
        pairwise_df = kwargs.pop("pairwise_df", self.pairwise_df)
        annotator_cols = kwargs.pop("annotator_cols", self.annotator_cols)
        llm_annotator_cols = kwargs.pop("llm_annotator_cols", self.llm_annotator_cols)
        annotated = kwargs.pop("annotated", self.annotated)
        llm_annotated = kwargs.pop("llm_annotated", self.llm_annotated)

        return _val.icc(
            pairwise_df=pairwise_df,
            annotator_cols=annotator_cols,
            llm_annotator_cols=llm_annotator_cols,
            annotated=annotated,
            llm_annotated=llm_annotated,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Additions to a Pairadigm (items or clients) (9a-add_items)
    # ------------------------------------------------------------------

    def add_items(
        self,
        new_data: pd.DataFrame,
        max_workers: int = 8,
        rate_limit_per_minute: Optional[int] = None,
        num_pairs_per_item: int = 10,
        random_seed: int = 42,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        allow_ties: bool = False,
        client_indices: Optional[Union[int, List[int]]] = None,
        debug_mode: bool = False,
        rescore: bool = True,
        normalization_scale: Union[str, Tuple] = "zero-to-one",
    ) -> None:
        """
        Add new items to an existing Pairadigm dataset, generate their
        breakdowns, pair them with existing items, annotate, and optionally
        rescore.

        Parameters
        ----------
        new_data : pd.DataFrame
            DataFrame with the same schema as ``self.data``.
        max_workers : int
        rate_limit_per_minute : int or None
        num_pairs_per_item : int
            Requested pairings for each new item against existing items.
        random_seed : int
        max_tokens : int
        temperature : float
        allow_ties : bool
        client_indices : int, list of int, or None
        debug_mode : bool
            Passed to breakdown generation.
        rescore : bool, default True
            If True, runs ``score_items()`` after annotation to refresh scores.
        normalization_scale : str or tuple
            Passed to ``score_items()`` if ``rescore=True``.
        """
        if self.paired:
            raise ValueError("add_items() is not supported for paired data.")
        if self.item_id_name not in new_data.columns:
            raise ValueError(f"new_data must have an '{self.item_id_name}' column.")
        if self.text_name is not None and self.text_name not in new_data.columns:
            raise ValueError(f"new_data must have a '{self.text_name}' column.")

        # Detect duplicates
        existing_ids = set(self.data[self.item_id_name])
        new_ids = set(new_data[self.item_id_name])
        overlap = existing_ids & new_ids
        if overlap:
            raise ValueError(
                f"{len(overlap)} duplicate item IDs found: {list(overlap)[:5]}... "
                "Remove duplicates before calling add_items()."
            )

        print(f"Adding {len(new_data)} new items to {len(self.data)} existing items...")

        # 1. Append to self.data
        self.data = pd.concat([self.data, new_data], ignore_index=True)

        # 2. Generate breakdowns for new items only
        if self.cgcot_prompts:
            print("\n[Step 1/3] Generating breakdowns for new items...")
            new_breakdowns = _bd.generate_breakdowns(
                data=new_data,
                item_id_name=self.item_id_name,
                text_name=self.text_name,
                cgcot_prompts=self.cgcot_prompts,
                clients=self.clients,
                model_names=self.model_names,
                client_indices=client_indices,
                max_workers=max_workers,
                rate_limit_per_minute=rate_limit_per_minute,
                max_tokens=max_tokens,
                temperature=temperature,
                debug_mode=debug_mode,
            )
            # Write breakdowns back to self.data
            bd_col = "CGCoT_Breakdown"
            if bd_col in self.data.columns:
                new_bd_series = new_data[self.item_id_name].map(
                    new_breakdowns if isinstance(new_breakdowns, dict)
                    else new_breakdowns
                )
                self.data.loc[self.data[self.item_id_name].isin(new_ids), bd_col] = (
                    new_bd_series.values
                )

        # 3. Generate pairings for new items against ALL existing items
        print("\n[Step 2/3] Generating pairings for new items...")
        new_pairings = self.generate_pairings(
            num_pairs_per_item=num_pairs_per_item,
            random_seed=random_seed,
            breakdowns=True,
            update_classObject=False,  # We'll merge manually
        )
        # Filter to only pairs involving at least one new item
        new_pair_mask = (
            new_pairings["item1"].isin(new_ids) |
            new_pairings["item2"].isin(new_ids)
        )
        new_pairs = new_pairings[new_pair_mask].copy()

        if self.pairwise_df is not None:
            self.pairwise_df = pd.concat(
                [self.pairwise_df, new_pairs], ignore_index=True
            )
        else:
            self.pairwise_df = new_pairs

        # 4. Annotate the new pairs
        print("\n[Step 3/3] Annotating new pairings...")
        self.generate_pairwise_annotations(
            max_workers=max_workers,
            update_classObject=True,
            max_tokens=max_tokens,
            temperature=temperature,
            allow_ties=allow_ties,
            client_indices=client_indices,
        )

        # 5. Optionally rescore
        if rescore:
            print("\nRescoring all items...")
            self.score_items(
                normalization_scale=normalization_scale,
                update_classObject=True,
                summarize=True,
            )

        print(f"\nadd_items() complete. Total items: {len(self.data)}.")

    def add_client(self, client: LLMClient) -> None:
        """
        Add a new :class:`~pairadigm.client.LLMClient` to this object.

        Call this after construction if you want to add a second (or third)
        LLM to use for ensemble annotation.

        Parameters
        ----------
        client : LLMClient
            A pre-initialised ``LLMClient`` instance to append.

        Examples
        --------
        >>> from pairadigm.client import LLMClient
        >>> gpt = LLMClient(api_key='sk-...', model_name='gpt-4o')
        >>> p.add_client(gpt)
        # Added client: gpt-4o (openai)
        """
        if not isinstance(client, LLMClient):
            raise TypeError("client must be an instance of LLMClient.")
        self.clients.append(client)
        self.model_names.append(client.model_name)
        print(f"Added client: {client.model_name} ({client.provider})")

        if getattr(self, "pairwise_df", None) is not None and self.llm_annotator_cols:
            has_decisions = any(c in self.pairwise_df.columns for c in self.llm_annotator_cols)
            if has_decisions:
                ans = input(
                    f"\nDo you want to generate breakdowns and pair annotations "
                    f"for the new client '{client.model_name}'? (y/n): "
                ).strip().lower()
                if ans in ["y", "yes"]:
                    new_idx = len(self.clients) - 1
                    print(f"\nGenerating breakdowns for '{client.model_name}'...")
                    if self.paired:
                        self.generate_breakdowns_from_paired(client_indices=new_idx)
                    else:
                        self.generate_breakdowns(client_indices=new_idx)
                        # Map the newly generated column to pairwise_df
                        bd_col = f"CGCoT_Breakdown_{client.model_name}" if len(self.clients) > 1 else "CGCoT_Breakdown"
                        uuid_to_desc = dict(zip(self.data[self.item_id_name], self.data[bd_col]))
                        bd1 = f"breakdown1_{client.model_name}" if len(self.clients) > 1 else "breakdown1"
                        bd2 = f"breakdown2_{client.model_name}" if len(self.clients) > 1 else "breakdown2"
                        self.pairwise_df[bd1] = self.pairwise_df["item1"].map(uuid_to_desc)
                        self.pairwise_df[bd2] = self.pairwise_df["item2"].map(uuid_to_desc)
                    print(f"\nGenerating pairwise annotations for '{client.model_name}'...")
                    self.generate_pairwise_annotations(client_indices=new_idx)

    # ------------------------------------------------------------------
    # Persistence (delegates to persistence.py)
    # ------------------------------------------------------------------

    def save(self, save_dir: str = None) -> None:
        """
        Save the current state to a structured directory.

        Writes metadata (JSON), data tables (Parquet), and configuration so
        the object can be fully restored later via :func:`load_pairadigm`.  If
        a ``save_dir`` was set at construction time it is used as the default.

        Parameters
        ----------
        save_dir : str, optional
            Path to the output directory.  Created if it does not exist.
            Defaults to ``self.save_dir`` if already set.

        Examples
        --------
        >>> p.save('my_project/pairadigm_output')

        >>> # If save_dir was set at construction time:
        >>> p.save()  # uses self.save_dir
        """
        if save_dir is None:
            if self.save_dir is None:
                raise ValueError("No save directory specified.")
            save_dir = self.save_dir
        self.save_dir = save_dir
        _persist.save_pairadigm(self, save_dir)


################################
# Module-level convenience functions
################################

# pair_items as a module-level alias (fix 3c)
pair_items = Pairadigm.pair_items


def load_pairadigm(
    save_dir: str, 
    api_keys: Optional[Union[str, List[str]]] = None,
    base_urls: Optional[Union[str, List[str]]] = None
) -> Pairadigm:
    """
    Load a saved :class:`Pairadigm` object from a structured directory.

    Reconstructs the full object — including data, pairwise annotations,
    scores, and client configuration — from files written by
    :meth:`Pairadigm.save`.

    Parameters
    ----------
    save_dir : str
        Path to the directory created by :meth:`Pairadigm.save` or
        :func:`pairadigm.persistence.save_pairadigm`.
    api_keys : str or list of str, optional
        API key(s) to assign to the loaded LLM client(s).  The number of keys
        must match the number of model names stored in the saved metadata.
        Pass ``None`` if your keys are set via environment variables.
    base_urls : str or list of str, optional
        Base URL(s) to assign to the loaded LLM client(s). The number of
        base URLs must match the number of model names.

    Returns
    -------
    Pairadigm
        A fully reconstructed :class:`Pairadigm` instance.

    Examples
    --------
    >>> from pairadigm import load_pairadigm
    >>> p = load_pairadigm('my_project/pairadigm_output', api_keys='YOUR_KEY')

    >>> # Multiple models:
    >>> p = load_pairadigm('output/', api_keys=['KEY_A', 'KEY_B'])
    """
    return _persist.load_pairadigm(save_dir, api_keys, base_urls)


def build_pairadigm(
    pairadigm_obj: Pairadigm,
    num_pairs_per_item: int = 10,
    random_seed: int = 42,
    max_workers: int = 8,
    rate_limit_per_minute: Optional[int] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    allow_ties: bool = False,
    normalization_scale: str = "zero-to-one",
    client_indices: Optional[Union[int, List[int]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete Pairadigm pipeline in a single call.

    Convenience function that chains together all major pipeline steps:
    breakdowns → pairings → LLM annotations → (optional) validation.
    Ideal for first-time users or quick exploratory analyses.

    For more control over individual steps (e.g. custom pairings, multiple
    models, human annotation upload), call the :class:`Pairadigm` methods
    directly.

    Parameters
    ----------
    pairadigm_obj : Pairadigm
        A fully initialised :class:`Pairadigm` instance.
    num_pairs_per_item : int, default 10
        Number of comparison partners per item for the pairing step.
    random_seed : int, default 42
        Random seed for reproducible pairings.
    max_workers : int, default 8
        Thread-pool size for parallel LLM calls.
    rate_limit_per_minute : int or None, optional
        API rate-limit cap.  ``None`` = no hard limit.
    max_tokens : int, default 1000
        Maximum tokens per LLM call.
    temperature : float, default 0.0
        Sampling temperature.  ``0.0`` = deterministic.
    allow_ties : bool, default False
        Whether the LLM may return ``'Tie'`` during comparisons.
    normalization_scale : str, default ``'zero-to-one'``
        Score normalisation method passed to :meth:`~Pairadigm.score_items`.
    client_indices : int or list of int, optional
        Which LLM client(s) to use.  ``None`` = all.
    verbose : bool, default True
        Print step-by-step progress banners.

    Returns
    -------
    dict
        Results dictionary with keys ``'breakdowns'``, ``'pairings'``,
        ``'annotations'``, and (if human annotations exist)
        ``'alt_test'``, ``'transitivity'``, ``'irr'``, ``'epsilon_sensitivity'``.

    Examples
    --------
    >>> from pairadigm import Pairadigm, build_pairadigm
    >>> p = Pairadigm(
    ...     data=df, item_id_name='essay_id', text_name='text',
    ...     cgcot_prompts=prompts, target_concept='persuasiveness',
    ...     api_key='YOUR_KEY',
    ... )
    >>> results = build_pairadigm(p, num_pairs_per_item=8)
    >>> scored = p.scored_df  # scores are stored on the object
    """
    if not isinstance(pairadigm_obj, Pairadigm):
        raise TypeError("pairadigm_obj must be a Pairadigm instance.")

    results: Dict[str, Any] = {}

    # Step 1: Breakdowns
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 1: GENERATING CGCOT BREAKDOWNS")
        print("=" * 70)

    try:
        if pairadigm_obj.paired:
            bd_res = pairadigm_obj.generate_breakdowns_from_paired(
                max_workers=max_workers,
                rate_limit_per_minute=rate_limit_per_minute,
                update_pairwise_df=True,
                max_tokens=max_tokens,
                temperature=temperature,
                client_indices=client_indices,
            )
        else:
            bd_res = pairadigm_obj.generate_breakdowns(
                max_workers=max_workers,
                rate_limit_per_minute=rate_limit_per_minute,
                update_dataframe=True,
                max_tokens=max_tokens,
                temperature=temperature,
                client_indices=client_indices,
                show_progress=verbose,
            )
        results["breakdowns"] = bd_res
        if verbose:
            print("✓ Breakdowns generated successfully")
    except Exception as exc:
        raise RuntimeError(f"Failed to generate breakdowns: {exc}") from exc

    # Step 2: Pairings (only for unpaired data)
    if not pairadigm_obj.paired:
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 2: GENERATING PAIRINGS")
            print("=" * 70)
        try:
            pairings = pairadigm_obj.generate_pairings(
                num_pairs_per_item=num_pairs_per_item,
                random_seed=random_seed,
                breakdowns=True,
                update_classObject=True,
            )
            results["pairings"] = pairings
            if verbose:
                print(f"✓ Generated {len(pairings)} pairings")
        except Exception as exc:
            raise RuntimeError(f"Failed to generate pairings: {exc}") from exc
    else:
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 2: SKIPPED (data is already paired)")
            print("=" * 70)
        results["pairings"] = pairadigm_obj.pairwise_df

    # Step 3: LLM Annotations
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: GENERATING PAIRWISE LLM ANNOTATIONS")
        print("=" * 70)
    try:
        annotations = pairadigm_obj.generate_pairwise_annotations(
            max_workers=max_workers,
            update_classObject=True,
            max_tokens=max_tokens,
            temperature=temperature,
            allow_ties=allow_ties,
            client_indices=client_indices,
        )
        results["annotations"] = annotations
        if verbose:
            print(f"✓ Generated annotations for {len(annotations)} pairs")
    except Exception as exc:
        raise RuntimeError(f"Failed to generate pairwise annotations: {exc}") from exc

    # Step 4: Validation (if human annotations exist)
    if pairadigm_obj.annotated and pairadigm_obj.annotator_cols:
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 4: VALIDATION AGAINST HUMAN ANNOTATIONS")
            print("=" * 70)

        for label, fn in [
            ("4a AltTest",
             lambda: pairadigm_obj.alt_test(scoring_function="accuracy", epsilon=0.1,
                                             q_fdr=0.05, test_all_llms=True)),
            ("4b Transitivity",
             lambda: pairadigm_obj.check_transitivity()),
            ("4c IRR",
             lambda: pairadigm_obj.irr(method="auto")),
            ("4d Epsilon sensitivity",
             lambda: pairadigm_obj.plot_epsilon_sensitivity(
                 epsilon_range=(-0.1, 0.25), epsilon_step=0.02,
                 test_all_llms=True, return_data=True)),
        ]:
            key = label.split()[1].lower()
            if verbose:
                print(f"\n[{label}]...")
            try:
                results[key] = fn()
                if verbose:
                    print(f"✓ {label} completed")
            except Exception as exc:
                if verbose:
                    print(f"⚠ {label} failed: {exc}")
                results[key] = None
    else:
        if verbose:
            print("\nNote: No human annotations found. Skipping validation.")

    if verbose:
        print("\n" + "=" * 70)
        print("BUILD COMPLETE")
        print(f"Results keys: {list(results.keys())}")
        print("=" * 70 + "\n")

    return results


# ---------------------------------------------------------------------------
# Helper: prune pairwise DataFrame to satisfy AltTest constraints
# ---------------------------------------------------------------------------

def _prune_for_alt_test(
    pairwise_df: pd.DataFrame,
    ordinal_cols: List[str],
    min_annotators_per_pair: int,
    min_pairs_per_annotator: int,
) -> pd.DataFrame:
    """Iteratively prune a pairwise DataFrame so it is compatible with AltTest.

    Constraints enforced:
    * Every pair (row) must have at least ``min_annotators_per_pair`` non-null
      per-annotator decision values.
    * Every annotator column must cover at least ``min_pairs_per_annotator``
      non-null pairs.

    The two rules are applied in alternating passes until the DataFrame stops
    changing (convergence), because removing annotator columns can leave pairs
    with too few annotators, and removing pairs can leave annotators with too
    few items.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Output of ``_build_decision_rows`` (already has ``annotator_*``
        columns).
    ordinal_cols : list of str
        Ordinal column names used when building the pairwise DataFrame (used
        to identify which columns are annotator-decision columns vs. metadata).
    min_annotators_per_pair : int
    min_pairs_per_annotator : int

    Returns
    -------
    pd.DataFrame
        Pruned copy with a reset index.
    """
    if pairwise_df.empty:
        return pairwise_df

    # Identify annotator-decision columns (prefix ``annotator_``).
    ann_prefix = "annotator_"
    non_ann_cols = [c for c in pairwise_df.columns if not c.startswith(ann_prefix)]

    df = pairwise_df.copy()

    # Announce the pruning step so silence isn't mistaken for a hang.
    _orig_pairs = len(df)
    _orig_anns = sum(1 for c in df.columns if c.startswith(ann_prefix))

    for _iteration in range(1000):  # hard cap to prevent infinite loops
        ann_cols = [c for c in df.columns if c.startswith(ann_prefix)]
        if not ann_cols:
            warnings.warn(
                "All annotator columns were removed during AltTest pruning. "
                "No valid data remains. Try relaxing min_annotators_per_pair "
                "or min_pairs_per_annotator.",
                UserWarning,
            )
            return df

        prev_shape = df.shape

        # --- Pass 1: drop annotator columns with too few pairs ---------------
        counts_per_ann = df[ann_cols].notna().sum()
        valid_ann_cols = counts_per_ann[counts_per_ann >= min_pairs_per_annotator].index.tolist()
        dropped_cols = set(ann_cols) - set(valid_ann_cols)
        if dropped_cols:
            df = df.drop(columns=list(dropped_cols))
            ann_cols = [c for c in df.columns if c.startswith(ann_prefix)]

        # --- Pass 2: drop pairs with too few annotators ----------------------
        if ann_cols:
            counts_per_pair = df[ann_cols].notna().sum(axis=1)
            df = df[counts_per_pair >= min_annotators_per_pair].reset_index(drop=True)
        else:
            df = df.iloc[0:0].reset_index(drop=True)  # no annotators → empty

        # Converged when nothing changed in this iteration
        if df.shape == prev_shape:
            break

    # Drop annotator columns that are now entirely null (clean-up).
    ann_cols_final = [c for c in df.columns if c.startswith(ann_prefix)]
    if ann_cols_final:
        all_null = df[ann_cols_final].isna().all()
        null_cols = all_null[all_null].index.tolist()
        if null_cols:
            df = df.drop(columns=null_cols)

    _final_pairs = len(df)
    _final_anns = sum(1 for c in df.columns if c.startswith(ann_prefix))
    print(
        f"[AltTest pruning] {_orig_pairs} → {_final_pairs} pairs, "
        f"{_orig_anns} → {_final_anns} annotator columns retained "
        f"(min_annotators_per_pair={min_annotators_per_pair}, "
        f"min_pairs_per_annotator={min_pairs_per_annotator})."
    )
    if _final_pairs == 0:
        warnings.warn(
            "After AltTest pruning, no pairs remain. "
            "Your dataset may be too sparse to satisfy both constraints simultaneously. "
            "Consider reducing min_annotators_per_pair or min_pairs_per_annotator.",
            UserWarning,
        )

    return df


def pair_from_ordinal(
    data: pd.DataFrame,
    ordinal_cols: Union[str, List[str]],
    item_id_col: str,
    method: str = "adjacent",
    max_pairs_per_item: int = 10,
    random_seed: int = 42,
    min_gap: int = 0,
    provided_pairs=None,
    annotator_id_col: str = None,
    item_text_col: str = None,
    min_annotators_per_pair: int = 3,
    min_pairs_per_annotator: int = 50,
):
    """
    Generate pairwise comparisons from ordinal annotations.

    When ``annotator_id_col`` is provided, the function operates in
    *per-annotator* mode:

    * Decision columns are named ``annotator_{col}_{annotator_id}`` and encode
      each annotator's individual verdict for the pair.
    * Pairs are generated **only** for items that share at least one annotator,
      so every output row is guaranteed to have at least one non-null decision
      column.  This is the correct behaviour for sparse annotation designs
      (e.g. 1 000 annotators where each item is rated by only 5 of them).
    * Mean ordinal levels (averaged across annotators per item) are still
      computed and stored as ``mean_{col}_1`` / ``mean_{col}_2`` to make the
      level-based pairing logic transparent, but they are **not** used as
      decision values.
    * The output is **automatically pruned** to satisfy the AltTest
      requirements controlled by ``min_annotators_per_pair`` and
      ``min_pairs_per_annotator``.  Pruning is iterative — removing
      under-represented annotators can expose pairs that fall below the
      per-pair threshold, and vice versa — so the loop runs until the
      DataFrame is stable.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing items and their ordinal scores.
    ordinal_cols : str or list of str
        Column(s) in ``data`` containing integer ordinal scores.
    item_id_col : str
        Item ID column in ``data``.
    method : str, default ``'adjacent'``
        Pairing strategy:

        * ``'adjacent'``  -- pairs items from adjacent ordinal levels only
          (differing by exactly ``max(1, min_gap)`` levels).
        * ``'all'``       -- all unique (i, j) combinations (ignores
          ``max_pairs_per_item``).
        * ``'random'``    -- random sample capped at ``max_pairs_per_item``
          per item.
        * ``'provided'``  -- use pairs supplied via ``provided_pairs``.

    max_pairs_per_item : int, default 10
        Maximum number of pairs any single item may appear in across the
        entire output.  Enforced globally (not per level-crossing).
        Not applied when ``method='all'``.
    random_seed : int, default 42
    min_gap : int, default 0
        Minimum ordinal level difference required to include a pair.
    provided_pairs : pd.DataFrame, optional
        DataFrame with ``item1`` and ``item2`` columns.  Required when
        ``method='provided'``.
    annotator_id_col : str, optional
        Column in ``data`` identifying the annotator for each row.
    item_text_col : str, optional
        Column in ``data`` containing the item text.  When supplied, two
        columns named ``{item_text_col}_1`` and ``{item_text_col}_2`` are
        added to the output DataFrame with the text for ``item1`` and
        ``item2`` respectively.
    min_annotators_per_pair : int, default 3
        Minimum number of annotators that must have a non-null decision for a
        pair to be retained.  Only applied when ``annotator_id_col`` is
        provided.  The AltTest requires at least 3 annotators per observation
        (``min_humans_per_instance`` in :func:`pairadigm.validation.alt_test`).
    min_pairs_per_annotator : int, default 50
        Minimum number of pairs an annotator must have annotated to keep their
        decision column.  Only applied when ``annotator_id_col`` is provided.
        The AltTest skips annotators with fewer than this many instances
        (``min_instances_per_human`` in :func:`pairadigm.validation.alt_test`).

    Returns
    -------
    pd.DataFrame
        Pairwise DataFrame with decision columns based on ordinal scores,
        pruned so that every pair has at least ``min_annotators_per_pair``
        non-null decisions and every annotator column covers at least
        ``min_pairs_per_annotator`` pairs.
        If ``item_text_col`` was provided, also includes
        ``{item_text_col}_1`` and ``{item_text_col}_2`` columns.
    """
    # Warn that this function is still under development and may not work as intended
    warnings.warn(
        "This function is still under development and may not work as intended. "
        "Please use with caution.",
        UserWarning,
    )

    if isinstance(ordinal_cols, str):
        ordinal_cols = [ordinal_cols]

    if item_id_col not in data.columns:
        raise ValueError(f"ID column '{item_id_col}' not found in data.")
    for col in ordinal_cols:
        if col not in data.columns:
            raise ValueError(f"Ordinal column '{col}' not found in data.")
    if annotator_id_col is not None and annotator_id_col not in data.columns:
        raise ValueError(f"Annotator column '{annotator_id_col}' not found in data.")
    if item_text_col is not None and item_text_col not in data.columns:
        raise ValueError(f"Text column '{item_text_col}' not found in data.")

    # Build a {item_id: text} lookup once so every return path can use it.
    if item_text_col is not None:
        # If the same item appears on multiple rows (annotator-level data),
        # we take the first non-null text occurrence.
        _text_lookup: dict = (
            data[[item_id_col, item_text_col]]
            .drop_duplicates(subset=[item_id_col])
            .set_index(item_id_col)[item_text_col]
            .to_dict()
        )
    else:
        _text_lookup = None

    def _attach_text(pairwise_df: pd.DataFrame) -> pd.DataFrame:
        """Attach item text columns to the pairwise DataFrame in-place."""
        if _text_lookup is None:
            return pairwise_df
        pairwise_df[f"{item_text_col}_1"] = pairwise_df["item1"].map(_text_lookup)
        pairwise_df[f"{item_text_col}_2"] = pairwise_df["item2"].map(_text_lookup)
        return pairwise_df

    def _decide(lvl_a, lvl_b):
        if lvl_a is None or lvl_b is None:
            return None
        if lvl_a > lvl_b:
            return "Text1"
        if lvl_b > lvl_a:
            return "Text2"
        return "Tie"

    def _pair_key(a, b):
        """Canonical, hashable, order-independent pair key."""
        try:
            return (a, b) if a < b else (b, a)
        except TypeError:
            a_s, b_s = str(a), str(b)
            return (a_s, b_s) if a_s < b_s else (b_s, a_s)

    # ------------------------------------------------------------------
    # Build id_to_scores and annotator structures
    # ------------------------------------------------------------------
    if annotator_id_col is not None:
        # {item_id: {annotator_id: {col: score}}}
        id_to_scores: dict = {}
        for _, row in data.iterrows():
            item_id = row[item_id_col]
            annotator = row[annotator_id_col]
            scores = {
                col: float(row[col])
                for col in ordinal_cols
                if pd.notna(row[col])
            }
            if scores:
                id_to_scores.setdefault(item_id, {})[annotator] = scores

        # Mean scores per item — used only for level-based pairing logic.
        # Computed with an explicit loop to guard against division by zero.
        id_to_mean_scores: dict = {}
        for item_id, annotator_scores in id_to_scores.items():
            means: dict = {}
            for col in ordinal_cols:
                vals = [
                    ann_scores[col]
                    for ann_scores in annotator_scores.values()
                    if col in ann_scores
                ]
                if vals:  # guard: at least one annotator scored this col
                    means[col] = sum(vals) / len(vals)
            if means:
                id_to_mean_scores[item_id] = means

        # Build annotator → items index for constructing the shared-annotator
        # pair graph efficiently.
        annotator_to_items: dict = {}
        for item_id, ann_scores in id_to_scores.items():
            for annotator in ann_scores:
                annotator_to_items.setdefault(annotator, []).append(item_id)

        # Enumerate every pair of items that share at least one annotator.
        # Restricting pair generation to this set guarantees every output row
        # has at least one non-null decision column (critical for sparse
        # annotation, e.g. 1 000 annotators × 5 items each).
        shared_pair_keys: set = set()
        for items in annotator_to_items.values():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    shared_pair_keys.add(_pair_key(items[i], items[j]))

        if not shared_pair_keys:
            warnings.warn(
                "No item pairs share an annotator — all decision columns will be "
                "None.  Check that annotator_id_col values overlap across items.",
                UserWarning,
            )
    else:
        # Flat {item_id: {col: score}} — last row wins when item appears twice
        id_to_scores = {}
        for _, row in data.iterrows():
            item_id = row[item_id_col]
            scores = {
                col: float(row[col])
                for col in ordinal_cols
                if pd.notna(row[col])
            }
            if scores:
                id_to_scores[item_id] = scores
        id_to_mean_scores = id_to_scores  # identical structure when no annotator col
        shared_pair_keys = None            # unrestricted

    def _is_valid_candidate(a, b) -> bool:
        """Pair is valid if it has no annotator restriction OR shares an annotator."""
        if shared_pair_keys is not None:
            return _pair_key(a, b) in shared_pair_keys
        return True

    # ------------------------------------------------------------------
    # Decision-row builder (used by adjacent / all / random)
    # ------------------------------------------------------------------
    def _build_decision_rows(pairs, primary_col, id_to_level) -> pd.DataFrame:
        rows: list = []
        all_decision_cols: set = set()

        for a, b in pairs:
            row_dict: dict = {"item1": a, "item2": b}

            if annotator_id_col is not None:
                # Prefix mean columns clearly so they are not confused with
                # per-annotator decision values.
                if a in id_to_level:
                    row_dict[f"mean_{primary_col}_1"] = id_to_level[a]
                if b in id_to_level:
                    row_dict[f"mean_{primary_col}_2"] = id_to_level[b]

                annotators_a = id_to_scores.get(a, {})
                annotators_b = id_to_scores.get(b, {})
                shared_annotators = set(annotators_a) & set(annotators_b)
                for annotator in shared_annotators:
                    for current_col in ordinal_cols:
                        lvl_a = annotators_a[annotator].get(current_col)
                        lvl_b = annotators_b[annotator].get(current_col)
                        col_name = f"annotator_{current_col}_{annotator}"
                        row_dict[col_name] = _decide(lvl_a, lvl_b)
                        all_decision_cols.add(col_name)
            else:
                row_dict[f"{primary_col}_1"] = id_to_level.get(a)
                row_dict[f"{primary_col}_2"] = id_to_level.get(b)
                for current_col in ordinal_cols:
                    lvl_a = id_to_scores.get(a, {}).get(current_col)
                    lvl_b = id_to_scores.get(b, {}).get(current_col)
                    row_dict[f"annotator_{current_col}"] = _decide(lvl_a, lvl_b)

            rows.append(row_dict)

        pairwise_df = pd.DataFrame(rows)
        # Ensure every decision column appears in every row (fill sparse cols)
        for col_name in all_decision_cols:
            if col_name not in pairwise_df.columns:
                pairwise_df[col_name] = None

        return pairwise_df

    # ------------------------------------------------------------------
    # Provided method
    # ------------------------------------------------------------------
    if method == "provided":
        if provided_pairs is None:
            raise ValueError("provided_pairs must be supplied when method='provided'.")
        if "item1" not in provided_pairs.columns or "item2" not in provided_pairs.columns:
            raise ValueError("provided_pairs must contain 'item1' and 'item2' columns.")

        rows: list = []
        all_decision_cols: set = set()

        for _, pair_row in provided_pairs.iterrows():
            a, b = pair_row["item1"], pair_row["item2"]
            row_dict = pair_row.to_dict()

            if annotator_id_col is not None:
                annotators_a = id_to_scores.get(a, {})
                annotators_b = id_to_scores.get(b, {})
                shared_annotators = set(annotators_a) & set(annotators_b)
                if not shared_annotators:
                    warnings.warn(
                        f"Provided pair ({a!r}, {b!r}) has no shared annotators; "
                        "decision columns will be None for this row.",
                        UserWarning,
                    )
                for annotator in shared_annotators:
                    for col in ordinal_cols:
                        lvl_a = annotators_a[annotator].get(col)
                        lvl_b = annotators_b[annotator].get(col)
                        col_name = f"annotator_{col}_{annotator}"
                        row_dict[col_name] = _decide(lvl_a, lvl_b)
                        all_decision_cols.add(col_name)
            else:
                for col in ordinal_cols:
                    lvl_a = id_to_scores.get(a, {}).get(col)
                    lvl_b = id_to_scores.get(b, {}).get(col)
                    row_dict[f"annotator_{col}"] = _decide(lvl_a, lvl_b)

            rows.append(row_dict)

        pairwise_df = pd.DataFrame(rows)
        for col_name in all_decision_cols:
            if col_name not in pairwise_df.columns:
                pairwise_df[col_name] = None

        pairwise_df = _attach_text(pairwise_df)
        if annotator_id_col is not None:
            pairwise_df = _prune_for_alt_test(
                pairwise_df, ordinal_cols,
                min_annotators_per_pair, min_pairs_per_annotator,
            )
        print(f"Generated decisions for {len(pairwise_df)} provided pairs.")
        return pairwise_df

    # ------------------------------------------------------------------
    # Level-based pair generation: adjacent / all / random
    # ------------------------------------------------------------------
    # Use the first ordinal col's mean scores to place each item on a level.
    primary_col = ordinal_cols[0]
    id_to_level = {
        item_id: round(scores[primary_col])
        for item_id, scores in id_to_mean_scores.items()
        if primary_col in scores
    }

    rng = random.Random(random_seed)
    item_list = list(id_to_level.keys())
    n = len(item_list)

    pairs: list = []
    seen: set = set()

    if method == "all":
        for i in range(n):
            for j in range(i + 1, n):
                a, b = item_list[i], item_list[j]
                if (
                    abs(id_to_level[a] - id_to_level[b]) >= min_gap
                    and _is_valid_candidate(a, b)
                ):
                    pairs.append((a, b))

    elif method == "adjacent":
        by_level: dict = {}
        for iid, lvl in id_to_level.items():
            by_level.setdefault(lvl, []).append(iid)
        sorted_levels = sorted(by_level)

        # Bug fix 1: use a single exact step so 'adjacent' means one specific
        # ordinal distance, not a window of two distances.
        step = max(1, min_gap)

        # Bug fix 2: per-item count is shared across ALL level-pair iterations
        # so the cap is enforced globally, not just within a single crossing.
        per_item_count: dict = {}

        for level in sorted_levels:
            next_levels = [l for l in sorted_levels if l - level == step]
            for nl in next_levels:
                hi_items = by_level[nl][:]
                lo_items = by_level[level][:]
                rng.shuffle(hi_items)
                rng.shuffle(lo_items)
                for hi in hi_items:
                    for lo in lo_items:
                        key = _pair_key(hi, lo)
                        if (
                            key not in seen
                            and per_item_count.get(hi, 0) < max_pairs_per_item
                            and per_item_count.get(lo, 0) < max_pairs_per_item
                            and _is_valid_candidate(hi, lo)
                        ):
                            pairs.append((hi, lo))
                            seen.add(key)
                            per_item_count[hi] = per_item_count.get(hi, 0) + 1
                            per_item_count[lo] = per_item_count.get(lo, 0) + 1

    elif method == "random":
        all_candidates = [
            (item_list[i], item_list[j])
            for i in range(n)
            for j in range(i + 1, n)
            if abs(id_to_level[item_list[i]] - id_to_level[item_list[j]]) >= min_gap
            and _is_valid_candidate(item_list[i], item_list[j])
        ]
        rng.shuffle(all_candidates)
        per_item: dict = {}
        for a, b in all_candidates:
            if (
                per_item.get(a, 0) < max_pairs_per_item
                and per_item.get(b, 0) < max_pairs_per_item
            ):
                pairs.append((a, b))
                per_item[a] = per_item.get(a, 0) + 1
                per_item[b] = per_item.get(b, 0) + 1
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'adjacent', 'all', 'random', or 'provided'."
        )

    if not pairs:
        warnings.warn(
            f"No pairs generated with method='{method}' and min_gap={min_gap}. "
            "Try reducing min_gap or using a different method.",
            UserWarning,
        )
        return pd.DataFrame(columns=["item1", "item2"])

    pairwise_df = _build_decision_rows(pairs, primary_col, id_to_level)
    pairwise_df = _attach_text(pairwise_df)
    if annotator_id_col is not None:
        pairwise_df = _prune_for_alt_test(
            pairwise_df, ordinal_cols,
            min_annotators_per_pair, min_pairs_per_annotator,
        )
    print(f"Generated {len(pairwise_df)} pairs from ordinal annotations (method='{method}').")
    return pairwise_df

