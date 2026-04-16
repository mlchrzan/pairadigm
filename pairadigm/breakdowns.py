"""
CGCoT (Concept-Grounded Chain-of-Thought) breakdown generation.

Provides:
  - generate_cgcot_breakdown : generate a single item's breakdown
  - generate_breakdowns      : parallel breakdown generation for unpaired or
                               paired data (pass item_id_cols + item_text_cols
                               to activate paired mode)

Fix 1g: rate-limiting is applied at the *submission* level (thread throttle),
not inside the worker, so the ThreadPoolExecutor is not blocked by sleep().
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from .client import LLMClient


# Default system message (module-level constant)
_DEFAULT_BREAKDOWN_SYSTEM_MSG = (
    "You are a precise and detail-oriented assistant working to uncover nuance "
    "in data. Respond to the prompt concisely. Restate the core ask/idea of the prompt "
    "in your response (without repeating it). Do not include any additional commentary, questions, or information."
)


# ---------------------------------------------------------------------------
# Core single-item breakdown
# ---------------------------------------------------------------------------

def generate_cgcot_breakdown(
    text: str,
    cgcot_prompts: List[str],
    client: LLMClient,
    rate_limit_per_minute: Optional[int] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    system_message: str = _DEFAULT_BREAKDOWN_SYSTEM_MSG,
    debug_mode: bool = False,
) -> str:
    """
    Generate a concept-specific CGCoT breakdown for a single text.

    Parameters
    ----------
    text : str
        The text to analyse.
    cgcot_prompts : list of str
        Ordered prompts (each may reference ``{text}`` and ``{previous_answers}``).
    client : LLMClient
        LLM client to use.
    rate_limit_per_minute : int or None
        If set, sleep between sequential prompts within a single item. This is
        **per-item** inter-prompt delay, not the cross-item rate-limit (that is
        handled at the dispatch level in generate_breakdowns).
    max_tokens : int
    temperature : float
    system_message : str
    debug_mode : bool, default False
        If ``False`` (default), the stored breakdown contains only the clean
        LLM response text (without ``"Prompt N response:"`` section headers).
        If ``True``, the original labelled format is preserved, which is useful
        when iterating on prompt designs.

    Returns
    -------
    str
        Breakdown string.  Format depends on ``debug_mode``.
    """
    lines = [f"Original Text: {text}"]
    prev_answers: List[str] = []
    sleep_time = (60.0 / rate_limit_per_minute) if rate_limit_per_minute else 0.0

    for i, prompt_template in enumerate(cgcot_prompts):
        full_prompt = prompt_template.format(
            text=text,
            previous_answers="\n".join(prev_answers),
        )
        if not full_prompt.strip():
            raise ValueError(
                "Empty prompt generated. Ensure cgcot_prompts do not have empty lines."
            )
        try:
            response = client.generate(
                prompt=full_prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            response = f"ERROR: {exc}"

        prev_answers.append(response)
        # debug_mode=True keeps labelled headers; default strips them for clean storage
        if debug_mode:
            lines.append(f"Prompt {i + 1} response: {response}")
        else:
            lines.append(response)

        # Intra-item sleep (between consecutive prompts for the same item)
        if i < len(cgcot_prompts) - 1 and sleep_time > 0:
            time.sleep(sleep_time)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_clients(
    clients: List[LLMClient],
    client_indices: Optional[Union[int, List[int]]],
) -> List[Tuple[int, LLMClient]]:
    """Return ``[(index, client), ...]`` from an optional index selector."""
    if client_indices is None:
        return list(enumerate(clients))
    if isinstance(client_indices, int):
        if client_indices >= len(clients):
            raise ValueError(
                f"client_indices {client_indices} out of range. "
                f"Only {len(clients)} client(s) available."
            )
        return [(client_indices, clients[client_indices])]
    if isinstance(client_indices, list):
        out = []
        for idx in client_indices:
            if idx >= len(clients):
                raise ValueError(
                    f"client_indices {idx} out of range. "
                    f"Only {len(clients)} client(s) available."
                )
            out.append((idx, clients[idx]))
        return out
    raise TypeError("client_indices must be None, int, or List[int]")


# ---------------------------------------------------------------------------
# generate_breakdowns  (unified: unpaired and paired modes, fix 1g)
# ---------------------------------------------------------------------------

def generate_breakdowns(
    data: pd.DataFrame,
    item_id_name: str,
    cgcot_prompts: List[str],
    clients: List[LLMClient],
    model_names: List[str],
    # Unpaired mode
    text_name: Optional[str] = None,
    # Paired mode
    item_id_cols: Optional[List[str]] = None,
    item_text_cols: Optional[List[str]] = None,
    # Shared options
    client_indices: Optional[Union[int, List[int]]] = None,
    max_workers: int = 8,
    rate_limit_per_minute: Optional[int] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    show_progress: bool = True,
    system_message: str = _DEFAULT_BREAKDOWN_SYSTEM_MSG,
    debug_mode: bool = False,
) -> Dict[int, Dict]:
    """
    Generate CGCoT breakdowns for all items in a DataFrame.

    Works in two modes, selected by which optional parameters are supplied:

    **Unpaired mode** — pass ``text_name``.
        Reads each item's text directly from ``data[text_name]``.

    **Paired mode** — pass ``item_id_cols`` and ``item_text_cols``.
        Extracts the unique items from both sides of the pair columns, deduplicates
        them, then generates one breakdown per unique item.

    Fix **1g**: rate-limit is applied at the submission level using a small
    ``time.sleep`` between ``executor.submit()`` calls, not inside the worker
    function.  This keeps threads non-blocked.

    Parameters
    ----------
    data : pd.DataFrame
        Item DataFrame (unpaired) or pair DataFrame (paired).
    item_id_name : str
        Column name that uniquely identifies each item.
    cgcot_prompts : list of str
    clients : list of LLMClient
    model_names : list of str
    text_name : str, optional
        **Unpaired mode.** Column in ``data`` that holds the raw text.
    item_id_cols : list of str, optional
        **Paired mode.** Two-element list of the pair ID columns
        (e.g. ``['item1', 'item2']``).
    item_text_cols : list of str, optional
        **Paired mode.** Two-element list of the pair text columns
        (e.g. ``['text1', 'text2']``).
    client_indices : int, list of int, or None
    max_workers : int
    rate_limit_per_minute : int or None
        Cross-item rate limit.  Sleep of ``60 / rate_limit_per_minute`` is
        inserted **between consecutive submissions** to the thread pool.
    max_tokens : int
    temperature : float
    show_progress : bool
    system_message : str
    debug_mode : bool, default False
        Passed through to :func:`generate_cgcot_breakdown`. When ``False``
        (default), stored breakdowns contain only clean LLM response text.
        When ``True``, ``"Prompt N response:"`` headers are preserved.

    Returns
    -------
    dict
        ``{client_index: {item_id: breakdown_str}}`` for all clients used.

    Raises
    ------
    ValueError
        If neither ``text_name`` (unpaired) nor both ``item_id_cols`` and
        ``item_text_cols`` (paired) are provided.
    """
    # ------------------------------------------------------------------
    # Build a unified {item_id: text} mapping regardless of input mode
    # ------------------------------------------------------------------
    paired_mode = item_id_cols is not None and item_text_cols is not None
    if paired_mode:
        item1_id_col, item2_id_col = item_id_cols
        item1_text_col, item2_text_col = item_text_cols
        item1_df = data[[item1_id_col, item1_text_col]].rename(
            columns={item1_id_col: item_id_name, item1_text_col: "_text"}
        )
        item2_df = data[[item2_id_col, item2_text_col]].rename(
            columns={item2_id_col: item_id_name, item2_text_col: "_text"}
        )
        items_df = (
            pd.concat([item1_df, item2_df], ignore_index=True)
            .drop_duplicates(subset=[item_id_name])
            .reset_index(drop=True)
        )
        text_mapping = dict(zip(items_df[item_id_name], items_df["_text"]))
    elif text_name is not None:
        items_df = data[[item_id_name]].drop_duplicates().reset_index(drop=True)
        text_mapping = dict(zip(data[item_id_name], data[text_name]))
    else:
        raise ValueError(
            "Must provide either 'text_name' (unpaired mode) or both "
            "'item_id_cols' and 'item_text_cols' (paired mode)."
        )

    # ------------------------------------------------------------------
    # Shared dispatch loop
    # ------------------------------------------------------------------
    clients_to_use = _resolve_clients(clients, client_indices)
    submit_sleep = (60.0 / rate_limit_per_minute) if rate_limit_per_minute else 0.0
    total_items = len(items_df)
    all_results: Dict[int, Dict] = {}

    for client_idx, client in clients_to_use:
        model_name = model_names[client_idx]
        print(f"\n{'='*70}")
        print(
            f"Generating breakdowns for {total_items} "
            f"{'unique ' if paired_mode else ''}items using: {model_name}"
        )
        if debug_mode:
            print("  [debug_mode=True: 'Prompt N response:' headers will be stored]")
        print(f"{'='*70}")

        results: Dict = {}
        completed = failed = 0
        futures = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for item_id in items_df[item_id_name]:
                futures[executor.submit(
                    generate_cgcot_breakdown,
                    text_mapping[item_id],
                    cgcot_prompts,
                    client,
                    None,   # no intra-item sleep when cross-item throttled
                    max_tokens,
                    temperature,
                    system_message,
                    debug_mode,
                )] = item_id
                # Fix 1g: throttle at submission level
                if submit_sleep > 0:
                    time.sleep(submit_sleep)

            pbar = tqdm(
                as_completed(futures),
                total=total_items,
                desc=f"[{model_name}]",
                disable=not show_progress,
            )
            for future in pbar:
                item_id = futures[future]
                try:
                    results[item_id] = future.result()
                    completed += 1
                except Exception as exc:
                    results[item_id] = f"ERROR: {exc}"
                    failed += 1
                if failed > 0:
                    pbar.set_postfix({"success": completed, "failed": failed})

        print(f"Completed: {completed}/{total_items} items")
        if failed:
            print(f"Failed: {failed} items")

        all_results[client_idx] = results

    return all_results
