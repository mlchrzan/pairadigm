"""
Concept-Guided Chain-of-Thought (CGCoT) breakdown generation functions.
"""

import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from .llm_client import LLMClient


def load_cgcot_prompts(file_path: str) -> List[str]:
    """
    Load CGCoT prompt templates from a text file.
    
    Parameters
    ----------
    file_path : str
        Path to the text file containing prompts (one per line)
        
    Returns
    -------
    List[str]
        List of prompt templates
        
    Examples
    --------
    >>> prompts = load_cgcot_prompts('prompts.txt')
    >>> len(prompts)
    3
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts = [line.strip() for line in file if line.strip()]
    
    if not prompts:
        raise ValueError(f"No prompts found in {file_path}")
    
    return prompts


def generate_cgcot_breakdown(
    text: str,
    model: str,
    concept_prompts: List[str],
    client: LLMClient = None,
    rate_limit_per_minute: int = 15
) -> str:
    """
    Generate concept-specific breakdown for a given text using CGCoT prompts.
    
    The function iteratively applies each prompt, building context from
    previous responses to create a comprehensive analysis.
    
    Parameters
    ----------
    text : str
        The input text to analyze
    model : str
        Model name to use
    concept_prompts : List[str]
        Sequential CGCoT prompt templates
    client : LLMClient, optional
        Reusable client instance
    rate_limit_per_minute : int
        API rate limit
        
    Returns
    -------
    str
        Concatenated concept-specific breakdown
        
    Examples
    --------
    >>> prompts = ["Analyze: {text}", "Expand on: {previous_answers}"]
    >>> breakdown = generate_cgcot_breakdown("Sample text", "gpt-4o", prompts)
    """
    if client is None:
        client = LLMClient(model_name=model)
    
    breakdown = [f"Original Text: {text}"]
    prev_answers = []
    sleep_time = 60.0 / rate_limit_per_minute
    
    for i, prompt_template in enumerate(concept_prompts):
        # Format the prompt with text and previous answers
        full_prompt = prompt_template.format(
            text=text,
            previous_answers="\n".join(prev_answers)
        )
        
        # Query the LLM
        response = client.generate(full_prompt)
        prev_answers.append(response)
        breakdown.append(f"Prompt {i+1} response: {response}")
        
        # Rate limiting (except for last prompt)
        if i < len(concept_prompts) - 1:
            time.sleep(sleep_time)
    
    return "\n".join(breakdown)


def generate_breakdowns_parallel(
    df: pd.DataFrame,
    cgcot_prompts: List[str],
    model: str = 'gemini-2.0-flash-exp',
    row_name: str = None,
    row_id: str = None,
    max_workers: int = 8
) -> Dict[str, str]:
    """
    Generate CGCoT breakdowns for multiple items in parallel.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing items to analyze
    cgcot_prompts : List[str]
        CGCoT prompt templates
    model : str
        Model name to use
    row_name : str
        Column name for text content
    row_id : str
        Column name for unique identifiers
    max_workers : int
        Number of parallel workers
        
    Returns
    -------
    Dict[str, str]
        Mapping from row_id to breakdown
        
    Examples
    --------
    >>> df = pd.DataFrame({'id': [1, 2], 'text': ['Text A', 'Text B']})
    >>> prompts = load_cgcot_prompts('prompts.txt')
    >>> breakdowns = generate_breakdowns_parallel(df, prompts, row_name='text', row_id='id')
    """
    if row_name is None or row_id is None:
        raise ValueError("row_name and row_id must be specified")
    
    if row_name not in df.columns or row_id not in df.columns:
        raise ValueError(f"Columns '{row_name}' or '{row_id}' not found in DataFrame")
    
    # Create a shared client for efficiency
    client = LLMClient(model_name=model)
    
    results = {}
    total = len(df)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                generate_cgcot_breakdown,
                row[row_name],
                model,
                cgcot_prompts,
                client
            ): row[row_id]
            for _, row in df.iterrows()
        }
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            item_id = futures[future]
            try:
                results[item_id] = future.result()
                if i % 10 == 0:
                    print(f"  Completed {i}/{total} breakdowns")
            except Exception as e:
                print(f"  Error processing item {item_id}: {e}")
                results[item_id] = f"ERROR: {e}"
    
    print(f"  Completed all {total} breakdowns")
    return results


def validate_cgcot_prompts(prompts: List[str]) -> bool:
    """
    Validate that CGCoT prompts contain required placeholders.
    
    Parameters
    ----------
    prompts : List[str]
        List of prompt templates
        
    Returns
    -------
    bool
        True if prompts are valid
        
    Raises
    ------
    ValueError
        If prompts are invalid
    """
    if not prompts:
        raise ValueError("Prompts list is empty")
    
    # First prompt should reference {text}
    if '{text}' not in prompts[0]:
        raise ValueError("First prompt must contain {text} placeholder")
    
    # Subsequent prompts should reference {previous_answers}
    for i, prompt in enumerate(prompts[1:], 1):
        if '{previous_answers}' not in prompt:
            raise ValueError(
                f"Prompt {i+1} should contain {{previous_answers}} placeholder"
            )
    
    return True