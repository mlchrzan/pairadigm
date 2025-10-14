"""
Pairwise comparison and annotation functions.
"""

import re
import time
import pandas as pd
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_client import LLMClient


def pairwise_compare(
    text1_breakdown: str,
    text2_breakdown: str,
    target_concept: str,
    model: str,
    client: LLMClient = None
) -> Tuple[str, str]:
    """
    Compare two CGCoT breakdowns to decide which expresses greater level of target concept.
    
    Parameters
    ----------
    text1_breakdown : str
        Breakdown for first text
    text2_breakdown : str
        Breakdown for second text
    target_concept : str
        Concept name for comparison (e.g., "objectivity", "political bias")
    model : str
        Model name to use
    client : LLMClient, optional
        Reusable client instance
        
    Returns
    -------
    Tuple[str, str]
        (decision, justification) where decision is "Text1" or "Text2"
        
    Examples
    --------
    >>> decision, justification = pairwise_compare(
    ...     "Breakdown A shows...",
    ...     "Breakdown B shows...",
    ...     "objectivity",
    ...     "gpt-4o"
    ... )
    """
    if client is None:
        client = LLMClient(model_name=model)
    
    comparison_prompt = f"""
Description 1: {text1_breakdown}

Description 2: {text2_breakdown}

Based on these two Descriptions, which Description expresses greater {target_concept}: Description 1 or Description 2? 
You must choose one of the descriptions.

Format your response as follows:
FINAL ANSWER: <Your choice of "Description 1" or "Description 2">
JUSTIFICATION: <Your CONCISE reasoning for the choice>
"""
    
    response = client.generate(
        comparison_prompt,
        system_message="You are a precise and detail-oriented assistant."
    )
    
    # Extract the final answer using regex
    answer_pattern = r"FINAL ANSWER:\s*(Description 1|Description 2|Tie)"
    match = re.search(answer_pattern, response, re.IGNORECASE)
    
    if match:
        extracted_answer = match.group(1)
        if extracted_answer.lower() == "description 1":
            final_answer = "Text1"
        elif extracted_answer.lower() == "description 2":
            final_answer = "Text2"
        else:
            final_answer = "ERROR"
    else:
        # Fallback: Use LLM to extract the answer
        extraction_prompt = f"""
In the following text, which Description is described to be expressing greater {target_concept}: Description 1 or Description 2? 
ONLY REPLY WITH "Description 1" or "Description 2". 

Text: {response}
"""
        extracted_answer = client.generate(
            extraction_prompt,
            system_message="You are a precise and detail-oriented assistant."
        ).strip()
        
        if extracted_answer == "Description 1":
            final_answer = "Text1"
        elif extracted_answer == "Description 2":
            final_answer = "Text2"
        else:
            final_answer = "ERROR"
    
    return final_answer, response


def generate_pairwise_annotations(
    uuid_pairings: pd.DataFrame,
    target_concept: str,
    model: str = 'gemini-2.0-flash-exp',
    rate_limit_per_minute: int = 15
) -> pd.DataFrame:
    """
    Run pairwise comparisons on all pairs sequentially.
    
    Parameters
    ----------
    uuid_pairings : pd.DataFrame
        DataFrame with 'breakdown1' and 'breakdown2' columns
    target_concept : str
        The concept to compare for
    model : str
        Model name to use
    rate_limit_per_minute : int
        API rate limit
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with added 'decision' and 'justification' columns
    """
    if 'breakdown1' not in uuid_pairings.columns or 'breakdown2' not in uuid_pairings.columns:
        raise ValueError("DataFrame must have 'breakdown1' and 'breakdown2' columns")
    
    client = LLMClient(model_name=model)
    decisions = []
    justifications = []
    sleep_time = 2 * (60.0 / rate_limit_per_minute)  # Up to 2 requests per comparison
    
    total = len(uuid_pairings)
    print(f"Annotating {total} pairs...")
    
    for idx, row in uuid_pairings.iterrows():
        breakdown1 = row['breakdown1']
        breakdown2 = row['breakdown2']
        
        # Run pairwise comparison
        decision, justification = pairwise_compare(
            breakdown1, breakdown2, target_concept, model, client
        )
        
        decisions.append(decision)
        justifications.append(justification)
        
        # Print progress
        if (idx + 1) % 50 == 0:
            print(f"  Completed {idx + 1}/{total} comparisons")
        
        # Sleep to respect rate limit
        if idx < total - 1:
            time.sleep(sleep_time)
    
    # Add results to dataframe
    uuid_pairings['decision'] = decisions
    uuid_pairings['justification'] = justifications
    
    return uuid_pairings


def generate_pairwise_annotations_parallel(
    uuid_pairings: pd.DataFrame,
    target_concept: str,
    model: str = 'gemini-2.0-flash-exp',
    max_workers: int = 8
) -> pd.DataFrame:
    """
    Run pairwise comparisons on all pairs in parallel.
    
    Parameters
    ----------
    uuid_pairings : pd.DataFrame
        DataFrame with 'breakdown1' and 'breakdown2' columns
    target_concept : str
        The concept to compare for
    model : str
        Model name to use
    max_workers : int
        Number of parallel workers
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with added 'decision' and 'justification' columns
    """
    if 'breakdown1' not in uuid_pairings.columns or 'breakdown2' not in uuid_pairings.columns:
        raise ValueError("DataFrame must have 'breakdown1' and 'breakdown2' columns")
    
    client = LLMClient(model_name=model)
    total = len(uuid_pairings)
    results = [None] * total
    
    print(f"Annotating {total} pairs in parallel...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                pairwise_compare,
                row['breakdown1'],
                row['breakdown2'],
                target_concept,
                model,
                client
            ): idx
            for idx, row in uuid_pairings.iterrows()
        }
        
        # Collect results
        for i, future in enumerate(as_completed(futures), 1):
            idx = futures[future]
            try:
                decision, justification = future.result()
            except Exception as e:
                decision, justification = "ERROR", str(e)
            
            results[idx] = (decision, justification)
            
            # Print progress
            if i % 50 == 0:
                print(f"  Completed {i}/{total} comparisons")
    
    # Add results to dataframe
    uuid_pairings['decision'] = [r[0] for r in results]
    uuid_pairings['justification'] = [r[1] for r in results]
    
    return uuid_pairings