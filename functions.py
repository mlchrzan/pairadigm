import pandas as pd 
import itertools
import random
import choix
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from typing import List, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
import plotly.graph_objects as go
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

##############################
# CGCoT Pairwise Comparison Functions 
##############################

# Function to get the cgcot prompts from the text file
def load_cgcot_prompts(file_path: str) -> List[str]:
    """
    Load CGCoT prompt templates from a text file.
    Args:
        file_path (str): Path to the text file containing prompts.
    Returns:
        List[str]: List of prompt templates.
    """
    with open(file_path, 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]
    return prompts

# Helper function to query LLM for breakdowns and comparisons
def query_llm(prompt: str, model: str, system_message: str=None) -> str:
    """
    Function to query the language model with a given prompt and return the response. 
    """
    if system_message is None:
        system_message = "You are an expert YuGiOh analyst specializing in evaluating cards for specific gameplay constructs. Keep your responses concise and focused on the analysis."
        
    if model=='gpt-4.1':
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0
        )
        return response.choices[0].message.content
    
    # Check if model contains 'gemini'
    elif 'gemini' in model:
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0
            ),
            contents=prompt
        )
        return response.text
    else:
        raise ValueError("Unsupported model specified.")
    
def generate_cgcot_breakdown(text, model, concept_prompts, rate_limit_per_minute=15) -> str:
    """
    Generate concept-specific breakdown for a given text using CGCoT prompts.
    Args:
        text (str): The input text to analyze
        concept_prompts (List[str]): Sequential CGCoT prompt templates
    Returns:
        str: Concatenated concept-specific breakdown
    """
    breakdown = [f"Original Text: {text}"]
    prev_answers = []
    sleep_time = 60.0 / rate_limit_per_minute
    for i, prompt_template in enumerate(concept_prompts):
        full_prompt = prompt_template.format(text=text, previous_answers="\n".join(prev_answers))
        response = query_llm(full_prompt, model=model)
        prev_answers.append(response)
        breakdown.append(f"Prompt {i+1} response: {response}")
        if i < len(concept_prompts) - 1:
            time.sleep(sleep_time)  # Wait to avoid rate limit
    return "\n".join(breakdown)

def generate_breakdowns_parallel(df, 
                                 cgcot_prompts, 
                                 model='gemini-2.5-flash', 
                                 row_name=None, 
                                 row_id=None, 
                                 max_workers=8):

    if row_name is None or row_id is None:
        raise ValueError("row_name and row_id must be specified to identify rows uniquely.")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_cgcot_breakdown, row[row_name], model, cgcot_prompts): row[row_id]
            for _, row in df.iterrows()
        }
        for future in as_completed(futures):
            uuid = futures[future]
            try:
                results[uuid] = future.result()
            except Exception as e:
                results[uuid] = f"ERROR: {e}"
    return results

# Helper function to create pairings ensuring connectivity and minimum pairs per item
def pair_items(items, num_pairs_per_item=10, random_seed=42):
    """
    Generate a connected subset of pairwise comparisons as a DataFrame.
    Args:
        items (list): Items to compare.
        num_pairs_per_item (int, optional): Min pairs per item.
        random_seed (int, optional): For reproducibility.
    Returns:
        pd.DataFrame: DataFrame with columns ['item1', 'item2'] representing pairings.
    """
    if random_seed is not None:
        random.seed(random_seed)

    n = len(items)
    if n < 2:
        return pd.DataFrame(columns=['item1', 'item2'])
    
    min_pairs = num_pairs_per_item or max(3, min(6, int(n ** 0.5)))
    all_pairs = set(itertools.combinations(items, 2))
    chosen_pairs = set()
    covered = {item: set() for item in items}

    # Start with a spanning chain for connectivity
    for i in range(n-1):
        pair = tuple(sorted((items[i], items[i+1])))
        chosen_pairs.add(pair)
        covered[items[i]].add(items[i+1])
        covered[items[i+1]].add(items[i])

    # Sample additional pairs to ensure min_pairs per item
    additional_pairs = list(all_pairs - chosen_pairs)
    random.shuffle(additional_pairs)
    for a,b in additional_pairs:
        if len(covered[a]) < min_pairs or len(covered[b]) < min_pairs:
            chosen_pairs.add((a,b))
            covered[a].add(b)
            covered[b].add(a)

    # Convert to DataFrame
    df = pd.DataFrame(list(chosen_pairs), columns=['item1', 'item2'])
    return df

def generate_pairings_df(df, row_id, num_pairs_per_item=10, random_seed=42):
    """ 
    Generate pairings for items in a DataFrame column.
    Args:
        df (pd.DataFrame): DataFrame containing items to pair.
        row_id (str): Column name with unique item identifiers.
        num_pairs_per_item (int, optional): Min pairs per item.
        random_seed (int, optional): For reproducibility.
    Returns:
        pd.DataFrame: DataFrame with pairings and associated breakdowns.
    """

    if row_id not in df.columns:
        raise ValueError(f"Column '{row_id}' not found in DataFrame.")

    if "CGCoT_Breakdown" not in df.columns:
        raise ValueError("Column 'CGCoT_Breakdown' not found in DataFrame.")

    uuid_pairings = pair_items(df[row_id].tolist(),
                               num_pairs_per_item=num_pairs_per_item,
                               random_seed=random_seed)

    uuid_to_desc = dict(zip(df[row_id], df['CGCoT_Breakdown']))
    uuid_pairings['breakdown1'] = uuid_pairings['item1'].map(uuid_to_desc)
    uuid_pairings['breakdown2'] = uuid_pairings['item2'].map(uuid_to_desc)

    return uuid_pairings

# Helper function to compare two breakdowns
def pairwise_compare(text1_breakdown: str, 
                     text2_breakdown: str, 
                     target_concept: str,
                     model: str):
    """
    Compare two CGCoT breakdowns to decide which expresses greater level of target concept.
    Args:
        text1_breakdown (str): Breakdown for first text
        text2_breakdown (str): Breakdown for second text
        target_concept (str): Concept name for comparison (e.g., "aversion to Republicans")
    Returns:
        str: "Text1" or "Text2"
        str: Full LLM response for transparency
    """

    comparison_prompt = f""" 
    Description 1: {text1_breakdown}
    Description 2: {text2_breakdown}
    Based on these two Descriptions, which Description expresses greater {target_concept}: Description 1 or Description 2? You must choose one of the descriptions.

    Format your response as follows:
    FINAL ANSWER: <Your choice of "Description 1" or "Description 2">
    JUSTIFICATION: <Your CONCISE reasoning for the choice>
    """

    response = query_llm(comparison_prompt, model=model, 
                         system_message="You are a precise and detail-oriented assistant.")
    
    # Use regex to extract the final answer
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
        # If regex fails, fallback to a direct extraction prompt
        extraction_prompt = f"""
        In the following text, which Description is described to be expressing greater {target_concept}: Description 1 or Description 2? ONLY REPLY WITH "Description 1" or "Description 2". Text: {response}
        """

        extracted_answer = query_llm(extraction_prompt, 
                                     model=model, 
                                     system_message="You are a precise and detail-oriented assistant.")
        extracted_answer = extracted_answer.strip()

        if extracted_answer == "Description 1":
            final_answer = "Text1"
        elif extracted_answer == "Description 2":
            final_answer = "Text2"
        else:
            final_answer = "ERROR"

    return final_answer, response

def generate_pairwise_annotations(uuid_pairings, target_concept=None, 
                                  rate_limit_per_minute=15):
    """
    Run pairwise comparisons on all pairs in uuid_pairings DataFrame.
    
    Args:
        uuid_pairings (pd.DataFrame): DataFrame with breakdown1 and breakdown2 columns
        target_concept (str): The concept to compare for (default: "objectivity")
    
    Returns:
        pd.DataFrame: Original dataframe with added 'decision' and 'justification' columns
    """
    if target_concept is None:
        raise ValueError("target_concept must be specified.")

    decisions = []
    justifications = []
    sleep_time = 2 * (60.0 / rate_limit_per_minute)  # Up to 2 requests per comparison
    
    for idx, row in uuid_pairings.iterrows():
        breakdown1 = row['breakdown1']
        breakdown2 = row['breakdown2']
        
        # Run pairwise comparison
        decision, justification = pairwise_compare(breakdown1, breakdown2, target_concept)
        
        decisions.append(decision)
        justifications.append(justification)
        
        # Print progress every 50 iterations
        if (idx + 1) % 50 == 0:
            print(f"Completed {idx + 1}/{len(uuid_pairings)} comparisons")

        # Sleep to respect rate limit
        if idx < len(uuid_pairings) - 1:
            time.sleep(sleep_time)

    # Add results to dataframe
    uuid_pairings['decision'] = decisions
    uuid_pairings['justification'] = justifications
    
    return uuid_pairings

def generate_pairwise_annotations_parallel(uuid_pairings, target_concept=None, max_workers=8):
    """
    Run pairwise comparisons on all pairs in uuid_pairings DataFrame in parallel.
    Args:
        uuid_pairings (pd.DataFrame): DataFrame with breakdown1 and breakdown2 columns
        target_concept (str): The concept to compare for (default: "objectivity")
        max_workers (int): Number of threads to use
    Returns:
        pd.DataFrame: Original dataframe with added 'decision' and 'justification' columns
    """
    if target_concept is None:
        raise ValueError("target_concept must be specified.")

    results = [None] * len(uuid_pairings)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(pairwise_compare, row['breakdown1'], row['breakdown2'], target_concept): idx
            for idx, row in uuid_pairings.iterrows()
        }
        for i, future in enumerate(as_completed(futures)):
            idx = futures[future]
            try:
                decision, justification = future.result()
            except Exception as e:
                decision, justification = "ERROR", str(e)
            results[idx] = (decision, justification)
            # Print progress every 50 iterations
            if (i + 1) % 50 == 0:
                print(f"Completed {i + 1}/{len(uuid_pairings)} comparisons")
    uuid_pairings['decision'] = [r[0] for r in results]
    uuid_pairings['justification'] = [r[1] for r in results]
    return uuid_pairings

def bradley_terry_scores(original_df, row_id, pairwise_df):
    """
    Compute Bradley-Terry scores from pairwise comparison results.
    
    Args:
        original_df (pd.DataFrame): Original DataFrame with item metadata
        row_id (int): The index of the row to compute scores for
        pairwise_df (pd.DataFrame): DataFrame with pairwise comparison results

    Returns:
        pd.Series: original_df with added 'Bradley_Terry_Score' column
    """

    # Filter out invalid decisions
    valid_df = pairwise_df[pairwise_df['decision'].isin(['Text1', 'Text2'])]

    # Prepare data for Bradley-Terry model
    item_to_idx = {item: idx for idx, item in enumerate(original_df[row_id].tolist())}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    comparisons = []
    for _, row in valid_df.iterrows():
        item1_idx = item_to_idx[row['item1']]
        item2_idx = item_to_idx[row['item2']]
        decision = row['decision']
        
        if decision == 'Text1':
            comparisons.append((item1_idx, item2_idx))
        elif decision == 'Text2':
            comparisons.append((item2_idx, item1_idx))

    if not comparisons:
        raise ValueError("No valid comparisons to compute Bradley-Terry scores.")

    # Fit Bradley-Terry model
    bt_scores = choix.ilsr_pairwise(len(item_to_idx), comparisons, alpha=0.1)

    # Normalize scores to [0, 1]
    bt_scores = (bt_scores - bt_scores.min()) / (bt_scores.max() - bt_scores.min())

    # Add scores to original DataFrame
    original_df['Bradley_Terry_Score'] = [bt_scores[item_to_idx[uuid]] for uuid in original_df[row_id]]

    print(f"Bradley-Terry model fitted with {len(comparisons)} comparisons")
    print(f"Mean objectivity score: {original_df['Bradley_Terry_Score'].mean():.3f}")
    print(f"Std objectivity score: {original_df['Bradley_Terry_Score'].std():.3f}")

    return original_df

def summarize_scores(df, 
                     text_col=None,
                     score_col='Bradley_Terry_Score'):
    """
    Summarize Bradley-Terry scores with basic statistics and print important descriptives.
    
    Args:
        df (pd.DataFrame): DataFrame with Bradley-Terry scores
        text_col (str): Column name for text that was scored
        score_col (str): Column name for scores
    
    Returns:
        dict: Summary statistics
    """

    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in DataFrame.")
    
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    
    # Check the range of your scores
    print(f"Score range: {df[score_col].min():.3f} to {df[score_col].max():.3f}")

    # Look at percentiles for interpretation
    print(f"25th percentile: {df[score_col].quantile(0.25):.3f}")
    print(f"50th percentile (median): {df[score_col].quantile(0.50):.3f}")
    print(f"75th percentile: {df[score_col].quantile(0.75):.3f}")

    # Compare specific items
    df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    top_score = df.iloc[0]
    low_score = df.iloc[-1]
    print(f"\nHighest scoring item on the construct (score: {top_score[score_col]:.3f}):")
    print(top_score[text_col])
    print(f"\nLowest scoring item on the construct (score: {low_score[score_col]:.3f}):")
    print(low_score[text_col]) 

    summary = {
        'mean': df[score_col].mean(),
        'median': df[score_col].median(),
        'std': df[score_col].std(),
        'min': df[score_col].min(),
        'max': df[score_col].max(),
        'count': df[score_col].count()
    }
    
    return summary

def plot_score_distribution(results_df,
                                  score_col='Bradley_Terry_Score', title='Distribution of Bradley-Terry Scores'):
    """
    Plots an interactive histogram of Bradley-Terry scores using Plotly Express.

    Args:
        results_df (pd.DataFrame): DataFrame containing the scores.
        score_col (str): Column name for the Bradley-Terry scores.
        title (str): Title for the plot.
    """
    fig = px.histogram(
        results_df,
        x=score_col,
        nbins=30,
        title=title,
        labels={score_col: 'Bradley-Terry Score'},
        color_discrete_sequence=['skyblue']
    )
    fig.add_vline(
        x=results_df[score_col].mean(),
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mean ({results_df[score_col].mean():.3f})",
        annotation_position="top right"
    )
    fig.update_layout(
        yaxis_title='Frequency',
        bargap=0.01,
        template='ggplot2'
    )
    fig.show()

def check_transitivity(df, annotator_col):
    """
    Check transitivity violations for a given annotator.
    Optimized version using dictionary lookups and early filtering.
    Arguments:
        df: DataFrame containing pairwise comparisons.
        annotator_col: Column name of the annotator to check.
    Returns:
        transitivity_score: percentage of transitive triples 
        violations: number of violations
        total_triples: number of triples evaluated
    """
    violations = 0
    total_triples = 0
    
    # Get all unique items
    items = list(set(df['item1'].unique()) | set(df['item2'].unique()))
    
    # Create a dictionary for fast lookup of comparisons
    comparisons = {}
    for _, row in df.iterrows():
        key1 = (row['item1'], row['item2'])
        key2 = (row['item2'], row['item1'])  # reverse order
        
        # Store decision for both orderings
        comparisons[key1] = row[annotator_col]
        comparisons[key2] = 1 - row[annotator_col]  # flip decision for reverse order
    
    # Check all possible triples
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            for k in range(j+1, len(items)):
                item_a, item_b, item_c = items[i], items[j], items[k]
                
                # Check if all three comparisons exist using dictionary lookup
                ab_key = (item_a, item_b)
                bc_key = (item_b, item_c)
                ac_key = (item_a, item_c)
                
                if all(key in comparisons for key in [ab_key, bc_key, ac_key]):
                    total_triples += 1
                    
                    ab_decision = comparisons[ab_key]
                    bc_decision = comparisons[bc_key]
                    ac_decision = comparisons[ac_key]
                    
                    # Check transitivity violations
                    if (ab_decision == 1 and bc_decision == 1 and ac_decision == 0) or \
                       (ab_decision == 0 and bc_decision == 0 and ac_decision == 1):
                        violations += 1
    
    transitivity_score = 1 - (violations / total_triples) if total_triples > 0 else 0
    return transitivity_score, violations, total_triples