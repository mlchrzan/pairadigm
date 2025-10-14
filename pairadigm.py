import pandas as pd
import itertools
import random
import choix
from google import genai
from google.genai import types
import openai 
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Union
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
import plotly.graph_objects as go
import time
import anthropic
import warnings
import pickle
from pathlib import Path

# AltTest Code is from https://github.com/nitaytech/AltTest 
import numpy as np
from scipy.stats import ttest_1samp
from typing import List, Dict, Any, Callable, Union, Tuple

##############################
# LLMClient class
##############################

class LLMClient:
    """
    Unified LLM client supporting multiple backends.
    
    Parameters
    ----------
    api_key : str, optional
        API key for the LLM service. If None, reads from environment
    model_name : str
        Model identifier (e.g., 'gemini-2.0-flash-exp', 'gpt-4o', 'claude-sonnet-4')
    provider : str, optional
        Force specific provider ('google', 'openai', 'anthropic'). 
        If None, infers from model_name
    """
    
    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = 'gemini-2.0-flash-exp',
            provider: Optional[str] = None):

        self.model_name = model_name
        self.provider = provider or self._infer_provider(model_name)
        self.api_key = api_key or self._get_api_key()
        self.client = self._initialize_client()
    
    def _infer_provider(self, model_name: str) -> str:
        """Infer provider from model name."""
        if 'gemini' in model_name.lower():
            return 'google'
        elif 'gpt' in model_name.lower():
            return 'openai'
        elif 'claude' in model_name.lower():
            return 'anthropic'
        else:
            raise ValueError(
                f"Cannot infer provider from model_name '{model_name}'. "
                "Please specify provider explicitly."
            )
    
    def _get_api_key(self) -> str:
        """Get API key from environment based on provider."""
        env_vars = {
            'google': 'GENAI_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY'
        }
        
        env_var = env_vars.get(self.provider)
        if not env_var:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key not found. Set {env_var} environment variable."
            )
        
        return api_key
    
    def _initialize_client(self):
        """Initialize the appropriate client."""
        if self.provider == 'google':
            from google import genai
            return genai.Client(api_key=self.api_key)
        
        elif self.provider == 'openai':
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        
        elif self.provider == 'anthropic':
            from anthropic import Anthropic
            return Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(
            self,
            prompt: str,
            system_message: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: int = 500) -> str:
        """
        Generate text using the LLM.
        
        Parameters
        ----------
        prompt : str
            User prompt
        system_message : str, optional
            System instruction
        temperature : float
            Sampling temperature
        max_tokens : int
            Maximum tokens to generate
            
        Returns
        -------
        str
            Generated text
        """
        if system_message is None:
            system_message = (
                "You are a precise and detail-oriented assistant specializing "
                "in analyzing text for specific concepts and constructs."
            )
        
        if self.provider == 'google':
            return self._generate_google(prompt, system_message, temperature)
        
        elif self.provider == 'openai':
            return self._generate_openai(prompt, system_message, temperature, max_tokens)
        
        elif self.provider == 'anthropic':
            return self._generate_anthropic(prompt, system_message, temperature, max_tokens)
    
    def _generate_google(
            self,
            prompt: str,
            system_message: str,
            temperature: float) -> str:
        """Generate using Google GenAI."""
        from google.genai import types
        
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=temperature
            ),
            contents=prompt
        )
        return response.text
    
    def _generate_openai(
            self,
            prompt: str,
            system_message: str,
            temperature: float,
            max_tokens: int) -> str:
        """Generate using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Anthropic."""
        response = self.client.messages.create(
            model=self.model_name,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text
    
##############################
# Pairadigm class
############################## 
class Pairadigm:
    def __init__(self, 
                 data: pd.DataFrame, 
                 item_id_name: Optional[str] = None,  
                 text_name: Optional[str] = None, 
                 paired: bool = False,
                 item_id_cols: Optional[List[str]] = None,
                 item_text_cols: Optional[List[str]] = None,
                 annotated: bool = False,
                 annotator_cols: Optional[List[str]] = None,
                 cgcot_prompts: Optional[List[str]] = None, 
                 model_name: str = 'gemini-2.0-flash-exp', 
                 api_key: Optional[str] = None, 
                 target_concept: Optional[str] = None): 
        """
        Main class for Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation.
        
        Supports flexible workflows:
        1. Start with list of items -> generate breakdowns -> pair -> annotate -> score -> validate
        2. Start with paired items -> generate breakdowns -> annotate -> score -> validate
        3. Start with human-annotated pairs -> generate breakdowns -> annotate -> score -> compare -> validate
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with items to compare
        item_id_name : str
            Column name for unique item identifiers
        text_name : str, optional
            Column name for item text/content
        item_id_cols : List[str], optional
            For paired data, list of two column names for the paired item IDs
        item_text_cols : List[str], optional
            For paired data, list of two column names for the paired item texts
        paired : bool, default=False
            Whether the input data is already paired
        annotated : bool, default=False
            Whether the input data already contains human annotations
        annotator_cols : List[str], optional
            For annotated data, list of column names containing human annotations
        cgcot_prompts : List[str], optional
            CGCoT prompt templates for breakdowns
        model_name : str, default='gemini-2.0-flash-exp'
            LLM model to use
        api_key : str, optional
            API key for LLM service
        target_concept : str, optional
            The concept to evaluate (e.g., "objectivity", "political bias")
        """
        
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        # Make sure the necessary columns exist if the data is a list of items
        if not paired:
            if item_id_name not in data.columns:
                raise ValueError(f"Column '{item_id_name}' not found in DataFrame")
            if text_name and text_name not in data.columns:
                raise ValueError(f"Column '{text_name}' not found in DataFrame")
            
        # Make sure the necessary columns exist if the data is paired
        if paired:
            if item_id_cols is None or len(item_id_cols) != 2:
                raise ValueError("For paired data, item_id_cols must be a list of two column names representing the paired items")
            for col in item_id_cols:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
            if item_text_cols is None or len(item_text_cols) != 2:
                raise ValueError("For paired data, item_text_cols must be a list of two column names representing the paired items' text")  
            for col in item_text_cols:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                
        # Make sure the necessary columns exist if the data is annotated
        if annotated:
            if annotator_cols is None or len(annotator_cols) < 1:
                raise ValueError("For annotated data, annotator_cols must be a list of column names containing human annotations")
            for col in annotator_cols:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                
        if annotated and not paired:
            raise ValueError("If data is annotated, it must also be paired (annotated=True). Please see example structure.")
        # AT SOME POINT INCLUDE AN EXAMPLE ... BUT NOT NOW

        if cgcot_prompts is None or not isinstance(cgcot_prompts, list) or len(cgcot_prompts) == 0:
            warnings.warn("cgcot_prompts must be a non-empty list of prompt templates. Some methods may not work until this is set. You can set the CGCOT prompts using .set_cgcot_prompts()", UserWarning)
        if target_concept is None:
            raise ValueError("target_concept must be specified")
        
        self.data = data.copy()
        self.item_id_name = item_id_name
        self.text_name = text_name
        self.paired = paired
        self.annotated = annotated
        self.item_id_cols = item_id_cols
        self.item_text_cols = item_text_cols
        self.annotator_cols = annotator_cols
        self.cgcot_prompts = cgcot_prompts
        self.model = model_name
        self.target_concept = target_concept
        
        # Initialize LLM client
        self.client = LLMClient(api_key=api_key, model_name=model_name)
        
        # Initialize result storage
        self.pairwise_df: Optional[pd.DataFrame] = None
        if paired:
            self.pairwise_df = data.copy()
        self.scored_df: Optional[pd.DataFrame] = None
        self.validation_results: Optional[Dict] = None

    def set_cgcot_prompts(self, prompts: Union[List[str], str]):
        """
        Update CGCoT prompt templates by either passing a list or a file path.
        
        Parameters
        ----------
        prompts : List[str] or str
            Either a list of CGCoT prompt templates or a file path to a text file
            containing prompts (one per line, separated by blank lines or delimiters)
        """
        if isinstance(prompts, str):
            # Treat as file path
            try:
                with open(prompts, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Split by double newlines (blank lines) or by triple dashes
                if '\n\n' in content:
                    prompt_list = [p.strip() for p in content.split('\n\n') if p.strip()]
                elif '---' in content:
                    prompt_list = [p.strip() for p in content.split('---') if p.strip()]
                else:
                    # Fallback: treat each line as a separate prompt
                    prompt_list = [line.strip() for line in content.split('\n') if line.strip()]
                
                if len(prompt_list) == 0:
                    raise ValueError("No valid prompts found in file")
                    
            except FileNotFoundError:
                raise FileNotFoundError(f"Prompt file not found: {prompts}")
            except Exception as e:
                raise ValueError(f"Error reading prompt file: {e}")

            # Validate the NEW prompts, not the old ones
            if self._validate_prompts(prompt_list):  
                self.cgcot_prompts = prompt_list
            
        elif isinstance(prompts, list):
            if len(prompts) == 0:
                raise ValueError("prompts must be a non-empty list of prompt templates")
            # Validate the NEW prompts, not the old ones
            if self._validate_prompts(prompts):
                self.cgcot_prompts = prompts
            
        else:
            raise TypeError("prompts must be either a list of strings or a file path string")

    def _validate_prompts(self, prompts: List[str]) -> bool:
        """
        Validate that prompts are properly formatted.
        
        Parameters
        ----------
        prompts : List[str]
            List of prompt templates to validate
            
        Returns
        -------
        bool
            True if prompts are valid
            
        Raises
        ------
        ValueError
            If prompts are invalid
        """
        if not prompts or not isinstance(prompts, list):
            raise ValueError("Prompts must be a non-empty list")
        
        for i, prompt in enumerate(prompts):
            if '{text}' not in prompt:
                raise ValueError(f"Prompt {i+1} is missing {{text}} placeholder: {prompt[:50]}...")
        
        return True

################################
# GENERATE CGCOT BREAKDOWNS AND PAIRWISE ANNOTATIONS
################################

    def _generate_cgcot_breakdown(
            self,
            text, 
            rate_limit_per_minute=None) -> str:
        """
        Generate concept-specific breakdown for a given text using CGCoT prompts.
        Args:
            text (str): The text to analyze.
        Returns:
            str: Concatenated concept-specific breakdown
        """
        breakdown = [f"Original Text: {text}"]
        prev_answers = []
        sleep_time = 0

        if rate_limit_per_minute:
            sleep_time = 60.0 / rate_limit_per_minute
        
        for i, prompt_template in enumerate(self.cgcot_prompts):
            full_prompt = prompt_template.format(text=text, previous_answers="\n".join(prev_answers))
            response = self.client.generate(
                prompt=full_prompt,
                temperature=0.0
            )
            prev_answers.append(response)
            breakdown.append(f"Prompt {i+1} response: {response}")
            
            if i < len(self.cgcot_prompts) - 1:
                time.sleep(sleep_time)  # Wait to avoid rate limit

        return "\n".join(breakdown)
    
    def generate_breakdowns(
            self,
            max_workers=8, 
            rate_limit_per_minute=None,
            update_dataframe=True) -> Dict[Union[str, int], str]:
        
        """
        Generate CGCoT breakdowns for all items in the DataFrame.
        Args:
            max_workers (int, optional): Number of parallel workers. Defaults to 8.
            rate_limit_per_minute (int, optional): Rate limit for LLM calls.
            update_dataframe (bool, optional): If True, adds breakdowns to self.data.
        Returns:
            Dict[Union[str, int], str]: Mapping of item IDs to breakdowns.
            Also updates self.data with a new 'CGCoT_Breakdown' column if update_dataframe is True.
        """

        if self.paired:
            raise ValueError("Data is marked as paired. generate_breakdowns() should only be called on unpaired item lists. Use generate_breakdowns_from_paired() instead.")

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_cgcot_breakdown, 
                    row[self.text_name],
                    rate_limit_per_minute): row[self.item_id_name]
                for _, row in self.data.iterrows()
            }
            for future in as_completed(futures):
                uuid = futures[future]
                try:
                    results[uuid] = future.result()
                except Exception as e:
                    results[uuid] = f"ERROR: {e}"

        if update_dataframe:
            self.data['CGCoT_Breakdown'] = self.data[self.item_id_name].map(results)
    
        return results
    
    def generate_breakdowns_from_paired(
            self, 
            max_workers: int = 8,
            rate_limit_per_minute: Optional[int] = None,
            update_pairwise_df: bool = True) -> Dict[Union[str, int], str]:
        """
        Generate CGCoT breakdowns for all unique items in paired DataFrame.
        Assumes self.data/self.pairwise_df contains paired format with item1_id, item2_id, item1_text, item2_text columns.
        
        Parameters
        ----------
        max_workers : int, default=8
            Number of parallel workers
        rate_limit_per_minute : int, optional
            Rate limit for LLM calls
        update_pairwise_df : bool, default=True
            If True, adds breakdown1 and breakdown2 columns to self.pairwise_df
            
        Returns
        -------
        Dict[Union[str, int], str]
            Mapping of item IDs to breakdowns
        """
        if not self.paired:
            raise ValueError("Data is not marked as paired. generate_breakdowns_from_paired() should only be called on paired item lists.")

        # Use pairwise_df if available, otherwise use self.data
        source_df = self.pairwise_df if self.pairwise_df is not None else self.data
        
        if len(self.item_id_cols) != 2:
            raise ValueError("item_id_cols must contain exactly 2 column names for paired data")
        
        # Get column names for item IDs and texts
        item1_id_col, item2_id_col = self.item_id_cols
        item1_text_col, item2_text_col = self.item_text_cols 
        
        # Extract unique items and their texts
        items_data = []
        
        # Add item1 data
        item1_data = source_df[[item1_id_col, item1_text_col]].rename(columns={
            item1_id_col: self.item_id_name,
            item1_text_col: 'text'
        })
        items_data.append(item1_data)
        
        # Add item2 data
        item2_data = source_df[[item2_id_col, item2_text_col]].rename(columns={
            item2_id_col: self.item_id_name,
            item2_text_col: 'text'
        })
        items_data.append(item2_data)
        
        # Combine and get unique items
        items_df = pd.concat(items_data, ignore_index=True).drop_duplicates(subset=[self.item_id_name])
        
        # Create text mapping
        text_mapping = dict(zip(items_df[self.item_id_name], items_df['text']))
        
        # Generate breakdowns for unique items
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_cgcot_breakdown, 
                    text_mapping[item_id],
                    rate_limit_per_minute): item_id
                for item_id in items_df[self.item_id_name]
            }
            
            for future in as_completed(futures):
                item_id = futures[future]
                try:
                    results[item_id] = future.result()
                except Exception as e:
                    results[item_id] = f"ERROR: {e}"
        
        # Update pairwise_df with breakdown columns if requested
        if update_pairwise_df and self.pairwise_df is not None:
            self.pairwise_df['breakdown1'] = self.pairwise_df[item1_id_col].map(results)
            self.pairwise_df['breakdown2'] = self.pairwise_df[item2_id_col].map(results)
        elif update_pairwise_df and self.pairwise_df is None:
            # Create pairwise_df from self.data with breakdown columns
            self.pairwise_df = source_df.copy()
            self.pairwise_df['breakdown1'] = self.pairwise_df[item1_id_col].map(results)
            self.pairwise_df['breakdown2'] = self.pairwise_df[item2_id_col].map(results)

        return results

    # Helper function to create pairings ensuring connectivity and minimum pairs per item
    @staticmethod
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

    def generate_pairings(
            self, 
            num_pairs_per_item=10, 
            random_seed=42,
            breakdowns=False,
            update_classObject=True) -> pd.DataFrame:
        """ 
        Generate pairings for items in a DataFrame column.
        Args:
            num_pairs_per_item (int): Minimum pairs per item. Defaults to 10.
            random_seed (int, optional): For reproducibility. Defaults to 42.
            breakdowns (bool, optional): If True, self.data has CGCOT_Breakdown column from generate_breakdowns(). Defaults to False.
            update_classObject (bool, optional): If True, updates self.pairwise_df. Defaults to True.
        Returns:
            pd.DataFrame: DataFrame with pairings and associated breakdowns.
        """

        # Pair items
        uuid_pairings = self.pair_items(
            self.data[self.item_id_name].tolist(),
            num_pairs_per_item,
            random_seed)
        
        # Map breakdowns to pairings if present
        if breakdowns:

            if "CGCoT_Breakdown" not in self.data.columns:
                raise ValueError("Column 'CGCoT_Breakdown' not found in DataFrame. Generate them using generate_breakdowns() first.")

            uuid_to_desc = dict(zip(self.data[self.item_id_name], self.data['CGCoT_Breakdown']))
            uuid_pairings['breakdown1'] = uuid_pairings['item1'].map(uuid_to_desc)
            uuid_pairings['breakdown2'] = uuid_pairings['item2'].map(uuid_to_desc)

            if update_classObject:
                self.pairwise_df = uuid_pairings

        return uuid_pairings

    # Helper function to compare two breakdowns
    @staticmethod
    def pairwise_compare(
        text1_breakdown: str, 
        text2_breakdown: str, 
        target_concept: str,
        client: LLMClient):
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

        response = client.generate(
            prompt=comparison_prompt,
            system_message="You are a precise and detail-oriented assistant.",
            temperature=0.0
        )

        # response = query_llm(comparison_prompt, model=model, 
        #                     system_message="You are a precise and detail-oriented assistant.")
        
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

            extracted_answer = client.generate(
                prompt=extraction_prompt,
                system_message="You are a precise and detail-oriented assistant.",
                temperature=0.0
            )

            # extracted_answer = query_llm(extraction_prompt, 
            #                             model=model, 
            #                             system_message="You are a precise and detail-oriented assistant.")
            extracted_answer = extracted_answer.strip()

            if extracted_answer == "Description 1":
                final_answer = "Text1"
            elif extracted_answer == "Description 2":
                final_answer = "Text2"
            else:
                final_answer = "ERROR"

        return final_answer, response

    def generate_pairwise_annotations(
        self,
        max_workers=8,
        update_classObject=True) -> pd.DataFrame:
        """
        Run pairwise comparisons on all pairs in the pairwise_df DataFrame in parallel.
        
        Args:
            max_workers (int): Number of threads to use
            update_classObject (bool, optional): If True, updates self.pairwise_df with results. Defaults to True.
        Returns:
            pd.DataFrame: Original dataframe with added 'decision' and 'justification' columns
        """

        if self.pairwise_df is None:
            raise ValueError("No pairwise_df found in the object. Generate pairings with breakdowns first using generate_pairings(breakdowns=True).")

        result_df = self.pairwise_df.copy()
        results = [None] * len(result_df)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.pairwise_compare, 
                    row['breakdown1'], 
                    row['breakdown2'], 
                    self.target_concept,
                    self.client
                ): idx
                for idx, row in result_df.iterrows()
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
                    print(f"Completed {i + 1}/{len(result_df)} comparisons")
        
        result_df['decision'] = [r[0] for r in results]
        result_df['justification'] = [r[1] for r in results]
        
        # Update instance if requested
        if update_classObject:
            self.pairwise_df = result_df
        
        return result_df

################################
# EVALUATION AND VALIDATION
################################
    @staticmethod
    def by_procedure(p_values: List[float], q: float) -> List[int]:
        """
        Perform Benjamini-Yekutieli procedure for FDR control under arbitrary dependence.
        Args:
            p_values (List[float]): List of p-values
            q (float): Desired FDR level
        Returns:
            List[int]: Indices of rejected hypotheses
        """
        
        # Convert p_values to a numpy array for easier manipulation
        p_values = np.array(p_values, dtype=float)
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvals = p_values[sorted_indices]

        # Compute the harmonic sum H_m = 1 + 1/2 + ... + 1/m
        H_m = np.sum(1.0 / np.arange(1, m + 1))

        # Compute the BY thresholds for each rank i
        by_thresholds = (np.arange(1, m + 1) / m) * (q / H_m)

        max_i = -1
        for i in range(m):
            if sorted_pvals[i] <= by_thresholds[i]:
                max_i = i
        if max_i == -1:
            return []
        rejected_sorted_indices = sorted_indices[:max_i + 1]
        return list(rejected_sorted_indices)

    @staticmethod
    def accuracy(pred: Any, annotations: List[Any]) -> float:
        return float(np.mean([pred == ann for ann in annotations]))

    @staticmethod
    def neg_rmse(pred: Union[int, float], annotations: List[Union[int, float]]) -> float:
        return -1 * float(np.sqrt(np.mean([(pred - ann) ** 2 for ann in annotations])))

    @staticmethod
    def sim(pred: str, annotations: List[str], similarity_func: Callable) -> float:
        return float(np.mean([similarity_func(pred, ann) for ann in annotations]))

    @staticmethod
    def ttest(indicators, epsilon: float) -> float:
        return ttest_1samp(indicators, epsilon, alternative='less').pvalue
    
    # Function to turn the annotations into a dictionary ready for alt_test
    def prep_for_alt_test(self) -> Tuple[Dict[Union[int, str], Any], 
                                         Dict[Union[int, str], 
                                              Dict[Union[int, str], Any]]]:
        """
        Prepare annotations from class data for alt_test function.
        
        Returns:
            Tuple[Dict[Union[int, str], Any], Dict[Union[int, str], Dict[Union[int, str], Any]]]: 
                (llm_annotations, humans_annotations)
        """
        if not self.annotated:
            raise ValueError("Data must be annotated (annotated=True) to prepare for alt_test")
        
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison data found. Run pairwise annotations first.")
        
        if 'decision' not in self.pairwise_df.columns:
            raise ValueError("No 'decision' column found. Run generate_pairwise_annotations() first.")
        
        # Use class attributes for column names
        item1_id_col, item2_id_col = self.item_id_cols
        
        llm_annotations = {}
        humans_annotations = {col: {} for col in self.annotator_cols}
        
        for _, row in self.pairwise_df.iterrows():
            item1_id = row[item1_id_col]
            item2_id = row[item2_id_col]
            decision = row['decision']
            
            # Process LLM annotations
            if decision == 'Text1':
                llm_annotations[item1_id] = llm_annotations.get(item1_id, 0) + 1
                llm_annotations[item2_id] = llm_annotations.get(item2_id, 0)
            elif decision == 'Text2':
                llm_annotations[item2_id] = llm_annotations.get(item2_id, 0) + 1
                llm_annotations[item1_id] = llm_annotations.get(item1_id, 0)
            else:
                continue
            
            # Process human annotations
            for col in self.annotator_cols:
                if col not in row or pd.isna(row[col]):
                    continue
                    
                human_decision = row[col]
                if human_decision == 'Text1':
                    humans_annotations[col][item1_id] = humans_annotations[col].get(item1_id, 0) + 1
                    humans_annotations[col][item2_id] = humans_annotations[col].get(item2_id, 0)
                elif human_decision == 'Text2':
                    humans_annotations[col][item2_id] = humans_annotations[col].get(item2_id, 0) + 1
                    humans_annotations[col][item1_id] = humans_annotations[col].get(item1_id, 0)
                else:
                    continue
        
        return llm_annotations, humans_annotations

    def alt_test(
        self,
        llm_annotations: Optional[Dict[Union[int, str], Any]] = None,
        humans_annotations: Optional[Dict[Union[int, str], Dict[Union[int, str], Any]]] = None,
        scoring_function: Union[str, Callable] = 'accuracy',
        epsilon: float = 0.1, # Value which the authors claim as an adjustment for cost of well-trained annotators
        q_fdr: float = 0.05,
        min_humans_per_instance: int = 2,
        min_instances_per_human: int = 30):
    
        """
        Perform the alternative hypothesis test to compare LLM annotations against human annotations.
        Args:
            llm_annotations (Optional[Dict[Union[int, str], Any]]): Mapping of instance IDs to LLM annotations. 
                If None, will be generated using the provided data for the instance and the prep_for_alt_test().
            humans_annotations (Optional[Dict[Union[int, str], Dict[Union[int, str], Any]]]): Mapping of human IDs to their annotations 
                (which are mappings of instance IDs to annotations). If None, will be generated using prep_for_alt_test().
            scoring_function (Union[str, Callable], optional): Scoring function to use ('accuracy', 'neg_rmse', or custom). Defaults to 'accuracy'.
            epsilon (float, optional): Adjustment value for t-test. Defaults to 0.1.
            q_fdr (float, optional): FDR level for BY procedure. Defaults to 0.05. Stands for False Discovery Rate.
            min_humans_per_instance (int, optional): Minimum annotators per instance to include. Defaults to 2.
            min_instances_per_human (int, optional): Minimum instances per annotator to include. Defaults to 30.
        Returns:
            Tuple[float, float]: (winning_rate, advantage_prob)
        """
        
        # Generate annotations if not provided
        if llm_annotations is None or humans_annotations is None:
            generated_llm_annotations, generated_humans_annotations = self.prep_for_alt_test()
            
            if llm_annotations is None:
                llm_annotations = generated_llm_annotations
            if humans_annotations is None:
                humans_annotations = generated_humans_annotations
        
        # prepare alignment scoring function
        if isinstance(scoring_function, str):
            if scoring_function == 'accuracy':
                scoring_function = self.accuracy
            elif scoring_function == 'neg_rmse':
                scoring_function = self.neg_rmse
            else:
                raise ValueError("Unknown scoring function")
        else:
            scoring_function = scoring_function

        # prepare sets - i_set has humans as keys, h_set has instances as keys
        i_set, h_set = {}, {}
        for h, anns in humans_annotations.items():
            i_set[h] = list(anns.keys())
            for i, ann in anns.items():
                if i not in h_set:
                    h_set[i] = []
                h_set[i].append(h)

        # remove instances with less than min_humans_per_instance
        instances_to_keep = {i for i in h_set if len(h_set[i]) >= min_humans_per_instance and i in llm_annotations}
        if len(instances_to_keep) < len(h_set):
            print(f"Dropped {len(h_set) - len(instances_to_keep)} instances with less than {min_humans_per_instance} annotators.")
        i_set = {h: [i for i in i_set[h] if i in instances_to_keep] for h in i_set}
        h_set = {i: h_set[i] for i in h_set if i in instances_to_keep}

        p_values, advantage_probs, humans = [], [], []
        for excluded_h in humans_annotations:
            llm_indicators = []
            excluded_indicators = []
            instances = [i for i in i_set[excluded_h] if i in llm_annotations]
            if len(instances) < min_instances_per_human:
                print(f"Skipping annotator {excluded_h} with only {len(instances)} instances < {min_instances_per_human}.")
                continue

            for i in instances:
                human_ann = humans_annotations[excluded_h][i]
                llm_ann = llm_annotations[i]
                remaining_anns = [humans_annotations[h][i] for h in h_set[i] if h != excluded_h]
                human_score = scoring_function(human_ann, remaining_anns)
                llm_score = scoring_function(llm_ann, remaining_anns)
                llm_indicators.append(1 if llm_score >= human_score else 0)
                excluded_indicators.append(1 if human_score >= llm_score else 0)

            diff_indicators = [exc_ind - llm_ind for exc_ind, llm_ind in zip(excluded_indicators, llm_indicators)]
            p_values.append(self.ttest(diff_indicators, epsilon))
            advantage_probs.append(float(np.mean(llm_indicators)))
            humans.append(excluded_h)

        rejected_indices = self.by_procedure(p_values, q_fdr)
        advantage_prob = float(np.mean(advantage_probs))
        winning_rate = len(rejected_indices) / len(humans)
        return winning_rate, advantage_prob
    
    def check_transitivity(self, annotator_cols=None):
        """
        Check transitivity violations for annotators.
        
        Arguments:
            annotator_cols: List of column names to check, or None to check all available annotators.
                        If None, will check 'decision' column (LLM) and any human annotator columns.
        
        Returns:
            dict: Dictionary with annotator names as keys and tuples of 
                (transitivity_score, violations, total_triples) as values
        """
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison data found. Run generate_pairwise_annotations() first.")
        
        df = self.pairwise_df
        
        # Determine which annotators to check
        if annotator_cols is None:
            # Check LLM decision column and any human annotator columns
            cols_to_check = []
            if 'decision' in df.columns:
                cols_to_check.append('decision')
            if self.annotated and self.annotator_cols:
                cols_to_check.extend(self.annotator_cols)
            if not cols_to_check:
                raise ValueError("No annotator columns found to check transitivity.")
        else:
            # Use provided columns
            cols_to_check = annotator_cols if isinstance(annotator_cols, list) else [annotator_cols]

            # If the "decision" column is included, ensure it's checked
            if 'decision' in cols_to_check and 'decision' not in df.columns:
                raise ValueError("Column 'decision' not found in pairwise DataFrame.")
            
            # If the decision column exists in the DataFrame but not in cols_to_check, add it
            if 'decision' in df.columns and 'decision' not in cols_to_check:
                cols_to_check.insert(0, 'decision')
                
            # Validate that all specified columns exist in the DataFrame
            for col in cols_to_check:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in pairwise DataFrame.")
        
        results = {}
        
        for annotator_col in cols_to_check:
            violations = 0
            total_triples = 0
            
            # Get all unique items
            items = list(set(df['item1'].unique()) | set(df['item2'].unique()))
            
            # Create a dictionary for fast lookup of comparisons
            comparisons = {}
            for _, row in df.iterrows():
                # Skip rows with missing annotations for this annotator
                if pd.isna(row[annotator_col]):
                    continue
                    
                key1 = (row['item1'], row['item2'])
                key2 = (row['item2'], row['item1'])  # reverse order
                
                # Handle different annotation formats
                decision = row[annotator_col]
                if decision == 'Text1':
                    comparisons[key1] = 1
                    comparisons[key2] = 0
                elif decision == 'Text2':
                    comparisons[key1] = 0
                    comparisons[key2] = 1
                elif isinstance(decision, (int, float)) and decision in [0, 1]:
                    # Handle binary numeric annotations
                    comparisons[key1] = int(decision)
                    comparisons[key2] = 1 - int(decision)
                else:
                    # Skip invalid decisions
                    continue
            
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
            results[annotator_col] = (transitivity_score, violations, total_triples)
        
        return results

################################
# SCORING AND SUMMARIZATION
################################

    def score_items(self, 
                    normalization_scale='zero-to-one',
                    update_classObject=True,
                    summarize=True) -> pd.DataFrame:
        """
        Compute Bradley-Terry scores from pairwise comparison results.
        
        Args:
            update_classObject (bool, optional): If True, updates self.scored_df. Defaults to True.

        Returns:
            pd.DataFrame: Original DataFrame with added 'Bradley_Terry_Score' column
        """
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison results found. Run generate_pairwise_annotations() first.")

        # Filter out invalid decisions
        valid_df = self.pairwise_df[self.pairwise_df['decision'].isin(['Text1', 'Text2'])]

        if len(valid_df) == 0:
            raise ValueError("No valid comparisons found to compute Bradley-Terry scores.")

        # Prepare data for Bradley-Terry model, handling different self.data formats for item mapping
        if self.paired:
            # For paired data, collect unique items from both item ID columns
            item1_col, item2_col = self.item_id_cols
            all_items = pd.concat([
                self.data[item1_col],
                self.data[item2_col]
            ]).unique().tolist()
            item_to_idx = {item: idx for idx, item in enumerate(all_items)}
        else:
            # For unpaired data, use the single item ID column
            item_to_idx = {item: idx for idx, item in enumerate(self.data[self.item_id_name].tolist())}

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

        if normalization_scale == 'zero-to-one':
            # Normalize scores to [0, 1]
            bt_scores = (bt_scores - bt_scores.min()) / (bt_scores.max() - bt_scores.min())
        elif normalization_scale == 'negative-one-to-one':
            # Normalize scores to [-1, 1]
            bt_scores = 2 * (bt_scores - bt_scores.min()) / (bt_scores.max() - bt_scores.min()) - 1
        elif normalization_scale == 'none':
            pass  # Keep raw scores
        else:
            raise ValueError("normalization_scale must be 'zero-to-one', 'negative-one-to-one', or 'none'")

        # Add scores to original DataFrame
        scored_df = self.data.copy()
        scored_df['Bradley_Terry_Score'] = [bt_scores[item_to_idx[uuid]] for uuid in scored_df[self.item_id_name]]

        print(f"Bradley-Terry model fitted with {len(comparisons)} comparisons")
        print(f"Mean {self.target_concept} score: {scored_df['Bradley_Terry_Score'].mean():.3f}")
        print(f"Std {self.target_concept} score: {scored_df['Bradley_Terry_Score'].std():.3f}")

        # Update instance if requested
        if update_classObject:
            self.scored_df = scored_df

        if summarize:
            summary = self.summarize_scores(df=scored_df, 
                                            text_col=self.text_name, 
                                            score_col='Bradley_Terry_Score')
            for k, v in summary.items():
                print(f"{k}: {v:.3f}")

        return scored_df

    def summarize_scores(
        self,
        df=None, 
        text_col=None,
        score_col='Bradley_Terry_Score'):
        """
        Summarize Bradley-Terry scores with basic statistics and print important descriptives.
        
        Args:
            df (pd.DataFrame, optional): DataFrame with Bradley-Terry scores. If None, uses self.scored_df
            text_col (str, optional): Column name for text that was scored. If None, uses self.text_name
            score_col (str): Column name for scores
        
        Returns:
            dict: Summary statistics
        """
        
        # Use class attributes as defaults
        if df is None:
            if self.scored_df is None:
                raise ValueError("No scored DataFrame found. Run score_items() first or provide df parameter.")
            df = self.scored_df
        
        if text_col is None:
            if self.text_name is None:
                raise ValueError("No column with item texts is specified. Provide text_col parameter or set text_name in constructor.")
            text_col = self.text_name

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
        df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
        top_score = df_sorted.iloc[0]
        low_score = df_sorted.iloc[-1]
        print(f"\nHighest scoring item on {self.target_concept} (score: {top_score[score_col]:.3f}):")
        print(top_score[text_col])
        print(f"\nLowest scoring item on {self.target_concept} (score: {low_score[score_col]:.3f}):")
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

    def plot_score_distribution(
        self, 
        score_col='Bradley_Terry_Score', 
        title=None,
        nbins=30,
        show_stats=True,
        color='skyblue',
        template='plotly_white',
        return_fig=False):
        """
        Plots an interactive histogram of Bradley-Terry scores using Plotly Express.

        Args:
            score_col (str): Column name for the Bradley-Terry scores.
            title (str, optional): Title for the plot. If None, auto-generates based on target_concept.
            nbins (int): Number of histogram bins.
            show_stats (bool): Whether to show mean line and statistics.
            color (str): Color for histogram bars.
            template (str): Plotly template to use.
            return_fig (bool): Whether to return the figure object instead of showing.
            
        Returns:
            plotly.graph_objects.Figure: If return_fig=True, returns the figure object.
        """
        # Validate inputs
        if self.scored_df is None:
            raise ValueError("No scored DataFrame found. Run score_items() first.")
        
        if score_col not in self.scored_df.columns:
            raise ValueError(f"Column '{score_col}' not found in scored DataFrame.")
        
        # Auto-generate title if not provided
        if title is None:
            title = f'Distribution of {self.target_concept.title()} Scores from Bradley-Terry Model'
        
        # Create histogram
        fig = px.histogram(
            self.scored_df,
            x=score_col,
            nbins=nbins,
            title=title,
            labels={score_col: f'{self.target_concept.title()} Score'},
            color_discrete_sequence=[color],
            marginal="box"  # Add box plot on top
        )
        
        if show_stats:
            mean_score = self.scored_df[score_col].mean()
            median_score = self.scored_df[score_col].median()
            
            # Add mean line
            fig.add_vline(
                x=mean_score,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_score:.3f}",
                annotation_position="top right"
            )
            
            # Add median line
            fig.add_vline(
                x=median_score,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Median: {median_score:.3f}",
                annotation_position="top left"
            )
            
            # Add text box with summary statistics
            stats_text = (
                f"Mean: {mean_score:.3f}<br>"
                f"Median: {median_score:.3f}<br>"
                f"Std: {self.scored_df[score_col].std():.3f}<br>"
                f"Count: {len(self.scored_df)}"
            )
            
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=stats_text,
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )
        
        # Update layout
        fig.update_layout(
            yaxis_title='Frequency',
            bargap=0.02,
            template=template,
            hovermode='x unified',
            showlegend=False
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate=f'<b>{self.target_concept.title()} Score</b>: %{{x}}<br>' +
                        '<b>Count</b>: %{y}<extra></extra>'
        )
        
        if return_fig:
            return fig
        else:
            fig.show()

    def plot_comparison_network(
            self, 
            centrality_measure='pagerank',
            decision_col='decision',
            return_fig=False):
        """
        Plots a network graph of pairwise comparisons using Plotly.

        Args:
            centrality_measure (str): Centrality measure to use. Options:
                'pagerank', 'in_degree', 'out_degree', 'betweenness', 'eigenvector', 'degree'
            return_fig (bool): Whether to return the figure object instead of showing.
            
        Returns:
            plotly.graph_objects.Figure: If return_fig=True, returns the figure object.
        """
        import networkx as nx

        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison results found. Run generate_pairwise_annotations() first.")

        if decision_col not in self.pairwise_df.columns:
            raise ValueError("No 'decision' column found. Run generate_pairwise_annotations() first.")

        # Calculate centrality based on parameter
        centrality_funcs = {
            'pagerank': nx.pagerank,
            'in_degree': nx.in_degree_centrality,
            'out_degree': nx.out_degree_centrality,
            'betweenness': nx.betweenness_centrality,
            'eigenvector': lambda G: nx.eigenvector_centrality(G, max_iter=1000),
            'degree': nx.degree_centrality
        }

        # Create a directed graph
        G = nx.DiGraph()

        # Add edges based on decisions
        for _, row in self.pairwise_df.iterrows():
            if row['decision'] == 'Text1':
                G.add_edge(row['item1'], row['item2'])
            elif row['decision'] == 'Text2':
                G.add_edge(row['item2'], row['item1'])

        if len(G.nodes()) == 0:
            raise ValueError("No valid comparisons found to create network graph.")

        pos = nx.spring_layout(G, seed=42)  # For consistent layout

        # Calculate centrality
            # centrality = nx.degree_centrality(G)
            # node_color = [centrality[node] for node in G.nodes()]
        centrality = centrality_funcs[centrality_measure](G)
        node_color = [centrality[node] for node in G.nodes()]
        
        if centrality_measure not in centrality_funcs:
            raise ValueError(f"Unknown centrality measure: {centrality_measure}")

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        colorbar_title = centrality_measure.replace('_', ' ').title()
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_color,
                size=10,
                colorbar=dict(
                    title=colorbar_title
                ),
                line_width=2),
            text=[str(node) for node in G.nodes()]
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'<br>Pairwise Comparison Network - {self.target_concept.title()}',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="",
                                showarrow=False,
                                xref="paper", yref="paper")],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        if return_fig:
            return fig
        else:
            fig.show()
    
    def save(self, filepath: str):
        """
        Save a Pairadigm object to a file using pickle.
        
        Parameters
        ----------
        filepath : str
            Path where the object should be saved. If no extension is provided,
            '.pkl' will be added automatically.
            
        Examples
        --------
        >>> pairadigm_obj.save('my_analysis.pkl')
        >>> pairadigm_obj.save('results/analysis')  # Saves as 'results/analysis.pkl'
        """
        # Ensure filepath has .pkl extension
        filepath = Path(filepath)
        if filepath.suffix != '.pkl':
            filepath = filepath.with_suffix('.pkl')
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Temporarily remove unpicklable client object
            client = self.client
            self.client = None
            
            # Temporarily remove the api_key if it exists
            if hasattr(self, 'api_key'):
                api_key = self.api_key
                self.api_key = None
            else:
                api_key = None

            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            # Restore client
            self.client = client
            
            print(f"Pairadigm object saved successfully to: {filepath}")
        except Exception as e:
            # Restore client even if save fails
            self.client = client
            if api_key is not None:
                self.api_key = api_key
            raise IOError(f"Failed to save Pairadigm object: {e}")

    @staticmethod
    def load(filepath: str) -> 'Pairadigm':
        """
        Load a Pairadigm object from a pickle file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved Pairadigm object file.
            
        Returns
        -------
        Pairadigm
            The loaded Pairadigm object.
            
        Examples
        --------
        >>> pairadigm_obj = Pairadigm.load('my_analysis.pkl')
        """
        filepath = Path(filepath)
        
        # Try adding .pkl extension if file not found
        if not filepath.exists() and filepath.suffix != '.pkl':
            filepath = filepath.with_suffix('.pkl')
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            
            if not isinstance(obj, Pairadigm):
                raise TypeError("Loaded object is not a Pairadigm instance")
            
            # Recreate the LLM client
            obj.client = LLMClient(model_name=obj.model)
            
            print(f"Pairadigm object loaded successfully from: {filepath}")
            return obj
        except Exception as e:
            raise IOError(f"Failed to load Pairadigm object: {e}")
        
##############################
# Other functions
##############################

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

def load_pairadigm(filepath: str) -> Pairadigm:
    """
    Load a Pairadigm object from a pickle file.
    
    This is a standalone function that can be used to load saved Pairadigm objects
    without needing to access the class method.
    
    Parameters
    ----------
    filepath : str
        Path to the saved Pairadigm object file.
        
    Returns
    -------
    Pairadigm
        The loaded Pairadigm object.
        
    Examples
    --------
    >>> from pairadigm import load_pairadigm
    >>> pairadigm_obj = load_pairadigm('my_analysis.pkl')
    """
    filepath = Path(filepath)
    
    # Try adding .pkl extension if file not found
    if not filepath.exists() and filepath.suffix != '.pkl':
        filepath = filepath.with_suffix('.pkl')
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        
        if not isinstance(obj, Pairadigm):
            raise TypeError("Loaded object is not a Pairadigm instance")
        
        # Recreate the LLM client
        obj.client = LLMClient(model_name=obj.model)
        
        print(f"Pairadigm object loaded successfully from: {filepath}")
        return obj
    except Exception as e:
        raise IOError(f"Failed to load Pairadigm object: {e}")