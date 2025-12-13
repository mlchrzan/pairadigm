# pairadigm.py
# Main class for Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation
# Current version 0.4.1

import pandas as pd
import itertools
import random
import choix
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Union
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
        Model identifier (e.g., 'gemini-2.0-flash-exp', 'gpt-4o', 'claude-sonnet-4', 'llama3.2', 'meta-llama/Llama-3.3-70B-Instruct')
    base_url : str, optional
        Base URL for the LLM service API (default: 'http://localhost:11434' for Ollama)
    provider : str, optional
        Force specific provider ('google', 'openai', 'anthropic', 'ollama', 'huggingface'). 
        If None, infers from model_name
    """
    
    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = 'gemini-2.0-flash-exp',
            base_url: Optional[str] = None,
            provider: Optional[str] = None):

        self.model_name = model_name
        self.provider = provider or self._infer_provider(model_name)
        self.base_url = base_url or self._get_default_base_url()
        self.api_key = api_key or self._get_api_key()
        self.client = self._initialize_client()
    
    def _infer_provider(self, model_name: str) -> str:
        """Infer provider from model name."""
        model_lower = model_name.lower()
        
        # Check for HuggingFace patterns (model names with / or common HF orgs)
        hf_patterns = ['/', 'meta-llama', 'mistralai', 'tiiuae', 'bigscience', 'eleutherai']
        if any(pattern in model_name for pattern in hf_patterns):
            return 'huggingface'
        
        # Check for Ollama-specific patterns (colon indicates local model tag)
        if ':' in model_name:
            return 'ollama'
        
        # Check for known Ollama model names
        ollama_models = ['llama', 'mistral', 'phi', 'qwen', 'gemma', 'deepseek', 'vicuna', 'orca']
        if any(x in model_lower for x in ollama_models):
            return 'ollama'
        
        # Then check for cloud providers
        if 'gemini' in model_lower:
            return 'google'
        elif model_lower.startswith('gpt-') and 'gpt-oss' not in model_lower:
            # Be more specific - only official OpenAI models start with 'gpt-'
            return 'openai'
        elif 'claude' in model_lower:
            return 'anthropic'
        else:
            # Default to ollama for unknown models (likely local)
            return 'ollama'
    
    def _get_default_base_url(self) -> Optional[str]:
        """Get default base URL based on provider."""
        if self.provider == 'ollama':
            return 'http://localhost:11434'
        return None
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment based on provider."""
        # Ollama doesn't require an API key
        if self.provider == 'ollama':
            return None
            
        env_vars = {
            'google': 'GENAI_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'huggingface': 'HUGGINGFACE_API_KEY'
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
            if self.base_url:
                return OpenAI(api_key=self.api_key, base_url=self.base_url)
            return OpenAI(api_key=self.api_key)
        
        elif self.provider == 'anthropic':
            from anthropic import Anthropic
            return Anthropic(api_key=self.api_key)
        
        elif self.provider == 'ollama':
            import ollama
            # Use native Ollama client
            return ollama.Client(host=self.base_url)
        
        elif self.provider == 'huggingface':
            from huggingface_hub import InferenceClient
            return InferenceClient(token=self.api_key)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(
            self,
            prompt: str,
            system_message: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: int = 1000) -> str:
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
        
        elif self.provider == 'ollama':
            return self._generate_ollama(prompt, system_message, temperature, max_tokens)
        
        elif self.provider == 'huggingface':
            return self._generate_huggingface(prompt, system_message, temperature, max_tokens)
    
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
        
        # Newer models use max_completion_tokens instead of max_tokens
        newer_models = ['gpt-4-turbo', 'gpt-5', 'gpt-5.1', 'gpt-4o', 'gpt-5-nano', 'gpt-5-mini']
        uses_completion_tokens = any(model in self.model_name.lower() for model in newer_models)
        
        # Some newer models don't support temperature parameter
        models_no_temp_support = ['gpt-5-nano']
        supports_temp = not any(model in self.model_name.lower() for model in models_no_temp_support)
        
        params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
        }
        
        # Only add temperature if supported
        if supports_temp:
            params["temperature"] = temperature
        
        if uses_completion_tokens:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**params)
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
    
    def _generate_ollama(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        max_tokens: int,
        thinking_mode=True
    ) -> str:
        """Generate using Ollama (OpenAI-compatible API)."""
        # Set thinking_mode to "high" if the model name contains gpt-oss
        if "gpt-oss" in self.model_name.lower():
            thinking_mode = "high"
        
        options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Some models treat this as a boolean (True/False)
            # Others (like gpt-oss) might accept strings "low", "medium", "high"
            'stream': False,
            "think": thinking_mode
        }
        
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            options=options
        )
        
        return response['message']['content']
    
    def _generate_huggingface(
        self,
        prompt: str,
        system_message: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using HuggingFace Hub Inference API."""
        # Format messages for chat completion
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use OpenAI-compatible chat completions endpoint
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract text from response
            return response.choices[0].message.content
                
        except Exception as e:
            raise RuntimeError(f"HuggingFace inference failed: {e}")

    
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
                 llm_annotator_cols: Optional[str] = None,
                 prior_breakdown_cols: Optional[List[str]] = None,
                 cgcot_prompts: Optional[List[str]] = None, 
                 model_name: Optional[Union[str, List[str]]] = 'gemini-2.0-flash-exp', 
                 api_key: Optional[Union[str, List[str]]] = None, 
                 base_url: Optional[Union[str, List[str]]] = None,
                 target_concept: Optional[str] = None,
                 llm_clients: Optional[Union[LLMClient, List[LLMClient]]] = None): 
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
            For pre-annotated data, list of column names containing human annotations
        llm_annotator_cols : List[str], optional
            For pre-annotated data, list of column names containing LLM annotations
        prior_breakdown_cols : List[str], optional
            For pre-annotated data, list of column names containing prior LLM breakdowns of the text
        cgcot_prompts : List[str], optional
            CGCoT prompt templates for breakdowns
        model_name : str or List[str], default='gemini-2.0-flash-exp'
            LLM model(s) to use. Can be a single model name or a list of model names.
        api_key : str or List[str], optional
            API key(s) for LLM service(s). Can be a single key or a list of keys.
        base_url : str or List[str], optional
            Base URL(s) for LLM service(s). Can be a single URL or a list of URLs.
        target_concept : str, optional
            The concept to evaluate (e.g., "objectivity", "political bias")
        llm_clients : LLMClient or List[LLMClient], optional
            Pre-initialized LLMClient(s). If provided, model_name and api_key are ignored.
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
            # Change the name of the item_id_cols to item1 and item2 for consistency
            if item_id_cols[0] != 'item1' or item_id_cols[1] != 'item2':
                data = data.rename(columns={
                    item_id_cols[0]: 'item1',
                    item_id_cols[1]: 'item2'
                })
                item_id_cols = ['item1', 'item2']
                
        # Make sure the necessary columns exist if the data is annotated
        num_llm_annotators = len(llm_annotator_cols) if llm_annotator_cols else 0
        if annotated:
            if (annotator_cols is None and llm_annotator_cols is None) or (len(annotator_cols) + num_llm_annotators) < 1:
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
        
        if prior_breakdown_cols is not None and len(prior_breakdown_cols) > 2:
            raise ValueError("prior_breakdown_cols can contain at most 2 column names for paired data or 1 column for unpaired data")
        
        self.data = data.copy()

        self.item_id_name = item_id_name
        self.text_name = text_name
        
        self.paired = paired
        self.item_id_cols = item_id_cols
        self.item_text_cols = item_text_cols
        
        self.annotated = annotated
        self.annotator_cols = annotator_cols
        
        self.llm_annotator_cols = llm_annotator_cols
        self.llm_annotated = llm_annotator_cols is not None
        # Initialize llm_annotator_cols as empty list if None
        if self.llm_annotator_cols is None:
            self.llm_annotator_cols = []

        self.prior_breakdown_cols = prior_breakdown_cols
        
        self.cgcot_prompts = cgcot_prompts
        self.target_concept = target_concept
        

        # Rename prior_breakdown_cols to breakdown1 and breakdown2 if paired (breakdown1 for unpaired)
        if prior_breakdown_cols is not None:
            if paired:
                if len(prior_breakdown_cols) != 2:
                    raise ValueError("For paired data, prior_breakdown_cols must contain exactly 2 column names")
                if prior_breakdown_cols[0] != 'breakdown1' or prior_breakdown_cols[1] != 'breakdown2':
                    self.data = self.data.rename(columns={
                        prior_breakdown_cols[0]: 'breakdown1',
                        prior_breakdown_cols[1]: 'breakdown2'
                    })
                    self.prior_breakdown_cols = ['breakdown1', 'breakdown2']
            else:
                if len(prior_breakdown_cols) != 1:
                    raise ValueError("For unpaired data, prior_breakdown_cols must contain exactly 1 column name")
                if prior_breakdown_cols[0] != 'breakdown1':
                    self.data = self.data.rename(columns={
                        prior_breakdown_cols[0]: 'breakdown1'
                    })
                    self.prior_breakdown_cols = ['breakdown1']
        
        # Initialize LLM client(s)
        if llm_clients is not None:
            # Use provided client(s)
            if isinstance(llm_clients, LLMClient):
                self.clients = [llm_clients]
            elif isinstance(llm_clients, list):
                self.clients = llm_clients
            else:
                raise TypeError("llm_clients must be LLMClient or List[LLMClient]")
            # Extract model names from clients
            self.model_names = [client.model_name for client in self.clients]
        else:
            # Create client(s) from model_name, base_url, and api_key
            if isinstance(model_name, str):
                model_name = [model_name]
            elif not isinstance(model_name, list):
                raise TypeError("model_name must be str or List[str]")
            
            # Normalize base_url to list
            if base_url is None:
                base_url = [None] * len(model_name)
            elif isinstance(base_url, str):
                base_url = [base_url] * len(model_name)
            elif isinstance(base_url, list):
                if len(base_url) != len(model_name):
                    raise ValueError("If base_url is a list, it must have the same length as model_name")
            else:
                raise TypeError("base_url must be str, list of str, or None")
            
            # Normalize api_key to list
            if api_key is None:
                api_key = [None] * len(model_name)
            elif isinstance(api_key, str):
                api_key = [api_key] * len(model_name)
            elif isinstance(api_key, list):
                if len(api_key) != len(model_name):
                    raise ValueError("If api_key is a list, it must have the same length as model_name")
            else:
                raise TypeError("api_key must be str, list of str, or None")
            
            # Create clients with matched parameters
            self.clients = [
                LLMClient(api_key=key, model_name=model, base_url=url, provider=None)
                for model, key, url in zip(model_name, api_key, base_url)
            ]
            
            self.model_names = model_name
        
        # For backward compatibility, set self.client to the first client
        self.client = self.clients[0]
        
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

    def get_clients_info(self) -> pd.DataFrame:
        """
        Get information about all LLM clients in this instance.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: index, model_name, provider
        """
        info = []
        for idx, client in enumerate(self.clients):
            info.append({
                'index': idx,
                'model_name': client.model_name,
                'provider': client.provider
            })
        return pd.DataFrame(info)

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

    def append_human_annotations(
        self,
        annotations: Union[pd.DataFrame, str],
        annotator_names: Union[str, List[str], None] = None,
        item1_col: str = 'item1',
        item2_col: str = 'item2',
        decision_cols: Optional[Union[str, List[str]]] = None, 
        validate_items: bool = True,
        overwrite: bool = False) -> None:

        """
        Upload human annotations to the Pairadigm object. Can handle single or multiple annotators.
        
        Parameters
        ----------
        annotations : pd.DataFrame or str
            Either a DataFrame containing annotations or a filepath to a CSV/Excel file.
            Must contain columns for item1, item2, and decision(s).
        annotator_names : str, List[str], or None
            Name(s) for annotator(s) (will be used as column name(s) in pairwise_df).
            - If str: Single annotator name for single decision column
            - If List[str]: Multiple annotator names for multiple decision columns
            - If None: Will auto-detect decision columns and use them as annotator names
        item1_col : str, default='item1'
            Column name for first item in the pair
        item2_col : str, default='item2'
            Column name for second item in the pair
        decision_cols : str, List[str], or None, optional
            Column name(s) for annotation decision(s). Should contain values like:
            'Text1', 'Text2', 0, 1, or similar.
            - If str: Single decision column
            - If List[str]: Multiple decision columns
            - If None: Auto-detects columns starting with 'decision' or 'annotator'
        validate_items : bool, default=True
            Whether to validate that annotated items exist in pairwise_df
        overwrite : bool, default=False
            Whether to overwrite existing annotations for annotator(s)
            
        Raises
        ------
        ValueError
            If required columns are missing, items don't match, or annotator already exists
        TypeError
            If annotations is not a DataFrame or valid filepath
            
        Examples
        --------
        >>> # Single annotator from DataFrame
        >>> human_anns = pd.DataFrame({
        ...     'item1': ['id1', 'id2'],
        ...     'item2': ['id2', 'id3'],
        ...     'decision': ['Text1', 'Text2']
        ... })
        >>> pairadigm_obj.append_human_annotations(
        ...     human_anns, 
        ...     annotator_names='annotator1'
        ... )
        
        >>> # Multiple annotators from DataFrame
        >>> multi_anns = pd.DataFrame({
        ...     'item1': ['id1', 'id2'],
        ...     'item2': ['id2', 'id3'],
        ...     'annotator1': ['Text1', 'Text2'],
        ...     'annotator2': ['Text2', 'Text1']
        ... })
        >>> pairadigm_obj.append_human_annotations(
        ...     multi_anns,
        ...     decision_cols=['annotator1', 'annotator2']
        ... )
        
        >>> # Auto-detect annotators
        >>> pairadigm_obj.append_human_annotations(multi_anns)
        
        >>> # From file
        >>> pairadigm_obj.append_human_annotations(
        ...     'annotations.csv',
        ...     annotator_names='annotator2'
        ... )
        """
        # Load annotations if filepath is provided
        if isinstance(annotations, str):
            filepath = Path(annotations)
            if not filepath.exists():
                raise FileNotFoundError(f"Annotation file not found: {filepath}")
            
            if filepath.suffix == '.csv':
                annotations_df = pd.read_csv(filepath)
            elif filepath.suffix in ['.xlsx', '.xls']:
                annotations_df = pd.read_excel(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .csv or .xlsx")
        elif isinstance(annotations, pd.DataFrame):
            annotations_df = annotations.copy()
        else:
            raise TypeError("annotations must be a pandas DataFrame or a filepath string")
        
        # Check if pairwise_df exists
        if self.pairwise_df is None:
            raise ValueError("No pairwise_df found. Generate pairings first using generate_pairings() or generate_pairwise_annotations()")
        
        # Auto-detect decision columns if not provided
        if decision_cols is None:
            # Look for columns that might contain decisions
            candidate_cols = [col for col in annotations_df.columns 
                            if col not in [item1_col, item2_col] and 
                            (col.startswith('decision') or col.startswith('annotator') or 
                             col.startswith('human'))]
            if not candidate_cols:
                raise ValueError("No decision columns found. Please specify decision_cols parameter.")
            decision_cols = candidate_cols
            print(f"Auto-detected decision columns: {decision_cols}")
        
        # Normalize decision_cols to list
        if isinstance(decision_cols, str):
            decision_cols = [decision_cols]
        elif not isinstance(decision_cols, list):
            raise TypeError("decision_cols must be a string, list of strings, or None")
        
        # Handle annotator_names
        if annotator_names is None:
            # Use decision column names as annotator names
            annotator_names = decision_cols
        elif isinstance(annotator_names, str):
            annotator_names = [annotator_names]
        elif not isinstance(annotator_names, list):
            raise TypeError("annotator_names must be a string, list of strings, or None")
        
        # Validate lengths match
        if len(annotator_names) != len(decision_cols):
            raise ValueError(
                f"Number of annotator_names ({len(annotator_names)}) must match "
                f"number of decision_cols ({len(decision_cols)})"
            )
        
        # Validate required columns
        required_cols = [item1_col, item2_col] + decision_cols
        missing_cols = [col for col in required_cols if col not in annotations_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for existing annotators
        if not overwrite:
            existing = [name for name in annotator_names if name in self.pairwise_df.columns]
            if existing:
                raise ValueError(
                    f"Annotator(s) {existing} already exist in pairwise_df. "
                    "Set overwrite=True to replace existing annotations."
                )
        
        # Standardize item column names
        annotations_df = annotations_df.rename(columns={
            item1_col: 'item1',
            item2_col: 'item2'
        })
        
        # Validate items if requested
        if validate_items:
            pairwise_items = set(self.pairwise_df['item1']).union(set(self.pairwise_df['item2']))
            annotation_items = set(annotations_df['item1']).union(set(annotations_df['item2']))
            
            missing_items = annotation_items - pairwise_items
            if missing_items:
                warnings.warn(
                    f"The following items in annotations are not in pairwise_df: {missing_items}. "
                    "These annotations will be ignored."
                )
        
        # Process each annotator
        for annotator_name, decision_col in zip(annotator_names, decision_cols):
            # Create a mapping from (item1, item2) pairs to decisions
            annotation_map = {}
            for _, row in annotations_df.iterrows():
                key = (row['item1'], row['item2'])
                annotation_map[key] = row[decision_col]
                
                # Also add reverse mapping for convenience
                reverse_key = (row['item2'], row['item1'])
                # Convert decision for reverse mapping
                if row[decision_col] == 'Text1' or row[decision_col] == 0:
                    reverse_decision = 'Text2' if row[decision_col] == 'Text1' else 1
                elif row[decision_col] == 'Text2' or row[decision_col] == 1:
                    reverse_decision = 'Text1' if row[decision_col] == 'Text2' else 0
                else:
                    reverse_decision = row[decision_col]  # Keep as is for other values
                annotation_map[reverse_key] = reverse_decision
            
            # Map annotations to pairwise_df
            def get_annotation(row):
                key = (row['item1'], row['item2'])
                return annotation_map.get(key, None)
            
            self.pairwise_df[annotator_name] = self.pairwise_df.apply(get_annotation, axis=1)
            
            # Update annotated status and annotator_cols
            if not self.annotated:
                self.annotated = True
                self.annotator_cols = [annotator_name]
            else:
                if annotator_name not in self.annotator_cols:
                    self.annotator_cols.append(annotator_name)
                elif overwrite:
                    # Already in list, just updated values
                    pass
            
            # Report statistics for this annotator
            non_null_count = self.pairwise_df[annotator_name].notna().sum()
            total_pairs = len(self.pairwise_df)
            coverage = non_null_count / total_pairs * 100
            
            print(f"Successfully uploaded annotations for '{annotator_name}'")
            print(f"  Coverage: {non_null_count}/{total_pairs} pairs ({coverage:.1f}%)")

        if self.item_id_cols is None:
            self.item_id_cols = ['item1', 'item2']
        
        # Report overall statistics
        print(f"\nHuman-annotated status: {self.annotated}")
        print(f"Total annotators: {len(self.annotator_cols)}")

################################
# GENERATE CGCOT BREAKDOWNS AND PAIRWISE ANNOTATIONS
################################

    def _generate_cgcot_breakdown(
            self,
            text,
            client: LLMClient,
            rate_limit_per_minute=None,
            max_tokens: int = 1000,
            temperature: float = 0.0) -> str:
        """
        Generate concept-specific breakdown for a given text using CGCoT prompts.
        Args:
            text (str): The text to analyze.
            client (LLMClient): The LLM client to use for generation.
            rate_limit_per_minute (int, optional): Rate limit for API calls.
            max_tokens (int): Maximum tokens for LLM response.
            temperature (float): Sampling temperature for LLM.
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

            # Ensure the prompt itself isn't empty
            if not full_prompt.strip():
                raise ValueError("ERROR: Empty prompt generated. Ensure the cgcot_prompts do not have empty lines.")
            else:
                try:
                    response = client.generate(
                        prompt=full_prompt,
                        system_message="You are a precise and detail-oriented assistant.",
                        temperature=temperature, 
                        max_tokens=max_tokens
                    )
                except Exception as e:
                    response = f"ERROR: {e}"

            prev_answers.append(response)
            breakdown.append(f"Prompt {i+1} response: {response}")
            
            if i < len(self.cgcot_prompts) - 1:
                time.sleep(sleep_time)  # Wait to avoid rate limit

        return "\n".join(breakdown)
    
    def generate_breakdowns(
        self,
        max_workers=8, 
        rate_limit_per_minute=None,
        update_dataframe=True,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        client_indices: Optional[Union[int, List[int]]] = None,
        show_progress: bool = True) -> Dict[Union[str, int], str]:
    
        """
        Generate CGCoT breakdowns for all items in the DataFrame.
        
        Parameters
        ----------
        max_workers : int, default=8
            Number of parallel workers
        rate_limit_per_minute : int, optional
            Rate limit for LLM calls
        update_dataframe : bool, default=True
            If True, adds breakdowns to self.data
        max_tokens : int, default=1000
            Maximum tokens for LLM response
        temperature : float, default=0.0
            Sampling temperature for LLM
        client_indices : int or List[int], optional
            Index/indices of client(s) to use. 
            If None, uses all clients. If int, uses single client. If list, uses multiple clients.
        show_progress : bool, default=True
            If True, displays a progress bar during generation
            
        Returns
        -------
        Dict[Union[str, int], str]
            Mapping of item IDs to breakdowns.
            Also updates self.data with new 'CGCoT_Breakdown' column(s) if update_dataframe is True.
        """

        if self.paired:
            raise ValueError("Data is marked as paired. generate_breakdowns() should only be called on unpaired item lists. Use generate_breakdowns_from_paired() instead.")

        # Determine which clients to use
        if client_indices is None:
            # Use all clients
            clients_to_use = list(enumerate(self.clients))
        elif isinstance(client_indices, int):
            # Use single client
            if client_indices >= len(self.clients):
                raise ValueError(f"client_indices {client_indices} out of range. Only {len(self.clients)} client(s) available.")
            clients_to_use = [(client_indices, self.clients[client_indices])]
        elif isinstance(client_indices, list):
            # Use specified clients
            clients_to_use = []
            for idx in client_indices:
                if idx >= len(self.clients):
                    raise ValueError(f"client_indices {idx} out of range. Only {len(self.clients)} client(s) available.")
                clients_to_use.append((idx, self.clients[idx]))
        else:
            raise TypeError("client_indices must be None, int, or List[int]")

        # Generate breakdowns for each client
        all_results = {}
        total_items = len(self.data)
        
        for client_idx, client in clients_to_use:
            model_name = self.model_names[client_idx]
            print(f"\n{'='*70}")
            print(f"Generating breakdowns for {total_items} items using: {model_name}")
            print(f"{'='*70}")
            
            results = {}
            completed = 0
            failed = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        self._generate_cgcot_breakdown, 
                        row[self.text_name],
                        client,
                        rate_limit_per_minute,
                        max_tokens,
                        temperature): row[self.item_id_name]
                    for _, row in self.data.iterrows()
                }
                
                # Process completions with progress tracking
                for future in as_completed(futures):
                    uuid = futures[future]
                    try:
                        results[uuid] = future.result()
                        completed += 1
                    except Exception as e:
                        results[uuid] = f"ERROR: {e}"
                        failed += 1
                    
                    # Show progress
                    if show_progress:
                        progress_pct = ((completed + failed) / total_items) * 100
                        status_bar = f"[{completed + failed}/{total_items}] {progress_pct:.1f}% complete"
                        
                        if failed > 0:
                            status_bar += f" ({completed} success, {failed} failed)"
                        
                        print(f"\r{status_bar}", end='', flush=True)
            
            # Final newline after progress bar
            if show_progress:
                print()  # Newline after progress
            
            # Print summary for this client
            print(f"Completed: {completed}/{total_items} items")
            if failed > 0:
                print(f"Failed: {failed} items")
            
            if update_dataframe:
                column_name = f'CGCoT_Breakdown_{model_name}' if len(self.clients) > 1 else 'CGCoT_Breakdown'
                self.data[column_name] = self.data[self.item_id_name].map(results)
            
            all_results[client_idx] = results

        if update_dataframe:
            print(f"\nBreakdowns added to [object name].data with column name(s): " +
                ", ".join(
                    [f'CGCoT_Breakdown_{self.model_names[idx]}' if len(self.clients) > 1 else 'CGCoT_Breakdown' 
                    for idx, _ in clients_to_use]
                ))
            return None
        
        # Return results for single client if only one was used, otherwise return all
        if len(clients_to_use) == 1:
            return all_results[clients_to_use[0][0]]
        return all_results

    # def generate_breakdowns(
    #         self,
    #         max_workers=8, 
    #         rate_limit_per_minute=None,
    #         update_dataframe=True,
    #         max_tokens: int = 1000,
    #         temperature: float = 0.0,
    #         client_indices: Optional[Union[int, List[int]]] = None) -> Dict[Union[str, int], str]:
        
    #     """
    #     Generate CGCoT breakdowns for all items in the DataFrame.
        
    #     Parameters
    #     ----------
    #     max_workers : int, default=8
    #         Number of parallel workers
    #     rate_limit_per_minute : int, optional
    #         Rate limit for LLM calls
    #     update_dataframe : bool, default=True
    #         If True, adds breakdowns to self.data
    #     max_tokens : int, default=1000
    #         Maximum tokens for LLM response
    #     temperature : float, default=0.0
    #         Sampling temperature for LLM
    #     client_indices : int or List[int], optional
    #         Index/indices of client(s) to use. 
    #         If None, uses all clients. If int, uses single client. If list, uses multiple clients.
            
    #     Returns
    #     -------
    #     Dict[Union[str, int], str]
    #         Mapping of item IDs to breakdowns.
    #         Also updates self.data with new 'CGCoT_Breakdown' column(s) if update_dataframe is True.
    #     """

    #     if self.paired:
    #         raise ValueError("Data is marked as paired. generate_breakdowns() should only be called on unpaired item lists. Use generate_breakdowns_from_paired() instead.")

    #     # Determine which clients to use
    #     if client_indices is None:
    #         # Use all clients
    #         clients_to_use = list(enumerate(self.clients))
    #     elif isinstance(client_indices, int):
    #         # Use single client
    #         if client_indices >= len(self.clients):
    #             raise ValueError(f"client_indices {client_indices} out of range. Only {len(self.clients)} client(s) available.")
    #         clients_to_use = [(client_indices, self.clients[client_indices])]
    #     elif isinstance(client_indices, list):
    #         # Use specified clients
    #         clients_to_use = []
    #         for idx in client_indices:
    #             if idx >= len(self.clients):
    #                 raise ValueError(f"client_indices {idx} out of range. Only {len(self.clients)} client(s) available.")
    #             clients_to_use.append((idx, self.clients[idx]))
    #     else:
    #         raise TypeError("client_indices must be None, int, or List[int]")

    #     # Generate breakdowns for each client
    #     all_results = {}
    #     for client_idx, client in clients_to_use:
    #         results = {}
    #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #             futures = {
    #                 executor.submit(
    #                     self._generate_cgcot_breakdown, 
    #                     row[self.text_name],
    #                     client,
    #                     rate_limit_per_minute,
    #                     max_tokens,
    #                     temperature): row[self.item_id_name]
    #                 for _, row in self.data.iterrows()
    #             }
    #             for future in as_completed(futures):
    #                 uuid = futures[future]
    #                 try:
    #                     results[uuid] = future.result()
    #                 except Exception as e:
    #                     results[uuid] = f"ERROR: {e}"

    #         if update_dataframe:
    #             column_name = f'CGCoT_Breakdown_{self.model_names[client_idx]}' if len(self.clients) > 1 else 'CGCoT_Breakdown'
    #             self.data[column_name] = self.data[self.item_id_name].map(results)
            
    #         all_results[client_idx] = results
    
    #     if update_dataframe:
    #         print(f"Breakdowns added to self.data with column name(s): " +
    #               ", ".join(
    #                   [f'CGCoT_Breakdown_{self.model_names[idx]}' if len(self.clients) > 1 else 'CGCoT_Breakdown' 
    #                    for idx, _ in clients_to_use]
    #               ))
    #         return None
        
    #     # Return results for single client if only one was used, otherwise return all
    #     if len(clients_to_use) == 1:
    #         return all_results[clients_to_use[0][0]]
    #     return all_results
    
    def generate_breakdowns_from_paired(
            self, 
            max_workers: int = 8,
            rate_limit_per_minute: Optional[int] = None,
            update_pairwise_df: bool = True,
            max_tokens: int = 1000,
            tempature: float = 0.0,
            client_indices: Optional[Union[int, List[int]]] = None) -> Dict[Union[str, int], str]:
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
        max_tokens : int, default=1000
            Maximum tokens for LLM response
        tempature : float, default=0.0
            Sampling temperature for LLM
        client_indices : int or List[int], optional
            Index/indices of client(s) to use. 
            If None, uses all clients. If int, uses single client. If list, uses multiple clients.
            
        Returns
        -------
        Dict[Union[str, int], str]
            Mapping of item IDs to breakdowns
        """
        if not self.paired:
            raise ValueError("Data is not marked as paired. generate_breakdowns_from_paired() should only be called on paired item lists.")

        # Determine which clients to use
        if client_indices is None:
            # Use all clients
            clients_to_use = list(enumerate(self.clients))
        elif isinstance(client_indices, int):
            # Use single client
            if client_indices >= len(self.clients):
                raise ValueError(f"client_indices {client_indices} out of range. Only {len(self.clients)} client(s) available.")
            clients_to_use = [(client_indices, self.clients[client_indices])]
        elif isinstance(client_indices, list):
            # Use specified clients
            clients_to_use = []
            for idx in client_indices:
                if idx >= len(self.clients):
                    raise ValueError(f"client_indices {idx} out of range. Only {len(self.clients)} client(s) available.")
                clients_to_use.append((idx, self.clients[idx]))
        else:
            raise TypeError("client_indices must be None, int, or List[int]")

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
        
        # Generate breakdowns for each client
        all_results = {}
        for client_idx, client in clients_to_use:
            results = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._generate_cgcot_breakdown, 
                        text_mapping[item_id],
                        client,
                        rate_limit_per_minute,
                        max_tokens,
                        tempature): item_id
                    for item_id in items_df[self.item_id_name]
                }
                
                for future in as_completed(futures):
                    item_id = futures[future]
                    try:
                        results[item_id] = future.result()
                    except Exception as e:
                        results[item_id] = f"ERROR: {e}"
            
            # Update pairwise_df with breakdown columns if requested
            if update_pairwise_df:
                breakdown1_col = f'breakdown1_{self.model_names[client_idx]}' if len(self.clients) > 1 else 'breakdown1'
                breakdown2_col = f'breakdown2_{self.model_names[client_idx]}' if len(self.clients) > 1 else 'breakdown2'
                
                if self.pairwise_df is not None:
                    self.pairwise_df[breakdown1_col] = self.pairwise_df[item1_id_col].map(results)
                    self.pairwise_df[breakdown2_col] = self.pairwise_df[item2_id_col].map(results)
                else:
                    # Create pairwise_df from self.data with breakdown columns
                    self.pairwise_df = source_df.copy()
                    self.pairwise_df[breakdown1_col] = self.pairwise_df[item1_id_col].map(results)
                    self.pairwise_df[breakdown2_col] = self.pairwise_df[item2_id_col].map(results)
            
            all_results[client_idx] = results

        # Return results for single client if only one was used, otherwise return all
        if len(clients_to_use) == 1:
            return all_results[clients_to_use[0][0]]
        return all_results

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

        # Add check for paired data
        if self.paired:
            raise ValueError(
                "Data is already in paired format. Cannot generate pairings from paired data. "
                "Use generate_pairwise_annotations() directly, or if you need to re-pair items, "
                "create a new Pairadigm object with paired=False."
            )

        # Pair items
        uuid_pairings = self.pair_items(
            self.data[self.item_id_name].tolist(),
            num_pairs_per_item,
            random_seed)
        
        # Set item_id_cols since data is now paired
        self.item_id_cols = ['item1', 'item2']
        
        # Map breakdowns to pairings if present
        if breakdowns:

            # Find any CGCoT breakdown columns. These may be named either
            # 'CGCoT_Breakdown' (single-client) or 'CGCoT_Breakdown_<model_name>'
            breakdown_cols = [c for c in self.data.columns if c.startswith('CGCoT_Breakdown')]
            if len(breakdown_cols) == 0:
                raise ValueError("No 'CGCoT_Breakdown' columns found in DataFrame. Generate them using generate_breakdowns() first.")

            # Create mapping from UUID to each breakdown column and attach to pairings
            for col in breakdown_cols:
                uuid_to_desc = dict(zip(self.data[self.item_id_name], self.data[col]))

                # If column is the generic name, create 'breakdown1'/'breakdown2'
                if col == 'CGCoT_Breakdown':
                    uuid_pairings['breakdown1'] = uuid_pairings['item1'].map(uuid_to_desc)
                    uuid_pairings['breakdown2'] = uuid_pairings['item2'].map(uuid_to_desc)
                else:
                    # Column name format is expected to be 'CGCoT_Breakdown_<model_name>'
                    # Keep the model_name suffix exactly as in the column to preserve downstream naming
                    suffix = col[len('CGCoT_Breakdown_'):]
                    uuid_pairings[f'breakdown1_{suffix}'] = uuid_pairings['item1'].map(uuid_to_desc)
                    uuid_pairings[f'breakdown2_{suffix}'] = uuid_pairings['item2'].map(uuid_to_desc)

            if update_classObject:
                self.pairwise_df = uuid_pairings
                # self.paired = True
                print(f"Pairwise DataFrame with breakdowns created and stored in self.pairwise_df")
        else:
            if update_classObject:
                self.pairwise_df = uuid_pairings
                # self.paired = True
                print(f"Pairwise DataFrame created and stored in self.pairwise_df")
        
        return uuid_pairings

    # Helper function to compare two breakdowns
    @staticmethod
    def pairwise_compare(
        text1_breakdown: str, 
        text2_breakdown: str, 
        target_concept: str,
        client: LLMClient, 
        max_tokens: int = 1000,
        temperature: float = 0.0,
        allow_ties: bool = False):
        """
        Compare two CGCoT breakdowns to decide which expresses greater level of target concept.
        Args:
            text1_breakdown (str): Breakdown for first text
            text2_breakdown (str): Breakdown for second text
            target_concept (str): Concept name for comparison (e.g., "aversion to Republicans")
            client (LLMClient): LLM client to use for comparison
            max_tokens (int, optional): Max tokens for LLM response. Defaults to 1000.
            temperature (float, optional): Sampling temperature for LLM. Defaults to 0.0
            allow_ties (bool, optional): If True, allows ties in comparison. Defaults to False.
        Returns:
            str: "Text1" or "Text2" (or 'Tie' if allowed)
            str: Full LLM response for transparency
        """

        if not allow_ties:
            comparison_prompt = f""" 
            Description 1: {text1_breakdown}
            Description 2: {text2_breakdown}
            Based on these two Descriptions, which Description expresses greater {target_concept}: Description 1 or Description 2? You must choose one of the descriptions.

            Format your response as follows:
            FINAL ANSWER: <Your choice of "Description 1" or "Description 2">
            JUSTIFICATION: <Your CONCISE reasoning for the choice>
            """
        else:
            comparison_prompt = f""" 
            Description 1: {text1_breakdown}
            Description 2: {text2_breakdown}
            Based on these two Descriptions, which Description expresses greater {target_concept}: Description 1, Description 2, or are they tied? You must choose one of the descriptions or indicate a tie.

            Format your response as follows:
            FINAL ANSWER: <Your choice of "Description 1", "Description 2", or "Tie">
            JUSTIFICATION: <Your CONCISE reasoning for the choice>
            """

        response = client.generate(
            prompt=comparison_prompt,
            system_message="You are a precise and detail-oriented assistant.",
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Use regex to extract the final answer
        answer_pattern = r"FINAL ANSWER:\s*(Description 1|Description 2|Tie)"
        match = re.search(answer_pattern, response, re.IGNORECASE)
        
        if match:
            extracted_answer = match.group(1)
            if extracted_answer.lower() == "description 1":
                final_answer = "Text1"
            elif extracted_answer.lower() == "description 2":
                final_answer = "Text2"
            elif extracted_answer.lower() == "tie" and allow_ties:
                final_answer = "Tie"
            else:
                final_answer = f"ERROR from pairadigm (not model): Regex match found but final answer did not include Text1 or Text2 (or, if allowed, Tie). Model response: {response}"
        else:
            # If regex fails, fallback to a direct extraction prompt
            extraction_prompt = f"""
            In the following text, which Description is described to be expressing greater {target_concept}: Description 1 or Description 2? ONLY REPLY WITH "Description 1" or "Description 2". Text: {response}
            """

            extracted_answer = client.generate(
                prompt=extraction_prompt,
                system_message="You are a precise and detail-oriented assistant.",
                temperature=temperature,
                max_tokens=max_tokens
            )

            extracted_answer = extracted_answer.strip()

            if extracted_answer == "Description 1":
                final_answer = "Text1"
            elif extracted_answer == "Description 2":
                final_answer = "Text2"
            elif extracted_answer == "Tie" and allow_ties:
                final_answer = "Tie"
            else:
                final_answer = f"ERROR from pairadigm (not model): Regex match NOT found even after recalling the model with an extraction prompt. Model response: {response}"

        return final_answer, response

    def generate_pairwise_annotations(
        self,
        max_workers=8,
        update_classObject=True,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        allow_ties=False,
        client_indices: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        """
        Run pairwise comparisons on all pairs in the pairwise_df DataFrame in parallel.
        
        Args:
            max_workers (int): Number of threads to use
            update_classObject (bool, optional): If True, updates self.pairwise_df with results. Defaults to True.
            max_tokens (int, optional): Max tokens for LLM response. Defaults to 1000.
            temperature (float, optional): Sampling temperature for LLM. Defaults to 0.
            allow_ties (bool, optional): If True, allows ties in comparisons. Defaults to False.
            client_indices (int or List[int], optional): Index/indices of client(s) to use. 
                If None, uses all clients. If int, uses single client. If list, uses multiple clients.
        Returns:
            pd.DataFrame: Original dataframe with added 'decision' and 'justification' columns
                (or multiple columns if using multiple clients)
        """

        if self.pairwise_df is None:
            raise ValueError("No pairwise_df found in the object. Generate pairings with breakdowns first using generate_pairings(breakdowns=True).")
        
        # Determine which clients to use
        if client_indices is None:
            # Use all clients
            clients_to_use = list(enumerate(self.clients))
        elif isinstance(client_indices, int):
            # Use single client
            if client_indices >= len(self.clients):
                raise ValueError(f"client_indices {client_indices} out of range. Only {len(self.clients)} client(s) available.")
            clients_to_use = [(client_indices, self.clients[client_indices])]
        elif isinstance(client_indices, list):
            # Use specified clients
            clients_to_use = []
            for idx in client_indices:
                if idx >= len(self.clients):
                    raise ValueError(f"client_indices {idx} out of range. Only {len(self.clients)} client(s) available.")
                clients_to_use.append((idx, self.clients[idx]))
        else:
            raise TypeError("client_indices must be None, int, or List[int]")

        result_df = self.pairwise_df.copy()
        
        # For each client, generate annotations
        for client_idx, client in clients_to_use:
            # Determine breakdown column names
            if len(self.clients) > 1:
                breakdown1_col = f'breakdown1_{self.model_names[client_idx]}'
                breakdown2_col = f'breakdown2_{self.model_names[client_idx]}'
                decision_col = f'decision_{self.model_names[client_idx]}'
                justification_col = f'justification_{self.model_names[client_idx]}'
            else:
                breakdown1_col = 'breakdown1'
                breakdown2_col = 'breakdown2'
                decision_col = 'decision'
                justification_col = 'justification'
            
            # Check if breakdown columns exist
            if breakdown1_col not in result_df.columns or breakdown2_col not in result_df.columns:
                raise ValueError(f"Breakdown columns '{breakdown1_col}' and '{breakdown2_col}' not found. Generate breakdowns for paired items using generate_breakdowns_from_paired(client_index={client_idx}) first.")
            
            results = [None] * len(result_df)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.pairwise_compare, 
                        row[breakdown1_col], 
                        row[breakdown2_col], 
                        self.target_concept,
                        client, 
                        max_tokens,
                        temperature,
                        allow_ties
                        
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
                        model_name = self.model_names[client_idx] if len(self.clients) > 1 else "default"
                        print(f"[{model_name}] Completed {i + 1}/{len(result_df)} comparisons")
            
            result_df[decision_col] = [r[0] for r in results]
            result_df[justification_col] = [r[1] for r in results]
            self.llm_annotator_cols.append(decision_col)
        
        # Update instance if requested
        if update_classObject:
            self.pairwise_df = result_df
            self.llm_annotated = True
        
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
    def prep_for_alt_test(
        self,
        llm_decision_col: Optional[str] = None
    ) -> Tuple[Dict[Union[int, str], Any], Dict[Union[int, str], Dict[Union[int, str], Any]]]:
        """
        Prepare annotations from class data for alt_test function.
        
        Args
        ----------
        llm_decision_col : str, optional
            Specific LLM decision column to use. If None and only one LLM was used,
            uses 'decision'. If multiple LLMs were used, this parameter is required.
            Format: 'decision' or 'decision_<model_name>'
        
        Returns
        -------
        Tuple[Dict[Union[int, str], Any], Dict[Union[int, str], Dict[Union[int, str], Any]]]
            (llm_annotations, humans_annotations)
        
        Raises
        ------
        ValueError
            If multiple LLM annotators exist but llm_decision_col is not specified
        """
        if not self.annotated:
            raise ValueError("Data must have human annotations to run the alt_test")
        
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison data found. Run generate_pairwise_annotations() first.")
        
        # Determine which decision column to use
        if llm_decision_col is None:
            # Check if there are multiple LLM decision columns
            decision_cols = [col for col in self.pairwise_df.columns if col.startswith('decision')]
            
            if len(decision_cols) == 0:
                raise ValueError("No 'decision' columns found. Run generate_pairwise_annotations() first.")
            elif len(decision_cols) == 1:
                llm_decision_col = decision_cols[0]
            else:
                # Multiple LLM annotators found
                available_models = [col.replace('decision_', '') for col in decision_cols if col != 'decision']
                if 'decision' in decision_cols:
                    available_models.insert(0, 'decision (default)')
                raise ValueError(
                    f"Multiple LLM decision columns found: {decision_cols}. "
                    f"Please specify which one to use via llm_decision_col parameter. "
                    f"Available models: {available_models}"
                )
        else:
            # Validate the specified column exists
            if llm_decision_col not in self.pairwise_df.columns:
                raise ValueError(
                    f"Column '{llm_decision_col}' not found in pairwise_df. "
                    f"Available columns: {list(self.pairwise_df.columns)}"
                )
        
        # Use class attributes for column names
        item1_id_col, item2_id_col = self.item_id_cols
        
        llm_annotations = {}
        humans_annotations = {col: {} for col in self.annotator_cols}
        
        for _, row in self.pairwise_df.iterrows():
            item1_id = row[item1_id_col]
            item2_id = row[item2_id_col]
            decision = row[llm_decision_col]
            
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
                if human_decision == 'Text1' or human_decision == 0:
                    humans_annotations[col][item1_id] = humans_annotations[col].get(item1_id, 0) + 1
                    humans_annotations[col][item2_id] = humans_annotations[col].get(item2_id, 0)
                elif human_decision == 'Text2' or human_decision == 1:
                    humans_annotations[col][item2_id] = humans_annotations[col].get(item2_id, 0) + 1
                    humans_annotations[col][item1_id] = humans_annotations[col].get(item1_id, 0)
                else:
                    continue
        
        print(f"Using LLM decision column: {llm_decision_col}")
        return llm_annotations, humans_annotations

    def alt_test(
        self,
        llm_annotations: Optional[Dict[Union[int, str], Any]] = None,
        humans_annotations: Optional[Dict[Union[int, str], Dict[Union[int, str], Any]]] = None,
        scoring_function: Union[str, Callable] = 'accuracy',
        epsilon: float = 0.1,
        q_fdr: float = 0.05,
        min_humans_per_instance: int = 2,
        min_instances_per_human: int = 30,
        llm_decision_col: Optional[str] = None,
        test_all_llms: bool = False) -> Union[Tuple[float, float], Dict[str, Tuple[float, float]]]:
        """
        Perform the alternative annotator test to compare LLM annotations against human annotations.
        
        Args:
            llm_annotations (Optional[Dict[Union[int, str], Any]]): Mapping of instance IDs to LLM annotations. 
                If None, will be generated using prep_for_alt_test().
            humans_annotations (Optional[Dict[Union[int, str], Dict[Union[int, str], Any]]]): Mapping of human IDs to their annotations. 
                If None, will be generated using prep_for_alt_test().
            scoring_function (Union[str, Callable], optional): Scoring function to use ('accuracy', 'neg_rmse', or custom). Defaults to 'accuracy'.
            epsilon (float, optional): Adjustment value for t-test. Defaults to 0.1.
            q_fdr (float, optional): FDR level for BY procedure. Defaults to 0.05.
            min_humans_per_instance (int, optional): Minimum annotators per instance to include. Defaults to 2.
            min_instances_per_human (int, optional): Minimum instances per annotator to include. Defaults to 30.
            llm_decision_col (str, optional): Specific LLM decision column to test. Format: 'decision' or 'decision_<model_name>'.
                If None and test_all_llms=False, will use single 'decision' column or raise error if multiple exist.
            test_all_llms (bool, optional): If True, tests all LLM decision columns and returns dict of results. Defaults to False.
        
        Returns:
            Union[Tuple[float, float], Dict[str, Tuple[float, float]]]: 
                If test_all_llms=False: (winning_rate, advantage_prob) for single LLM
                If test_all_llms=True: Dict mapping model names to (winning_rate, advantage_prob)
        """
        
        # Validate that we have human annotations
        if not self.annotated:
            raise ValueError("Data must have human annotations to run the alt_test")
        
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison data found. Run generate_pairwise_annotations() first.")
        
        # Find all available decision columns
        decision_cols = [col for col in self.pairwise_df.columns if col.startswith('decision')]
        
        if len(decision_cols) == 0:
            raise ValueError("No 'decision' columns found. Run generate_pairwise_annotations() first.")
        
        # Determine which columns to test
        if test_all_llms:
            # Test all available LLM decision columns
            cols_to_test = decision_cols
            print(f"Testing all {len(cols_to_test)} LLM decision columns: {cols_to_test}")
        else:
            # Test single column
            if llm_decision_col is None:
                if len(decision_cols) == 1:
                    cols_to_test = [decision_cols[0]]
                else:
                    available_models = [col.replace('decision_', '') for col in decision_cols if col != 'decision']
                    if 'decision' in decision_cols:
                        available_models.insert(0, 'decision (default)')
                    raise ValueError(
                        f"Multiple LLM decision columns found: {decision_cols}. "
                        f"Please specify which one to use via llm_decision_col parameter, "
                        f"or set test_all_llms=True to test all. "
                        f"Available models: {available_models}"
                    )
            else:
                if llm_decision_col not in self.pairwise_df.columns:
                    raise ValueError(
                        f"Column '{llm_decision_col}' not found in pairwise_df. "
                        f"Available columns: {list(self.pairwise_df.columns)}"
                    )
                cols_to_test = [llm_decision_col]
        
        # Prepare alignment scoring function
        if isinstance(scoring_function, str):
            if scoring_function == 'accuracy':
                scoring_function = self.accuracy
            elif scoring_function == 'neg_rmse':
                scoring_function = self.neg_rmse
            else:
                raise ValueError("Unknown scoring function")
        
        # Get human annotations once (shared across all LLM tests)
        if humans_annotations is None:
            _, humans_annotations = self.prep_for_alt_test(llm_decision_col=cols_to_test[0])
        
        # Store results for each LLM
        results = {}
        
        # Test each LLM decision column
        for col in cols_to_test:
            # Get LLM annotations for this column
            if llm_annotations is None or test_all_llms:
                llm_anns, _ = self.prep_for_alt_test(llm_decision_col=col)
            else:
                llm_anns = llm_annotations
            
            # Prepare sets - i_set has humans as keys, h_set has instances as keys
            i_set, h_set = {}, {}
            for h, anns in humans_annotations.items():
                i_set[h] = list(anns.keys())
                for i, ann in anns.items():
                    if i not in h_set:
                        h_set[i] = []
                    h_set[i].append(h)

            # Remove instances with less than min_humans_per_instance
            instances_to_keep = {i for i in h_set if len(h_set[i]) >= min_humans_per_instance and i in llm_anns}
            if len(instances_to_keep) < len(h_set):
                print(f"[{col}] Dropped {len(h_set) - len(instances_to_keep)} instances with less than {min_humans_per_instance} annotators.")
            i_set = {h: [i for i in i_set[h] if i in instances_to_keep] for h in i_set}
            h_set = {i: h_set[i] for i in h_set if i in instances_to_keep}

            p_values, advantage_probs, humans = [], [], []
            for excluded_h in humans_annotations:
                llm_indicators = []
                excluded_indicators = []
                instances = [i for i in i_set[excluded_h] if i in llm_anns]
                if len(instances) < min_instances_per_human:
                    print(f"[{col}] Skipping annotator {excluded_h} with only {len(instances)} instances < {min_instances_per_human}.")
                    continue

                for i in instances:
                    human_ann = humans_annotations[excluded_h][i]
                    llm_ann = llm_anns[i]
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
            winning_rate = len(rejected_indices) / len(humans) if len(humans) > 0 else 0.0
            
            # Store results
            model_name = col.replace('decision_', '') if col != 'decision' else 'default'
            results[model_name] = (winning_rate, advantage_prob)
            
            # Print results for this LLM
            print(f"\n{'='*70}")
            print(f"ALT-TEST RESULTS - {model_name.upper()}")
            print(f"{'='*70}")
            print(f"Winning Rate (ω): {winning_rate:.3f}")
            print(f"Advantage Probability: {advantage_prob:.3f}")
            print(f"Tested against {len(humans)} human annotators")
            print(f"{'='*70}\n")
        
        # Return single tuple if testing one LLM, dict if testing multiple
        if len(results) == 1:
            return list(results.values())[0]
        else:
            return results

    def dawid_skene_alt_test(
        self,
        decision_col: Optional[str] = None,
        num_classes: int = 2,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_seed: Optional[int] = None,
        alpha: float = 0.05,
        use_by_correction: bool = True,
        test_all_llms: bool = False) -> Union[Dict, Dict[str, Dict]]:
        """
        Perform Dawid-Skene validation comparing LLM annotations against human annotators.
        
        This method implements the Dawid-Skene model to estimate annotator reliability,
        then computes weighted agreement margins and performs statistical testing with
        optional Benjamini-Yekutieli FDR correction.
        
        Parameters
        ----------
        decision_col : str, optional
            Column name containing LLM decisions. Format: 'decision' or 'decision_<model_name>'.
            If None and test_all_llms=False, will use single 'decision' column or raise error if multiple exist.
        num_classes : int, default=2
            Number of classes (e.g., 2 for binary Text1/Text2)
        max_iter : int, default=100
            Maximum iterations for Dawid-Skene EM algorithm
        tol : float, default=1e-6
            Convergence tolerance for Dawid-Skene
        alpha : float, default=0.05
            Significance level for hypothesis testing
        use_by_correction : bool, default=True
            Whether to apply Benjamini-Yekutieli FDR correction
        test_all_llms : bool, default=False
            If True, tests all LLM decision columns and returns dict of results
            
        Returns
        -------
        Union[Dict, Dict[str, Dict]]
            If test_all_llms=False: Single dict with results for one LLM
            If test_all_llms=True: Dict mapping model names to their result dicts
            
            Each result dict contains:
            - 'annotator_weights': Reliability weights from Dawid-Skene
            - 'label_probs': Estimated true label probabilities
            - 'margins': List of (annotator_id, margin) tuples
            - 'advantage_probabilities': Per-annotator test results
            - 'winning_rate': Proportion of annotators where LLM significantly outperforms
            - 'convergence_iteration': Iteration where Dawid-Skene converged
            
        Raises
        ------
        ValueError
            If data is not annotated or required columns are missing
        """
        
        if not self.annotated:
            raise ValueError("Data must have human annotations to run the Dawid-Skene version of the alt_test.")
        
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison data found")
        
        # Find all available decision columns
        decision_cols = [col for col in self.pairwise_df.columns if col.startswith('decision')]
        
        if len(decision_cols) == 0:
            raise ValueError("No 'decision' columns found. Run generate_pairwise_annotations() first.")
        
        # Determine which columns to test
        if test_all_llms:
            # Test all available LLM decision columns
            cols_to_test = decision_cols
            print(f"Testing all {len(cols_to_test)} LLM decision columns: {cols_to_test}")
        else:
            # Test single column
            if decision_col is None:
                if len(decision_cols) == 1:
                    cols_to_test = [decision_cols[0]]
                else:
                    available_models = [col.replace('decision_', '') for col in decision_cols if col != 'decision']
                    if 'decision' in decision_cols:
                        available_models.insert(0, 'decision (default)')
                    raise ValueError(
                        f"Multiple LLM decision columns found: {decision_cols}. "
                        f"Please specify which one to use via decision_col parameter, "
                        f"or set test_all_llms=True to test all. "
                        f"Available models: {available_models}"
                    )
            else:
                if decision_col not in self.pairwise_df.columns:
                    raise ValueError(
                        f"Column '{decision_col}' not found in pairwise_df. "
                        f"Available columns: {list(self.pairwise_df.columns)}"
                    )
                cols_to_test = [decision_col]
        
        # Store results for each LLM
        all_results = {}
        
        # Test each LLM decision column
        for col in cols_to_test:
            # Step 1: Prepare annotation matrix
            instances = self.pairwise_df.index.tolist()
            num_instances = len(instances)
            num_annotators = len(self.annotator_cols)
            
            # Create annotation matrix: rows=instances, cols=annotators
            annotator_labels = np.zeros((num_instances, num_annotators), dtype=int)
            
            for j, annotator_col in enumerate(self.annotator_cols):
                for i in range(num_instances):
                    val = self.pairwise_df.iloc[i][annotator_col]
                    if pd.isna(val):
                        annotator_labels[i, j] = 0  # Default
                    elif val == 'Text1' or val == 0:
                        annotator_labels[i, j] = 0
                    elif val == 'Text2' or val == 1:
                        annotator_labels[i, j] = 1
                    else:
                        annotator_labels[i, j] = 0  # Default for invalid
            
            # Create LLM labels array
            llm_labels = np.zeros(num_instances, dtype=int)
            for i in range(num_instances):
                val = self.pairwise_df.iloc[i][col]
                if val == 'Text1' or val == 0:
                    llm_labels[i] = 0
                elif val == 'Text2' or val == 1:
                    llm_labels[i] = 1
            
            # Step 2: Dawid-Skene Model
            def dawid_skene_em(labels, num_classes, max_iter, tol, random_seed):
                n_instances, n_annotators = labels.shape
                
                np.random.seed(random_seed)

                # Initialize with majority vote
                label_probs = np.zeros((n_instances, num_classes))
                for i in range(n_instances):
                    majority = np.bincount(labels[i], minlength=num_classes)
                    label_probs[i] = majority / majority.sum()
                
                # Initialize confusion matrices with noise
                confusion_matrices = np.full((n_annotators, num_classes, num_classes), 
                                            1 / num_classes)
                confusion_matrices += np.random.randn(n_annotators, num_classes, num_classes) * 0.01
                confusion_matrices = np.abs(confusion_matrices)
                
                # Normalize rows
                for j in range(n_annotators):
                    for c in range(num_classes):
                        confusion_matrices[j, c] /= confusion_matrices[j, c].sum()
                
                prev_label_probs = label_probs.copy()
                converged_iter = max_iter
                
                for iteration in range(max_iter):
                    # E-step
                    for i in range(n_instances):
                        for c in range(num_classes):
                            prob = 1.0
                            for j in range(n_annotators):
                                observed = labels[i, j]
                                prob *= confusion_matrices[j, c, observed]
                            label_probs[i, c] = prob
                        
                        # Normalize
                        total = np.sum(label_probs[i])
                        if total > 0:
                            label_probs[i] /= total
                        else:
                            label_probs[i] = 1 / num_classes
                    
                    # M-step
                    for j in range(n_annotators):
                        for c in range(num_classes):
                            for k in range(num_classes):
                                numerator = sum(label_probs[i, c] 
                                            for i in range(n_instances) 
                                            if labels[i, j] == k)
                                denominator = sum(label_probs[i, c] 
                                                for i in range(n_instances))
                                if denominator > 0:
                                    confusion_matrices[j, c, k] = numerator / denominator
                                else:
                                    confusion_matrices[j, c, k] = 1 / num_classes
                    
                    # Check convergence
                    if iteration > 0 and np.linalg.norm(label_probs - prev_label_probs) < tol:
                        converged_iter = iteration
                        break
                    prev_label_probs = label_probs.copy()
                
                # Compute annotator weights
                annotator_weights = np.array([
                    np.mean(np.diag(confusion_matrices[j])) 
                    for j in range(n_annotators)
                ])
                
                return label_probs, annotator_weights, converged_iter
            
            label_probs, annotator_weights, conv_iter = dawid_skene_em(
                annotator_labels, num_classes, max_iter, tol, random_seed
            )
            
            # Step 3: Compute Weighted-ACC and Margins
            margins = []
            
            for j in range(num_annotators):
                other_indices = [k for k in range(num_annotators) if k != j]
                weights = annotator_weights[other_indices]
                
                for i in range(num_instances):
                    # Weighted agreement of LLM with other annotators
                    llm_agreements = [
                        int(llm_labels[i] == annotator_labels[i, k]) 
                        for k in other_indices
                    ]
                    llm_weighted_acc = np.average(llm_agreements, weights=weights)
                    
                    # Weighted agreement of pulled-out annotator with others
                    human_agreements = [
                        int(annotator_labels[i, j] == annotator_labels[i, k]) 
                        for k in other_indices
                    ]
                    human_weighted_acc = np.average(human_agreements, weights=weights)
                    
                    # Margin
                    delta = llm_weighted_acc - human_weighted_acc
                    margins.append((j, delta))
            
            # Step 4: Paired t-test per annotator
            annotator_margins = pd.DataFrame(margins, columns=["annotator", "margin"])
            advantage_probabilities = {}
            winning_count = 0
            
            # Perform t-tests and collect p-values
            p_values = []
            for j in range(num_annotators):
                deltas = annotator_margins[annotator_margins["annotator"] == j]["margin"].values
                t_stat, p_value = ttest_1samp(deltas, popmean=0, alternative='greater')
                p_values.append(p_value)
                
                annotator_name = self.annotator_cols[j]
                advantage_probabilities[annotator_name] = {
                    "mean_margin": np.mean(deltas),
                    "p_value": p_value
                }
            
            # Step 5: Apply correction if requested
            if use_by_correction:
                from statsmodels.stats.multitest import multipletests
                reject, corrected_p_values, _, _ = multipletests(
                    p_values, alpha=alpha, method='fdr_by'
                )
                
                for j, annotator_name in enumerate(advantage_probabilities.keys()):
                    advantage_probabilities[annotator_name]["corrected_p_value"] = corrected_p_values[j]
                    advantage_probabilities[annotator_name]["reject_null"] = reject[j]
                    if reject[j]:
                        winning_count += 1
            else:
                for j, annotator_name in enumerate(advantage_probabilities.keys()):
                    advantage_probabilities[annotator_name]["reject_null"] = p_values[j] < alpha
                    if p_values[j] < alpha:
                        winning_count += 1
            
            # Compute winning rate
            winning_rate = winning_count / num_annotators
            
            # Store results
            model_name = col.replace('decision_', '') if col != 'decision' else 'default'
            all_results[model_name] = {
                'annotator_weights': annotator_weights,
                'label_probs': label_probs,
                'margins': margins,
                'advantage_probabilities': advantage_probabilities,
                'winning_rate': winning_rate,
                'convergence_iteration': conv_iter
            }
            
            # Print results for this LLM
            print("\n" + "="*70)
            print(f"DAWID-SKENE VALIDATION RESULTS - {model_name.upper()}")
            print("="*70)
            print(f"Converged at iteration: {conv_iter}")
            print(f"\nAnnotator Reliability Weights:")
            for j, annotator_col in enumerate(self.annotator_cols):
                print(f"  {annotator_col}: {annotator_weights[j]:.4f}")
            
            print(f"\nAdvantage Probabilities per Annotator:")
            for annotator, stats in advantage_probabilities.items():
                print(f"\n{annotator}:")
                print(f"  Mean margin: {stats['mean_margin']:.4f}")
                print(f"  p-value: {stats['p_value']:.4f}")
                if use_by_correction:
                    print(f"  Corrected p-value: {stats['corrected_p_value']:.4f}")
                print(f"  Reject null: {stats['reject_null']}")
            
            print(f"\nOverall Winning Rate (ω): {winning_rate:.2f}")
            print("="*70 + "\n")
        
        # Return single dict if testing one LLM, dict of dicts if testing multiple
        if len(all_results) == 1:
            return list(all_results.values())[0]
        else:
            return all_results

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
            if self.llm_annotated and self.llm_annotator_cols:
                cols_to_check.extend(self.llm_annotator_cols)
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
            # Use encoding: 1 = item1 wins, 0 = item2 wins, 0.5 = tie
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
                elif decision == 'Tie':
                    # For ties, both directions are equal
                    comparisons[key1] = 0.5
                    comparisons[key2] = 0.5
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
                            
                            # Check transitivity violations accounting for ties
                            # Using preference encoding: 1 = A > B, 0 = B > A, 0.5 = A = B
                            # Transitivity violations:
                            # 1. If A > B (1) and B > C (1), then A must > C (1), not = C (0.5) or < C (0)
                            # 2. If A < B (0) and B < C (0), then A must < C (0), not = C (0.5) or > C (1)
                            # 3. If A = B (0.5) and B = C (0.5), then A must = C (0.5), not > or < C
                            # 4. If A = B (0.5) and B > C (1), then A must > C (1), not = C (0.5) or < C (0)
                            # 5. If A = B (0.5) and B < C (0), then A must < C (0), not = C (0.5) or > C (1)
                            # 6. If A > B (1) and B = C (0.5), then A must > C (1), not = C (0.5) or < C (0)
                            # 7. If A < B (0) and B = C (0.5), then A must < C (0), not = C (0.5) or > C (1)
                            
                            is_violation = False
                            
                            # Case 1: A > B and B > C
                            if ab_decision == 1 and bc_decision == 1:
                                if ac_decision != 1:
                                    is_violation = True
                            
                            # Case 2: A < B and B < C
                            elif ab_decision == 0 and bc_decision == 0:
                                if ac_decision != 0:
                                    is_violation = True
                            
                            # Case 3: A = B and B = C
                            elif ab_decision == 0.5 and bc_decision == 0.5:
                                if ac_decision != 0.5:
                                    is_violation = True
                            
                            # Case 4: A = B and B > C
                            elif ab_decision == 0.5 and bc_decision == 1:
                                if ac_decision != 1:
                                    is_violation = True
                            
                            # Case 5: A = B and B < C
                            elif ab_decision == 0.5 and bc_decision == 0:
                                if ac_decision != 0:
                                    is_violation = True
                            
                            # Case 6: A > B and B = C
                            elif ab_decision == 1 and bc_decision == 0.5:
                                if ac_decision != 1:
                                    is_violation = True
                            
                            # Case 7: A < B and B = C
                            elif ab_decision == 0 and bc_decision == 0.5:
                                if ac_decision != 0:
                                    is_violation = True
                            
                            if is_violation:
                                violations += 1
            
            transitivity_score = 1 - (violations / total_triples) if total_triples > 0 else 0
            results[annotator_col] = (transitivity_score, violations, total_triples)
        
        return results

    def dawid_skene_annotator_ranking(
        self,
        annotator_cols: Optional[List[str]] = None,
        num_classes: int = 2,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_seed: Optional[int] = None,
        return_confusion_matrices: bool = False) -> pd.DataFrame:
        """
        Apply Dawid-Skene model to rank all annotators (human and LLM) by reliability.
        
        This method estimates each annotator's reliability by computing their accuracy
        (diagonal sum of confusion matrix) using the Dawid-Skene EM algorithm.
        
        Parameters
        ----------
        annotator_cols : List[str], optional
            List of annotator column names to include. If None, uses all available
            annotators (human annotator_cols + LLM decision columns)
        num_classes : int, default=2
            Number of classes (e.g., 2 for binary Text1/Text2)
        max_iter : int, default=100
            Maximum iterations for Dawid-Skene EM algorithm
        tol : float, default=1e-6
            Convergence tolerance for Dawid-Skene
        return_confusion_matrices : bool, default=False
            If True, includes confusion matrices in the returned DataFrame
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'annotator': Annotator name
            - 'reliability': Reliability score (mean diagonal of confusion matrix)
            - 'rank': Rank (1 = most reliable)
            - 'type': 'Human' or 'LLM'
            - 'confusion_matrix': (optional) Full confusion matrix as nested array
            
        Examples
        --------
        >>> ranking = pairadigm_obj.dawid_skene_annotator_ranking()
        >>> print(ranking[['annotator', 'reliability', 'rank', 'type']])
        """
        
        if not self.llm_annotated and not self.annotated:
            raise ValueError("Data must have annotations to rank annotators.")
        
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison data found")
        
        if random_seed is None:
            raise ValueError("A seed is required for reproducibility of the EM algorithm. Recommended practice is to run results over multiple seeds to avoid seed hacking.")
        
        # Determine which annotators to include
        if annotator_cols is None:
            annotator_cols = []
            
            # Add human annotators
            if self.annotator_cols:
                annotator_cols.extend(self.annotator_cols)
            
            # Add LLM annotators (decision columns)
            llm_cols = [col for col in self.pairwise_df.columns if col.startswith('decision')]
            annotator_cols.extend(llm_cols)
            
            if not annotator_cols:
                raise ValueError("No annotator columns found.")
        
        # Validate all columns exist
        for col in annotator_cols:
            if col not in self.pairwise_df.columns:
                raise ValueError(f"Column '{col}' not found in pairwise_df")
        
        # Get unique instances
        instances = self.pairwise_df.index.tolist()
        num_instances = len(instances)
        num_annotators = len(annotator_cols)
        
        print(f"Ranking {num_annotators} annotators across {num_instances} instances...")
        
        # Create annotation matrix: rows=instances, cols=annotators
        annotator_labels = np.full((num_instances, num_annotators), -1, dtype=int)
        
        for j, col in enumerate(annotator_cols):
            for i in range(num_instances):
                val = self.pairwise_df.iloc[i][col]
                if pd.isna(val):
                    continue  # Keep as -1 for missing
                elif val == 'Text1' or val == 0:
                    annotator_labels[i, j] = 0
                elif val == 'Text2' or val == 1:
                    annotator_labels[i, j] = 1
        
        # Dawid-Skene EM Algorithm
        def dawid_skene_em(labels, num_classes, max_iter, tol, random_seed):
            n_instances, n_annotators = labels.shape
            
            np.random.seed(random_seed)

            # Initialize with majority vote (ignoring missing -1 values)
            label_probs = np.zeros((n_instances, num_classes))
            for i in range(n_instances):
                valid_labels = labels[i][labels[i] >= 0]
                if len(valid_labels) > 0:
                    majority = np.bincount(valid_labels, minlength=num_classes)
                    label_probs[i] = majority / majority.sum()
                else:
                    label_probs[i] = 1 / num_classes
            
            # Initialize confusion matrices
            confusion_matrices = np.full((n_annotators, num_classes, num_classes), 
                                        1 / num_classes)
            confusion_matrices += np.random.randn(n_annotators, num_classes, num_classes) * 0.01
            confusion_matrices = np.abs(confusion_matrices)
            
            # Normalize rows
            for j in range(n_annotators):
                for c in range(num_classes):
                    row_sum = confusion_matrices[j, c].sum()
                    if row_sum > 0:
                        confusion_matrices[j, c] /= row_sum
            
            prev_label_probs = label_probs.copy()
            converged_iter = max_iter
            
            for iteration in range(max_iter):
                # E-step
                for i in range(n_instances):
                    for c in range(num_classes):
                        prob = 1.0
                        for j in range(n_annotators):
                            observed = labels[i, j]
                            if observed >= 0:  # Skip missing annotations
                                prob *= confusion_matrices[j, c, observed]
                        label_probs[i, c] = prob
                    
                    # Normalize
                    total = np.sum(label_probs[i])
                    if total > 0:
                        label_probs[i] /= total
                    else:
                        label_probs[i] = 1 / num_classes
                
                # M-step
                for j in range(n_annotators):
                    for c in range(num_classes):
                        for k in range(num_classes):
                            numerator = sum(label_probs[i, c] 
                                        for i in range(n_instances) 
                                        if labels[i, j] == k)
                            denominator = sum(label_probs[i, c] 
                                            for i in range(n_instances)
                                            if labels[i, j] >= 0)  # Count only valid annotations
                            if denominator > 0:
                                confusion_matrices[j, c, k] = numerator / denominator
                            else:
                                confusion_matrices[j, c, k] = 1 / num_classes
                
                # Check convergence
                if iteration > 0 and np.linalg.norm(label_probs - prev_label_probs) < tol:
                    converged_iter = iteration
                    print(f"Dawid-Skene converged at iteration {iteration}")
                    break
                prev_label_probs = label_probs.copy()
            
            # Compute annotator reliability (mean diagonal)
            annotator_reliability = np.array([
                np.mean(np.diag(confusion_matrices[j])) 
                for j in range(n_annotators)
            ])
            
            return annotator_reliability, confusion_matrices, converged_iter
        
        # Run Dawid-Skene
        reliability_scores, confusion_matrices, conv_iter = dawid_skene_em(
            annotator_labels, num_classes, max_iter, tol, random_seed
        )
        
        # Create results DataFrame
        results = []
        for j, col in enumerate(annotator_cols):
            annotator_type = 'LLM' if col.startswith('decision') else 'Human'
            
            result_dict = {
                'annotator': col,
                'reliability': reliability_scores[j],
                'type': annotator_type
            }
            
            if return_confusion_matrices:
                result_dict['confusion_matrix'] = confusion_matrices[j].tolist()
            
            results.append(result_dict)
        
        results_df = pd.DataFrame(results)
        
        # Add rank (1 = most reliable)
        results_df['rank'] = results_df['reliability'].rank(ascending=False, method='min').astype(int)
        
        # Sort by reliability
        results_df = results_df.sort_values('reliability', ascending=False).reset_index(drop=True)
        
        # Print summary
        print("\n" + "="*70)
        print("DAWID-SKENE ANNOTATOR RANKING")
        print("="*70)
        print(f"Converged at iteration: {conv_iter}")
        print(f"\nTop 5 Most Reliable Annotators:")
        print(results_df[['rank', 'annotator', 'reliability', 'type']].head())
        print("\n" + "="*70 + "\n")
        
        return results_df

    def irr(
        self,
        method: str = 'auto',
        alpha_level: str = 'nominal',
        min_overlap: int = 2) -> Dict[str, Dict[str, float]]:
        """
        Calculate inter-rater reliability (IRR) between annotators.
        
        Computes IRR separately for:
        - Human annotators only
        - LLM annotators only
        - All annotators combined
        
        Uses Cohen's Kappa for 2 raters, Fleiss' Kappa or Krippendorff's Alpha for 3+ raters.
        Automatically handles tie values if present in the data.
        
        Parameters
        ----------
        method : str, default='auto'
            IRR method to use:
            - 'auto': Uses Cohen's Kappa for 2 raters, Krippendorff's Alpha for 3+
            - 'cohens_kappa': Cohen's Kappa (only for 2 raters)
            - 'fleiss_kappa': Fleiss' Kappa (for 3+ raters, assumes complete overlap)
            - 'krippendorff': Krippendorff's Alpha (handles missing data)
        alpha_level : str, default='nominal'
            Level of measurement for Krippendorff's Alpha:
            - 'nominal': For categorical data
            - 'ordinal': For ordered categories
            - 'interval': For numeric scales
            - 'ratio': For ratio scales
        min_overlap : int, default=2
            Minimum number of annotators required per item for inclusion
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary with keys 'human', 'llm', 'all', each containing:
            - 'method': Method used
            - 'score': IRR score
            - 'n_annotators': Number of annotators
            - 'n_items': Number of items
            - 'interpretation': Qualitative interpretation
            
        Raises
        ------
        ValueError
            If insufficient annotators exist for IRR calculation
            
        Examples
        --------
        >>> results = pairadigm_obj.irr()
        >>> print(f"Human IRR: {results['human']['score']:.3f}")
        >>> print(f"LLM IRR: {results['llm']['score']:.3f}")
        >>> print(f"All IRR: {results['all']['score']:.3f}")
        """
        from sklearn.metrics import cohen_kappa_score
        import numpy as np
        import pandas as pd
        
        if not self.annotated and not self.llm_annotated:
            raise ValueError("No annotations found. Data must have human or LLM annotations.")
        
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison data found.")
        
        def interpret_kappa(score: float) -> str:
            """Interpret kappa score using Landis & Koch (1977) scale."""
            if score < 0:
                return "Poor (worse than chance)"
            elif score < 0.20:
                return "Slight"
            elif score < 0.40:
                return "Fair"
            elif score < 0.60:
                return "Moderate"
            elif score < 0.80:
                return "Substantial"
            else:
                return "Almost Perfect"
        
        def cohens_kappa(annotations1, annotations2):
            """Calculate Cohen's Kappa for two raters."""
            # Filter to items both annotated
            mask = (~pd.isna(annotations1)) & (~pd.isna(annotations2))
            if mask.sum() < 2:
                raise ValueError("Insufficient overlapping annotations (need at least 2)")
            
            return cohen_kappa_score(annotations1[mask], annotations2[mask])
        
        def fleiss_kappa(annotation_matrix, num_categories=None):
            """
            Calculate Fleiss' Kappa for multiple raters.
            annotation_matrix: rows=items, cols=raters
            num_categories: int, optional - number of categories (detected if None)
            """
            # Remove rows with missing data
            complete_cases = ~np.isnan(annotation_matrix).any(axis=1)
            if complete_cases.sum() < 2:
                raise ValueError("Insufficient complete cases for Fleiss' Kappa")
            
            matrix = annotation_matrix[complete_cases]
            n, k = matrix.shape  # n items, k raters
            
            # Get unique categories
            if num_categories is None:
                categories = np.unique(matrix[~np.isnan(matrix)])
                n_cat = len(categories)
            else:
                categories = np.arange(num_categories)
                n_cat = num_categories
            
            # Build frequency table
            freq_table = np.zeros((n, n_cat))
            for i, cat in enumerate(categories):
                freq_table[:, i] = (matrix == cat).sum(axis=1)
            
            # Calculate p_j (proportion of all assignments in category j)
            p_j = freq_table.sum(axis=0) / (n * k)
            
            # Calculate P_i (extent of agreement for item i)
            P_i = (freq_table ** 2).sum(axis=1) - k
            P_i = P_i / (k * (k - 1))
            
            # Calculate P_bar (mean of P_i)
            P_bar = P_i.mean()
            
            # Calculate P_e_bar (expected agreement by chance)
            P_e_bar = (p_j ** 2).sum()
            
            # Calculate Fleiss' Kappa
            if P_e_bar == 1:
                return 1.0
            kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
            return kappa
        
        def krippendorff_alpha(annotation_matrix, level='nominal', num_categories=None):
            """
            Calculate Krippendorff's Alpha.
            annotation_matrix: rows=items, cols=raters
            level: measurement level ('nominal', 'ordinal', 'interval', 'ratio')
            num_categories: int, optional - number of categories (detected if None)
            Handles missing data.
            """
            matrix = annotation_matrix.copy()
            n_items, n_raters = matrix.shape
            
            # Build coincidence matrix
            if num_categories is None:
                categories = np.unique(matrix[~np.isnan(matrix)])
                n_cat = len(categories)
            else:
                categories = np.arange(num_categories)
                n_cat = num_categories
            cat_to_idx = {cat: i for i, cat in enumerate(categories)}
            
            coincidence = np.zeros((n_cat, n_cat))
            
            for i in range(n_items):
                valid_ratings = matrix[i][~np.isnan(matrix[i])]
                n_valid = len(valid_ratings)
                
                if n_valid < 2:
                    continue
                
                for c1 in valid_ratings:
                    for c2 in valid_ratings:
                        if c1 != c2 or level == 'nominal':
                            idx1, idx2 = cat_to_idx[c1], cat_to_idx[c2]
                            coincidence[idx1, idx2] += 1 / (n_valid - 1)
            
            # Calculate observed disagreement
            n_total = coincidence.sum()
            if n_total == 0:
                raise ValueError("No valid pairs for Krippendorff's Alpha")
            
            D_o = 0
            for c1_idx in range(n_cat):
                for c2_idx in range(n_cat):
                    if c1_idx != c2_idx:
                        if level == 'nominal':
                            delta = 1
                        elif level == 'ordinal':
                            delta = (c1_idx - c2_idx) ** 2
                        elif level in ['interval', 'ratio']:
                            delta = (categories[c1_idx] - categories[c2_idx]) ** 2
                        else:
                            delta = 1
                        D_o += coincidence[c1_idx, c2_idx] * delta
            
            D_o /= n_total
            
            # Calculate expected disagreement
            n_c = coincidence.sum(axis=0) + coincidence.sum(axis=1)
            D_e = 0
            for c1_idx in range(n_cat):
                for c2_idx in range(n_cat):
                    if c1_idx != c2_idx:
                        if level == 'nominal':
                            delta = 1
                        elif level == 'ordinal':
                            delta = (c1_idx - c2_idx) ** 2
                        elif level in ['interval', 'ratio']:
                            delta = (categories[c1_idx] - categories[c2_idx]) ** 2
                        else:
                            delta = 1
                        D_e += n_c[c1_idx] * n_c[c2_idx] * delta
            
            D_e /= (n_total * (n_total - 1))
            
            if D_e == 0:
                return 1.0
            alpha = 1 - (D_o / D_e)
            return alpha
        
        def calculate_irr(annotator_cols, label):
            """Calculate IRR for a set of annotators."""
            if len(annotator_cols) == 0:
                return None
            
            if len(annotator_cols) == 1:
                raise ValueError(f"Cannot calculate IRR for {label} with only 1 annotator")
            
            # First pass: detect unique values to determine if ties are present
            unique_values = set()
            for col in annotator_cols:
                col_values = self.pairwise_df[col].dropna().unique()
                unique_values.update(col_values)
            
            # Check for tie values
            has_ties = any(val in ['Tie', 'tie', 2] for val in unique_values)
            num_categories = 3 if has_ties else 2
            
            # Build annotation matrix
            annotation_matrix = np.full((len(self.pairwise_df), len(annotator_cols)), np.nan)
            
            for j, col in enumerate(annotator_cols):
                for i in range(len(self.pairwise_df)):
                    val = self.pairwise_df.iloc[i][col]
                    if pd.isna(val):
                        continue
                    elif val == 'Text1' or val == 0:
                        annotation_matrix[i, j] = 0
                    elif val == 'Text2' or val == 1:
                        annotation_matrix[i, j] = 1
                    elif val in ['Tie', 'tie', 2]:
                        annotation_matrix[i, j] = 2
            
            # Filter items with sufficient overlap
            overlap_counts = (~np.isnan(annotation_matrix)).sum(axis=1)
            valid_items = overlap_counts >= min_overlap
            
            if valid_items.sum() < 2:
                raise ValueError(f"Insufficient items with {min_overlap}+ annotators for {label}")
            
            filtered_matrix = annotation_matrix[valid_items]
            
            # Choose method
            n_annotators = len(annotator_cols)
            
            if method == 'auto':
                if n_annotators == 2:
                    chosen_method = 'cohens_kappa'
                else:
                    chosen_method = 'krippendorff'
            else:
                chosen_method = method
            
            # Calculate IRR
            if chosen_method == 'cohens_kappa':
                if n_annotators != 2:
                    raise ValueError("Cohen's Kappa requires exactly 2 annotators")
                score = cohens_kappa(filtered_matrix[:, 0], filtered_matrix[:, 1])
            
            elif chosen_method == 'fleiss_kappa':
                if n_annotators < 3:
                    raise ValueError("Fleiss' Kappa requires 3+ annotators")
                score = fleiss_kappa(filtered_matrix, num_categories=num_categories)
            
            elif chosen_method == 'krippendorff':
                score = krippendorff_alpha(filtered_matrix, level=alpha_level, num_categories=num_categories)
            
            else:
                raise ValueError(f"Unknown method: {chosen_method}")
            
            return {
                'method': chosen_method,
                'score': score,
                'n_annotators': n_annotators,
                'n_items': valid_items.sum(),
                'interpretation': interpret_kappa(score)
            }
        
        # Calculate IRR for each group
        results = {}
        
        # Human annotators
        if self.annotated and self.annotator_cols:
            try:
                results['human'] = calculate_irr(self.annotator_cols, 'human annotators')
            except ValueError as e:
                results['human'] = {'error': str(e)}
        
        # LLM annotators
        if self.llm_annotated and self.llm_annotator_cols:
            try:
                results['llm'] = calculate_irr(self.llm_annotator_cols, 'LLM annotators')
            except ValueError as e:
                results['llm'] = {'error': str(e)}
        
        # All annotators
        all_cols = []
        if self.annotator_cols:
            all_cols.extend(self.annotator_cols)
        if self.llm_annotator_cols:
            all_cols.extend(self.llm_annotator_cols)
        
        if len(all_cols) >= 2:
            try:
                results['all'] = calculate_irr(all_cols, 'all annotators')
            except ValueError as e:
                results['all'] = {'error': str(e)}
        
        # Print results
        print("\n" + "="*70)
        print("INTER-RATER RELIABILITY RESULTS")
        print("="*70)
        
        for group in ['human', 'llm', 'all']:
            if group in results:
                print(f"\n{group.upper()} ANNOTATORS:")
                if 'error' in results[group]:
                    print(f"  Error: {results[group]['error']}")
                else:
                    r = results[group]
                    print(f"  Method: {r['method'].replace('_', ' ').title()}")
                    print(f"  Score: {r['score']:.3f}")
                    print(f"  Interpretation: {r['interpretation']}")
                    print(f"  Annotators: {r['n_annotators']}")
                    print(f"  Items: {r['n_items']}")
        
        print("="*70 + "\n")
        
        # Convert results to DataFrame
        df_rows = []
        for group in ['human', 'llm', 'all']:
            if group in results:
                row = {'group': group}
                if 'error' in results[group]:
                    row['error'] = results[group]['error']
                    row['method'] = None
                    row['score'] = None
                    row['n_annotators'] = None
                    row['n_items'] = None
                    row['interpretation'] = None
                else:
                    row['error'] = None
                    row['method'] = results[group]['method']
                    row['score'] = results[group]['score']
                    row['n_annotators'] = results[group]['n_annotators']
                    row['n_items'] = results[group]['n_items']
                    row['interpretation'] = results[group]['interpretation']
                df_rows.append(row)
        
        results_df = pd.DataFrame(df_rows)

        return results_df

################################
# SCORING AND SUMMARIZATION OF ITEMS
################################
    
    def _DEP_score_items(self, 
                    normalization_scale='zero-to-one',
                    update_classObject=True,
                    summarize=True,
                    decision_col: str = 'decision') -> pd.DataFrame:
        """
        Compute Bradley-Terry scores from pairwise comparison results.
        
        Args:
            normalization_scale (str): How to normalize scores. Options: 'zero-to-one', 'negative-one-to-one', 'none'
            update_classObject (bool, optional): If True, updates self.scored_df. Defaults to True.
            summarize (bool, optional): If True, prints summary statistics. Defaults to True.
            decision_col (str, optional): Name of the decision column to use. Defaults to 'decision'.
                For multiple clients, use format 'decision_<model_name>'

        Returns:
            pd.DataFrame: Original DataFrame with added 'Bradley_Terry_Score' column
        """
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison results found. Run generate_pairwise_annotations() first.")

        if decision_col not in self.pairwise_df.columns:
            available_decision_cols = [col for col in self.pairwise_df.columns if col.startswith('decision')]
            raise ValueError(
                f"Decision column '{decision_col}' not found in pairwise_df."
                f"Available decision columns: {available_decision_cols}"
            )

        # Filter out invalid decisions
        valid_df = self.pairwise_df[self.pairwise_df[decision_col].isin(['Text1', 'Text2'])]

        if len(valid_df) == 0:
            raise ValueError("No valid comparisons found to compute Bradley-Terry scores. Please make sure the value in the decision_col are 'Text1' or 'Text2'.")
        
        if len(valid_df) < len(self.pairwise_df):
            warnings.warn("Some rows filtered out due to not containing 'Text1' or 'Text2' in the decision_col. If scoring human annotations, please adjust those values accordingly.")

        # Prepare data for Bradley-Terry model, handling different self.data formats for item mapping
        if self.paired:
            # For paired data, collect unique items from both item ID columns
            item1_col, item2_col = self.item_id_cols
            all_items = pd.concat([
                self.pairwise_df[item1_col],
                self.pairwise_df[item2_col]
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
            decision = row[decision_col]
            
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

        # Determine score column name
        score_col_name = 'Bradley_Terry_Score' if decision_col == 'decision' else f'Bradley_Terry_Score_{decision_col.replace("decision_", "")}'

        # Create scored DataFrame differently based on paired/unpaired data
        if self.paired:
            # For paired data, create a new DataFrame with unique items and their scores
            scored_df = pd.DataFrame({
                'item_id': list(item_to_idx.keys()),
                score_col_name: [bt_scores[item_to_idx[item]] for item in item_to_idx.keys()]
            })
        else:
            # For unpaired data, add scores to original DataFrame or scored_df if it exists
            if self.scored_df is not None:
                scored_df = self.scored_df.copy()
            else:
                scored_df = self.data.copy()

            scored_df[score_col_name] = [bt_scores[item_to_idx[uuid]] for uuid in scored_df[self.item_id_name]]

        model_label = decision_col.replace('decision_', '') if decision_col != 'decision' else 'default'
        print(f"[{model_label}] Bradley-Terry model fitted with {len(comparisons)} comparisons")
        print(f"[{model_label}] Mean {self.target_concept} score: {scored_df[score_col_name].mean():.3f}")
        print(f"[{model_label}] Std {self.target_concept} score: {scored_df[score_col_name].std():.3f}")

        if summarize:
        # For paired data, we don't have text_col in scored_df, so skip summarize or handle differently
            if self.paired:
                print("\nSummary statistics:")
                summary = {
                    'mean': scored_df[score_col_name].mean(),
                    'median': scored_df[score_col_name].median(),
                    'std': scored_df[score_col_name].std(),
                    'min': scored_df[score_col_name].min(),
                    'max': scored_df[score_col_name].max(),
                    'count': scored_df[score_col_name].count()
                }
                for k, v in summary.items():
                    print(f"{k}: {v:.3f}")
            else:
                summary = self.summarize_scores(df=scored_df, 
                                                text_col=self.text_name, 
                                                score_col=score_col_name)
                for k, v in summary.items():
                    print(f"{k}: {v:.3f}")
        
        # Update instance if requested
        if update_classObject:
            self.scored_df = scored_df
            
        return scored_df

    def score_items(self, 
                normalization_scale='zero-to-one',
                update_classObject=True,
                summarize=True,
                decision_col: str = 'decision',
                use_davidson: Optional[bool] = None) -> pd.DataFrame:
        """
        Compute Bradley-Terry or Davidson scores from pairwise comparison results.
        Automatically detects ties and uses Davidson model if present, Bradley-Terry otherwise.
        
        Args:
            normalization_scale (str): How to normalize scores. Options: 'zero-to-one', 'negative-one-to-one', 'none'
            update_classObject (bool, optional): If True, updates self.scored_df. Defaults to True.
            summarize (bool, optional): If True, prints summary statistics. Defaults to True.
            decision_col (str, optional): Name of the decision column to use. Defaults to 'decision'.
                For multiple clients, use format 'decision_<model_name>'
            use_davidson (bool, optional): Force use of Davidson model. If None, auto-detects based on ties.

        Returns:
            pd.DataFrame: Original DataFrame with added score column
        """
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparison results found. Run generate_pairwise_annotations() first.")

        if decision_col not in self.pairwise_df.columns:
            available_decision_cols = [col for col in self.pairwise_df.columns if col.startswith('decision')]
            raise ValueError(
                f"Decision column '{decision_col}' not found in pairwise_df."
                f"Available decision columns: {available_decision_cols}"
            )

        # Check for ties in the data
        tie_values = ['Tie', 'tie', 2, 0.5]
        has_ties = self.pairwise_df[decision_col].isin(tie_values).any()
        
        # Determine which model to use
        if use_davidson is None:
            use_davidson = has_ties
            if has_ties:
                num_ties = self.pairwise_df[decision_col].isin(tie_values).sum()
                print(f"Detected {num_ties} ties in data. Using Davidson model.")
        
        # Filter valid decisions based on model
        if use_davidson:
            valid_values = ['Text1', 'Text2', 'Tie', 'tie', 0, 1, 2, 0.5]
        else:
            valid_values = ['Text1', 'Text2', 0, 1]
        
        valid_df = self.pairwise_df[self.pairwise_df[decision_col].isin(valid_values)]

        if len(valid_df) == 0:
            raise ValueError("No valid comparisons found to compute scores.")
        
        if len(valid_df) < len(self.pairwise_df):
            warnings.warn(f"Some rows filtered out due to invalid decision values. Using {len(valid_df)}/{len(self.pairwise_df)} comparisons.")

        # Prepare item mapping
        if self.paired:
            item1_col, item2_col = self.item_id_cols
            all_items = pd.concat([
                self.pairwise_df[item1_col],
                self.pairwise_df[item2_col]
            ]).unique().tolist()
            item_to_idx = {item: idx for idx, item in enumerate(all_items)}
        else:
            item_to_idx = {item: idx for idx, item in enumerate(self.data[self.item_id_name].tolist())}

        idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        n_items = len(item_to_idx)

        if use_davidson:
            # Prepare data for Davidson model
            # Create comparison matrix: wins[i,j] = number of times i beat j
            # ties[i,j] = number of ties between i and j
            wins = np.zeros((n_items, n_items))
            ties = np.zeros((n_items, n_items))
            
            for _, row in valid_df.iterrows():
                i = item_to_idx[row['item1']]
                j = item_to_idx[row['item2']]
                decision = row[decision_col]
                
                if decision in ['Text1', 0]:
                    wins[i, j] += 1
                elif decision in ['Text2', 1]:
                    wins[j, i] += 1
                elif decision in ['Tie', 'tie', 2]:
                    ties[i, j] += 1
                    ties[j, i] += 1  # Symmetric
            
            # Fit Davidson model using MM algorithm
            # Initialize with uniform scores
            scores = np.ones(n_items)
            nu = 1.0  # Tie parameter
            max_iter = 1000
            tol = 1e-6
            
            for iteration in range(max_iter):
                scores_old = scores.copy()
                
                # Update scores
                for i in range(n_items):
                    numerator = 0
                    denominator = 0
                    
                    for j in range(n_items):
                        if i != j:
                            # Wins
                            numerator += wins[i, j]
                            denominator += (wins[i, j] + wins[j, i]) / (scores[i] + scores[j])
                            
                            # Ties
                            numerator += 0.5 * ties[i, j]
                            denominator += ties[i, j] * nu / (scores[i] + scores[j] + 2 * nu)
                    
                    if denominator > 0:
                        scores[i] = numerator / denominator
                
                # Normalize to prevent overflow
                scores = scores / scores.sum() * n_items
                
                # Check convergence
                if np.linalg.norm(scores - scores_old) < tol:
                    print(f"Davidson model converged in {iteration + 1} iterations")
                    break
            
            bt_scores = scores
            model_name = "Davidson"
            
        else:
            # Use Bradley-Terry model (original implementation)
            comparisons = []
            for _, row in valid_df.iterrows():
                item1_idx = item_to_idx[row['item1']]
                item2_idx = item_to_idx[row['item2']]
                decision = row[decision_col]
                
                if decision in ['Text1', 0]:
                    comparisons.append((item1_idx, item2_idx))
                elif decision in ['Text2', 1]:
                    comparisons.append((item2_idx, item1_idx))

            if not comparisons:
                raise ValueError("No valid comparisons to compute Bradley-Terry scores.")

            # Fit Bradley-Terry model
            bt_scores = choix.ilsr_pairwise(len(item_to_idx), comparisons, alpha=0.1)
            model_name = "Bradley-Terry"

        # Normalize scores
        if normalization_scale == 'zero-to-one':
            bt_scores = (bt_scores - bt_scores.min()) / (bt_scores.max() - bt_scores.min())
        elif normalization_scale == 'negative-one-to-one':
            bt_scores = 2 * (bt_scores - bt_scores.min()) / (bt_scores.max() - bt_scores.min()) - 1
        elif normalization_scale == 'none':
            pass
        else:
            raise ValueError("normalization_scale must be 'zero-to-one', 'negative-one-to-one', or 'none'")

        # Determine score column name
        score_col_name = f'{model_name.replace("-", "_")}_Score' if decision_col == 'decision' else f'{model_name.replace("-", "_")}_Score_{decision_col.replace("decision_", "")}'

        # Create scored DataFrame
        if self.paired:
            scored_df = pd.DataFrame({
                'item_id': list(item_to_idx.keys()),
                score_col_name: [bt_scores[item_to_idx[item]] for item in item_to_idx.keys()]
            })
        else:
            if self.scored_df is not None:
                scored_df = self.scored_df.copy()
            else:
                scored_df = self.data.copy()
            scored_df[score_col_name] = [bt_scores[item_to_idx[uuid]] for uuid in scored_df[self.item_id_name]]

        model_label = decision_col.replace('decision_', '') if decision_col != 'decision' else 'default'
        print(f"[{model_label}] {model_name} model fitted with {len(valid_df)} comparisons")
        if use_davidson:
            num_ties = valid_df[decision_col].isin(tie_values).sum()
            print(f"[{model_label}] Including {num_ties} tie decisions")
        print(f"[{model_label}] Mean {self.target_concept} score: {scored_df[score_col_name].mean():.3f}")
        print(f"[{model_label}] Std {self.target_concept} score: {scored_df[score_col_name].std():.3f}")

        if summarize:
            if self.paired:
                print("\nSummary statistics:")
                summary = {
                    'mean': scored_df[score_col_name].mean(),
                    'median': scored_df[score_col_name].median(),
                    'std': scored_df[score_col_name].std(),
                    'min': scored_df[score_col_name].min(),
                    'max': scored_df[score_col_name].max(),
                    'count': scored_df[score_col_name].count()
                }
                for k, v in summary.items():
                    print(f"{k}: {v:.3f}")
            else:
                summary = self.summarize_scores(df=scored_df, 
                                                text_col=self.text_name, 
                                                score_col=score_col_name)
                for k, v in summary.items():
                    print(f"{k}: {v:.3f}")
        
        if update_classObject:
            self.scored_df = scored_df
            
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
            raise ValueError(f"Column '{score_col}' not found in scored DataFrame. Available columns: {self.scored_df.columns}")
        
        # Auto-generate title if not provided
        if title is None:
            title = f'Distribution of {self.target_concept.title()} Scores'
        
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
            raise ValueError("No 'decision' column found. Run generate_pairwise_annotations() first or appened_human_annotations().")
        
        # Turn decision_col to 'decision' for consistency below
        if decision_col != 'decision':
            self.pairwise_df['decision'] = self.pairwise_df[decision_col]

        # Check for ties and warn if found
        tie_values = ['Tie', 'tie', 2, 0.5]
        ties_present = self.pairwise_df['decision'].isin(tie_values).any()
        if ties_present:
            num_ties = self.pairwise_df['decision'].isin(tie_values).sum()
            total_comparisons = len(self.pairwise_df)
            tie_percentage = (num_ties / total_comparisons) * 100
            warnings.warn(
                f"Network plot excludes {num_ties} tie decisions ({tie_percentage:.1f}% of comparisons). "
                f"Ties represent no directional preference and cannot be represented as directed edges.",
                UserWarning
            )

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
            if row['decision'] == 'Text1' or row['decision'] == 0:
                G.add_edge(row['item1'], row['item2'])
            elif row['decision'] == 'Text2' or row['decision'] == 1:
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

    def plot_epsilon_sensitivity(
        self,
        epsilon_range: Tuple[float, float] = (-0.1, 0.25),
        epsilon_step: float = 0.01,
        test_all_llms: bool = True,
        figsize: Tuple[float, float] = (12, 7),
        style: str = 'whitegrid',
        palette: str = 'husl',
        show_annotations: bool = True,
        return_data: bool = False) -> Union[None, Tuple[plt.Figure, pd.DataFrame]]:
        """
        Plot winning rate as a function of epsilon values for ALT-TEST validation.
        
        This visualization helps determine appropriate epsilon thresholds for different
        annotator quality levels (crowdworkers, trained annotators, experts).
        
        Parameters
        ----------
        epsilon_range : Tuple[float, float], default=(-0.1, 0.25)
            Range of epsilon values to test (start, end)
        epsilon_step : float, default=0.01
            Step size between epsilon values
        test_all_llms : bool, default=True
            Whether to test all LLM models or just the default
        figsize : Tuple[float, float], default=(12, 7)
            Figure size (width, height) in inches
        style : str, default='whitegrid'
            Seaborn style to use ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        palette : str, default='husl'
            Color palette for different models
        show_annotations : bool, default=True
            Whether to show reference lines for annotator types
        return_data : bool, default=False
            If True, returns (figure, data_df) instead of just showing plot
            
        Returns
        -------
        None or Tuple[plt.Figure, pd.DataFrame]
            If return_data=True, returns the figure and a DataFrame with all results
            
        Examples
        --------
        >>> # Basic usage
        >>> pairadigm_obj.plot_epsilon_sensitivity()
        
        >>> # Get underlying data
        >>> fig, data = pairadigm_obj.plot_epsilon_sensitivity(return_data=True)
        >>> print(data.head())
        
        >>> # Custom epsilon range for crowdworker validation
        >>> pairadigm_obj.plot_epsilon_sensitivity(epsilon_range=(0.0, 0.15))
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Check for annotations
        if not self.annotated:
            raise ValueError("Data must have human annotations to perform the alt-test and epsilon sensitivity analysis")
        if not self.annotator_cols or len(self.annotator_cols) == 0:
            raise ValueError("No annotator columns found for human annotations")
        if not self.llm_annotator_cols or len(self.llm_annotator_cols) == 0:
            raise ValueError("No LLM annotator columns found for LLM annotations")
        
        # Generate epsilon values
        epsilon_values = np.arange(epsilon_range[0], epsilon_range[1] + epsilon_step, epsilon_step)
        
        print(f"Testing {len(epsilon_values)} epsilon values from {epsilon_range[0]} to {epsilon_range[1]}...")
        
        # Store results
        all_results = []
        
        # Calculate winning rate for each epsilon
        for i, eps in enumerate(epsilon_values):
            try:
                result = self.alt_test(epsilon=eps, test_all_llms=test_all_llms)
                all_results.append(result)
                    
            except Exception as e:
                print(f"Warning: Failed at epsilon={eps:.3f}: {e}")
                all_results.append(None)
        
        # Extract model names from first valid result
        if test_all_llms:
            model_names = list(all_results[0].keys()) if all_results[0] else []
        else:
            model_names = ['default']
        
        # Prepare data for plotting
        plot_data = []
        for eps, result in zip(epsilon_values, all_results):
            if result is None:
                continue
            if test_all_llms:
                for model in model_names:
                    plot_data.append({
                        'epsilon': eps,
                        'winning_rate': result[model][0],
                        'advantage_prob': result[model][1],
                        'model': model
                    })
            else:
                plot_data.append({
                    'epsilon': eps,
                    'winning_rate': result[0],
                    'advantage_prob': result[1],
                    'model': 'default'
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Set style
        sns.set_style(style)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot lines for each model
        colors = sns.color_palette(palette, n_colors=len(model_names))
        
        for idx, model in enumerate(model_names):
            model_data = df_plot[df_plot['model'] == model]
            ax.plot(model_data['epsilon'], model_data['winning_rate'], 
                marker='o', markersize=4, linewidth=2.5, 
                label=model, color=colors[idx], alpha=0.8)
        
        # Add reference lines if requested
        if show_annotations:
            # Horizontal line at 0.5
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, 
                    alpha=0.7, label='Approval Threshold', zorder=1)
            
            # Vertical lines for annotator types
            reference_lines = [
                (0.10, 'Crowdworkers', '#e74c3c'),
                (0.15, 'Trained', '#f39c12'),
                (0.20, 'Experts', '#27ae60')
            ]
            
            for eps_val, label, color in reference_lines:
                if epsilon_range[0] <= eps_val <= epsilon_range[1]:
                    ax.axvline(x=eps_val, color=color, linestyle=':', 
                            linewidth=2, alpha=0.6, zorder=1)
                    
                    # Add text annotation
                    y_pos = ax.get_ylim()[1] * 0.95
                    ax.text(eps_val, y_pos, f' {label}\n ε={eps_val}', 
                        rotation=0, verticalalignment='top',
                        horizontalalignment='left', fontsize=9,
                        color=color, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', edgecolor=color, alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Epsilon (ε)', fontsize=12, weight='bold')
        ax.set_ylabel('Winning Rate (ω)', fontsize=12, weight='bold')
        ax.set_title(f'Epsilon Sensitivity Analysis: {self.target_concept.title()}\n' + 
                    f'Winning Rate vs. Epsilon Threshold',
                    fontsize=14, weight='bold', pad=20)
        
        # Legend
        ax.legend(title='Model', title_fontsize=11, fontsize=10,
                loc='best', frameon=True, shadow=True)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Set limits
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(epsilon_range[0] - 0.01, epsilon_range[1] + 0.01)
        
        plt.tight_layout()
        
        if return_data:
            return fig, df_plot
        else:
            plt.show()

##############################
# PERSISTENCE METHODS
##############################

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
            # Temporarily remove unpicklable client objects
            clients = self.clients
            client = self.client
            self.clients = None
            self.client = None
            
            # Temporarily remove the api_key if it exists
            if hasattr(self, 'api_key'):
                api_key = self.api_key
                self.api_key = None
            else:
                api_key = None

            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            
            # Restore clients
            self.clients = clients
            self.client = client
            if api_key is not None:
                self.api_key = api_key
            
            print(f"Pairadigm object saved successfully to: {filepath}")
        except Exception as e:
            # Restore clients even if save fails
            self.clients = clients
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
            
            # Recreate the LLM clients (without requiring API keys for Ollama)
            obj.clients = [LLMClient(model_name=model_name, api_key=None) for model_name in obj.model_names]
            obj.client = obj.clients[0]  # For backward compatibility
            
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
        
        # Recreate the LLM clients (without requiring API keys for Ollama)
        obj.clients = [LLMClient(model_name=model_name, api_key=None) for model_name in obj.model_names]
        obj.client = obj.clients[0]  # For backward compatibility
        
        print(f"Pairadigm object loaded successfully from: {filepath}")
        return obj
    except Exception as e:
        raise IOError(f"Failed to load Pairadigm object: {e}")
    
def build_pairadigm(
    pairadigm_obj: Pairadigm,
    num_pairs_per_item: int = 10,
    random_seed: int = 42,
    max_workers: int = 8,
    rate_limit_per_minute: Optional[int] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    allow_ties: bool = False,
    normalization_scale: str = 'zero-to-one',
    client_indices: Optional[Union[int, List[int]]] = None,
    verbose: bool = True) -> Dict[str, Any]:
    """
    Execute the full basic workflow for Pairadigm analysis.
    
    This function automates the complete pipeline:
    1. Generate CGCoT breakdowns for all items
    2. Generate pairings from items
    3. Generate pairwise LLM annotations
    4. (Optional) If human annotators exist: run ALT-TEST validation, check transitivity, 
       compute IRR, and plot epsilon sensitivity
    
    Parameters
    ----------
    pairadigm_obj : Pairadigm
        Pairadigm object to process
    num_pairs_per_item : int, default=10
        Minimum pairs per item for pairing generation
    random_seed : int, default=42
        Random seed for reproducibility
    max_workers : int, default=8
        Number of parallel workers for LLM calls
    rate_limit_per_minute : int, optional
        Rate limit for API calls
    max_tokens : int, default=1000
        Maximum tokens for LLM responses
    temperature : float, default=0.0
        Sampling temperature for LLM
    allow_ties : bool, default=False
        Whether to allow tie decisions in pairwise comparisons
    normalization_scale : str, default='zero-to-one'
        How to normalize Bradley-Terry scores ('zero-to-one', 'negative-one-to-one', 'none')
    client_indices : int or List[int], optional
        Specific client(s) to use for generation
    verbose : bool, default=True
        Whether to print progress messages
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'breakdowns': Breakdown generation results
        - 'pairings': Generated pairings DataFrame
        - 'annotations': LLM annotation results DataFrame
        - 'alt_test': ALT-TEST results (if human annotations exist)
        - 'transitivity': Transitivity check results (if human annotations exist)
        - 'irr': Inter-rater reliability results (if human annotations exist)
        - 'epsilon_sensitivity_data': Data from epsilon sensitivity plot (if human annotations exist)
        
    Raises
    ------
    ValueError
        If required components are missing or invalid
        
    Examples
    --------
    >>> # Basic workflow
    >>> results = build_pairadigm(pairadigm_obj)
    >>> print(results.keys())
    
    >>> # With custom parameters
    >>> results = build_pairadigm(
    ...     pairadigm_obj,
    ...     num_pairs_per_item=15,
    ...     allow_ties=True,
    ...     client_indices=[0, 1]
    ... )
    
    >>> # Check validation results if human annotations exist
    >>> if 'alt_test' in results:
    ...     print(f"Winning rate: {results['alt_test'][0]:.3f}")
    ...     print(f"Advantage prob: {results['alt_test'][1]:.3f}")
    """
    
    if not isinstance(pairadigm_obj, Pairadigm):
        raise TypeError("pairadigm_obj must be a Pairadigm instance")
    
    results = {}
    
    # ============================================================
    # STEP 1: Generate CGCoT Breakdowns
    # ============================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: GENERATING CGCOT BREAKDOWNS")
        print("="*70)
    
    if pairadigm_obj.cgcot_prompts is None or len(pairadigm_obj.cgcot_prompts) == 0:
        raise ValueError(
            "CGCoT prompts must be set before building. "
            "Use pairadigm_obj.set_cgcot_prompts() to configure prompts."
        )
    
    try:
        if pairadigm_obj.paired:
            # For paired data, generate breakdowns from paired items
            breakdown_results = pairadigm_obj.generate_breakdowns_from_paired(
                max_workers=max_workers,
                rate_limit_per_minute=rate_limit_per_minute,
                update_pairwise_df=True,
                max_tokens=max_tokens,
                tempature=temperature,
                client_indices=client_indices
            )
        else:
            # For unpaired data, generate breakdowns for all items
            breakdown_results = pairadigm_obj.generate_breakdowns(
                max_workers=max_workers,
                rate_limit_per_minute=rate_limit_per_minute,
                update_dataframe=True,
                max_tokens=max_tokens,
                temperature=temperature,
                client_indices=client_indices,
                show_progress=verbose
            )
        
        results['breakdowns'] = breakdown_results
        if verbose:
            print("✓ Breakdowns generated successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to generate breakdowns: {e}")
    
    # ============================================================
    # STEP 2: Generate Pairings
    # ============================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: GENERATING PAIRINGS")
        print("="*70)
    
    try:
        pairings_df = pairadigm_obj.generate_pairings(
            num_pairs_per_item=num_pairs_per_item,
            random_seed=random_seed,
            breakdowns=True,
            update_classObject=True
        )
        results['pairings'] = pairings_df
        if verbose:
            print(f"✓ Generated {len(pairings_df)} pairings")
    except Exception as e:
        raise RuntimeError(f"Failed to generate pairings: {e}")
    
    # ============================================================
    # STEP 3: Generate Pairwise LLM Annotations
    # ============================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: GENERATING PAIRWISE LLM ANNOTATIONS")
        print("="*70)
    
    try:
        annotations_df = pairadigm_obj.generate_pairwise_annotations(
            max_workers=max_workers,
            update_classObject=True,
            max_tokens=max_tokens,
            temperature=temperature,
            allow_ties=allow_ties,
            client_indices=client_indices
        )
        results['annotations'] = annotations_df
        if verbose:
            print(f"✓ Generated annotations for {len(annotations_df)} pairs")
    except Exception as e:
        raise RuntimeError(f"Failed to generate pairwise annotations: {e}")
    
    # ============================================================
    # STEP 4: Optional - Validation with Human Annotations
    # ============================================================
    if pairadigm_obj.annotated and pairadigm_obj.annotator_cols:
        if verbose:
            print("\n" + "="*70)
            print("STEP 4: VALIDATION AGAINST HUMAN ANNOTATIONS")
            print("="*70)
        
        # 4a: ALT-TEST
        if verbose:
            print("\n[4a] Running ALT-TEST...")
        try:
            alt_test_results = pairadigm_obj.alt_test(
                scoring_function='accuracy',
                epsilon=0.1,
                q_fdr=0.05,
                test_all_llms=True
            )
            results['alt_test'] = alt_test_results
            if verbose:
                print("✓ ALT-TEST completed")
        except Exception as e:
            if verbose:
                print(f"⚠ ALT-TEST failed: {e}")
            results['alt_test'] = None
        
        # 4b: Transitivity Check
        if verbose:
            print("\n[4b] Checking transitivity...")
        try:
            transitivity_results = pairadigm_obj.check_transitivity()
            results['transitivity'] = transitivity_results
            
            if verbose:
                print("✓ Transitivity check completed:")
                for annotator, (score, violations, total) in transitivity_results.items():
                    print(f"  {annotator}: {score:.3f} ({violations}/{total} violations)")
        except Exception as e:
            if verbose:
                print(f"⚠ Transitivity check failed: {e}")
            results['transitivity'] = None
        
        # 4c: Inter-Rater Reliability
        if verbose:
            print("\n[4c] Computing inter-rater reliability...")
        try:
            irr_results = pairadigm_obj.irr(
                method='auto',
                alpha_level='nominal',
                min_overlap=2
            )
            results['irr'] = irr_results
            if verbose:
                print("✓ IRR computed successfully")
        except Exception as e:
            if verbose:
                print(f"⚠ IRR computation failed: {e}")
            results['irr'] = None
        
        # 4d: Epsilon Sensitivity Analysis
        if verbose:
            print("\n[4d] Plotting epsilon sensitivity...")
        try:
            fig, epsilon_data = pairadigm_obj.plot_epsilon_sensitivity(
                epsilon_range=(-0.1, 0.25),
                epsilon_step=0.02,
                test_all_llms=True,
                show_annotations=True,
                return_data=True
            )
            results['epsilon_sensitivity_data'] = epsilon_data
            if verbose:
                print("✓ Epsilon sensitivity analysis completed")
        except Exception as e:
            if verbose:
                print(f"⚠ Epsilon sensitivity analysis failed: {e}")
            results['epsilon_sensitivity_data'] = None
    else:
        if verbose:
            print("\n" + "="*70)
            print("Note: No human annotations found. Skipping validation steps.")
            print("To enable validation, use append_human_annotations() first.")
            print("="*70)
    
    # ============================================================
    # Final Summary
    # ============================================================
    if verbose:
        print("\n" + "="*70)
        print("BUILD COMPLETE")
        print("="*70)
        print(f"Results keys: {list(results.keys())}")
        print("="*70 + "\n")
    
    return results

