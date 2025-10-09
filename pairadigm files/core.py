"""
Pairadigm: Main class for CGCoT-based pairwise annotation and validation.
"""

import pandas as pd
from typing import Optional, List, Dict, Union
from .cgcot import generate_breakdowns_parallel
from .pairing import generate_pairings_df
from .comparison import generate_pairwise_annotations, generate_pairwise_annotations_parallel
from .scoring import bradley_terry_scores, summarize_scores
from .validation import check_transitivity, compare_annotators, alternate_annotator_test
from .visualization import plot_score_distribution, plot_comparison_network
from .llm_client import LLMClient


class Pairadigm:
    """
    Main class for Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation.
    
    Supports flexible workflows:
    1. Start with raw items -> generate breakdowns -> pair -> annotate -> score
    2. Start with paired items -> annotate -> score
    3. Start with annotated pairs -> score and validate
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with items to compare
    row_id : str
        Column name for unique item identifiers
    row_name : str, optional
        Column name for item text/content
    cgcot_prompts : List[str], optional
        CGCoT prompt templates for breakdowns
    model_name : str, default='gemini-2.0-flash-exp'
        LLM model to use
    api_key : str, optional
        API key for LLM service
    target_concept : str, optional
        The concept to evaluate (e.g., "objectivity", "political bias")
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        row_id: str,
        row_name: Optional[str] = None,
        cgcot_prompts: Optional[List[str]] = None,
        model_name: str = 'gemini-2.0-flash-exp',
        api_key: Optional[str] = None,
        target_concept: Optional[str] = None
    ):
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if row_id not in data.columns:
            raise ValueError(f"Column '{row_id}' not found in DataFrame")
        
        self.data = data.copy()
        self.row_id = row_id
        self.row_name = row_name
        self.cgcot_prompts = cgcot_prompts
        self.model_name = model_name
        self.target_concept = target_concept
        
        # Initialize LLM client
        self.client = LLMClient(api_key=api_key, model_name=model_name)
        
        # Initialize result storage
        self.pairwise_df: Optional[pd.DataFrame] = None
        self.scored_df: Optional[pd.DataFrame] = None
        self.validation_results: Optional[Dict] = None
        
    def generate_breakdowns(
        self,
        max_workers: int = 8,
        rate_limit_per_minute: int = 15
    ) -> pd.DataFrame:
        """
        Generate CGCoT breakdowns for all items.
        
        Parameters
        ----------
        max_workers : int
            Number of parallel workers
        rate_limit_per_minute : int
            API rate limit
            
        Returns
        -------
        pd.DataFrame
            Data with CGCoT_Breakdown column added
        """
        if self.cgcot_prompts is None:
            raise ValueError("cgcot_prompts must be provided to generate breakdowns")
        if self.row_name is None:
            raise ValueError("row_name must be specified to generate breakdowns")
            
        print(f"Generating CGCoT breakdowns for {len(self.data)} items...")
        
        breakdowns = generate_breakdowns_parallel(
            df=self.data,
            cgcot_prompts=self.cgcot_prompts,
            model=self.model_name,
            row_name=self.row_name,
            row_id=self.row_id,
            max_workers=max_workers
        )
        
        self.data['CGCoT_Breakdown'] = self.data[self.row_id].map(breakdowns)
        print("✓ Breakdowns generated successfully")
        return self.data
    
    def generate_pairings(
        self,
        num_pairs_per_item: int = 10,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate pairwise comparisons ensuring connectivity.
        
        Parameters
        ----------
        num_pairs_per_item : int
            Minimum number of pairs per item
        random_seed : int
            For reproducibility
            
        Returns
        -------
        pd.DataFrame
            Pairings with breakdown columns
        """
        if 'CGCoT_Breakdown' not in self.data.columns:
            raise ValueError("Generate breakdowns first or provide data with CGCoT_Breakdown column")
            
        print(f"Generating pairwise comparisons ({num_pairs_per_item} pairs/item)...")
        
        self.pairwise_df = generate_pairings_df(
            df=self.data,
            row_id=self.row_id,
            num_pairs_per_item=num_pairs_per_item,
            random_seed=random_seed
        )
        
        print(f"✓ Generated {len(self.pairwise_df)} pairwise comparisons")
        return self.pairwise_df
    
    def annotate(
        self,
        pairwise_df: Optional[pd.DataFrame] = None,
        parallel: bool = True,
        max_workers: int = 8,
        rate_limit_per_minute: int = 15
    ) -> pd.DataFrame:
        """
        Annotate pairwise comparisons using LLM.
        
        Parameters
        ----------
        pairwise_df : pd.DataFrame, optional
            Pre-existing pairings to annotate. If None, uses self.pairwise_df
        parallel : bool
            Whether to use parallel processing
        max_workers : int
            Number of parallel workers
        rate_limit_per_minute : int
            API rate limit
            
        Returns
        -------
        pd.DataFrame
            Annotated pairings with decision and justification columns
        """
        if self.target_concept is None:
            raise ValueError("target_concept must be specified for annotation")
            
        if pairwise_df is not None:
            self.pairwise_df = pairwise_df
            
        if self.pairwise_df is None:
            raise ValueError("No pairings available. Generate pairings first or provide pairwise_df")
            
        print(f"Annotating {len(self.pairwise_df)} pairs for '{self.target_concept}'...")
        
        if parallel:
            self.pairwise_df = generate_pairwise_annotations_parallel(
                uuid_pairings=self.pairwise_df,
                target_concept=self.target_concept,
                model=self.model_name,
                max_workers=max_workers
            )
        else:
            self.pairwise_df = generate_pairwise_annotations(
                uuid_pairings=self.pairwise_df,
                target_concept=self.target_concept,
                model=self.model_name,
                rate_limit_per_minute=rate_limit_per_minute
            )
            
        print("✓ Annotation completed")
        return self.pairwise_df
    
    def compute_scores(
        self,
        pairwise_df: Optional[pd.DataFrame] = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Compute Bradley-Terry scores from pairwise comparisons.
        
        Parameters
        ----------
        pairwise_df : pd.DataFrame, optional
            Annotated pairings. If None, uses self.pairwise_df
        normalize : bool
            Whether to normalize scores to [0, 1]
            
        Returns
        -------
        pd.DataFrame
            Original data with Bradley_Terry_Score column added
        """
        if pairwise_df is not None:
            self.pairwise_df = pairwise_df
            
        if self.pairwise_df is None:
            raise ValueError("No annotated pairings available")
            
        print("Computing Bradley-Terry scores...")
        
        self.scored_df = bradley_terry_scores(
            original_df=self.data,
            row_id=self.row_id,
            pairwise_df=self.pairwise_df,
            normalize=normalize
        )
        
        return self.scored_df
    
    def validate(
        self,
        check_llm_transitivity: bool = True,
        human_annotations: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Validate annotation quality and check transitivity.
        
        Parameters
        ----------
        check_llm_transitivity : bool
            Whether to check LLM annotation transitivity
        human_annotations : pd.DataFrame, optional
            Human annotations for comparison (same format as pairwise_df)
            
        Returns
        -------
        Dict
            Validation results including transitivity metrics
        """
        if self.pairwise_df is None:
            raise ValueError("No annotations to validate")
            
        results = {}
        
        if check_llm_transitivity:
            print("Checking LLM annotation transitivity...")
            llm_transitivity = check_transitivity(
                pairwise_df=self.pairwise_df,
                annotator_name="LLM"
            )
            results['llm_transitivity'] = llm_transitivity
            print(f"  LLM transitivity: {llm_transitivity['transitivity_rate']:.2%}")
            
        if human_annotations is not None:
            print("Comparing LLM vs. human annotations...")
            comparison = compare_annotators(
                df1=self.pairwise_df,
                df2=human_annotations,
                annotator1_name="LLM",
                annotator2_name="Human"
            )
            results['annotator_comparison'] = comparison
            print(f"  Agreement: {comparison['agreement_rate']:.2%}")
            print(f"  Cohen's Kappa: {comparison['cohens_kappa']:.3f}")
            
            # Check human transitivity too
            human_transitivity = check_transitivity(
                pairwise_df=human_annotations,
                annotator_name="Human"
            )
            results['human_transitivity'] = human_transitivity
            print(f"  Human transitivity: {human_transitivity['transitivity_rate']:.2%}")
            
        self.validation_results = results
        return results
    
    def run_alternate_annotator_test(
        self,
        human_annotations: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict:
        """
        Run the Alternative Annotator Test to determine if LLM can replace human annotators.
        
        Parameters
        ----------
        human_annotations : pd.DataFrame
            Human annotations for comparison
        alpha : float
            Significance level for hypothesis tests
            
        Returns
        -------
        Dict
            Test results with recommendation
        """
        if self.pairwise_df is None or self.scored_df is None:
            raise ValueError("Must have both annotations and scores computed")
            
        print("Running Alternative Annotator Test...")
        
        test_results = alternate_annotator_test(
            llm_pairwise_df=self.pairwise_df,
            human_pairwise_df=human_annotations,
            llm_scores_df=self.scored_df,
            row_id=self.row_id,
            alpha=alpha
        )
        
        print("\n" + "="*60)
        print("ALTERNATIVE ANNOTATOR TEST RESULTS")
        print("="*60)
        print(f"Agreement Rate: {test_results['agreement_rate']:.2%}")
        print(f"Cohen's Kappa: {test_results['cohens_kappa']:.3f}")
        print(f"Score Correlation: {test_results['score_correlation']:.3f}")
        print(f"Recommendation: {test_results['recommendation']}")
        print("="*60 + "\n")
        
        return test_results
    
    def summarize(self, text_col: Optional[str] = None) -> Dict:
        """Summarize Bradley-Terry scores with statistics."""
        if self.scored_df is None:
            raise ValueError("Must compute scores first")
            
        text_col = text_col or self.row_name
        if text_col is None:
            raise ValueError("text_col must be specified")
            
        return summarize_scores(
            df=self.scored_df,
            text_col=text_col,
            score_col='Bradley_Terry_Score'
        )
    
    def plot_scores(self, **kwargs):
        """Plot distribution of Bradley-Terry scores."""
        if self.scored_df is None:
            raise ValueError("Must compute scores first")
        return plot_score_distribution(self.scored_df, **kwargs)
    
    def plot_network(self):
        """Plot network graph of pairwise comparisons."""
        if self.pairwise_df is None:
            raise ValueError("No pairwise comparisons to plot")
        return plot_comparison_network(self.pairwise_df)
    
    def export_results(self, filepath: str):
        """
        Export all results to an Excel file with multiple sheets.
        
        Parameters
        ----------
        filepath : str
            Output Excel file path
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            if self.scored_df is not None:
                self.scored_df.to_excel(writer, sheet_name='Scores', index=False)
            if self.pairwise_df is not None:
                self.pairwise_df.to_excel(writer, sheet_name='Pairwise_Comparisons', index=False)
            if self.validation_results is not None:
                # Convert validation results to DataFrame
                validation_summary = pd.DataFrame([{
                    'Metric': k,
                    'Value': str(v)
                } for k, v in self.validation_results.items()])
                validation_summary.to_excel(writer, sheet_name='Validation', index=False)
                
        print(f"✓ Results exported to {filepath}")