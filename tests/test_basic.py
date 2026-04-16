"""
Pairadigm v1.0 test suite.

Uses unittest.mock to avoid any real LLM API calls.
Run with:  pytest tests/ -v --tb=short
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import pairadigm


# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------

def _make_item_df(n: int = 10) -> pd.DataFrame:
    """Create a minimal unpaired item DataFrame."""
    return pd.DataFrame(
        {
            "id":   [f"item_{i}" for i in range(n)],
            "text": [f"This is text number {i}." for i in range(n)],
        }
    )


def _mock_llm_client() -> MagicMock:
    client = MagicMock()
    client.model_name = "mock-model"
    client.provider   = "ollama"
    client.generate   = MagicMock(return_value="FINAL ANSWER: Description 1\nJUSTIFICATION: test")
    return client


def _make_pairwise_df(n_pairs: int = 20) -> pd.DataFrame:
    """Create a simple pairwise DataFrame with known decisions."""
    items = [f"item_{i}" for i in range(8)]
    pairs = [
        (items[i % len(items)], items[(i + 1) % len(items)])
        for i in range(n_pairs)
    ]
    return pd.DataFrame(
        {
            "item1":       [p[0] for p in pairs],
            "item2":       [p[1] for p in pairs],
            "breakdown1":  [f"breakdown {p[0]}" for p in pairs],
            "breakdown2":  [f"breakdown {p[1]}" for p in pairs],
            "decision":    (["Text1", "Text2"] * (n_pairs // 2))[:n_pairs],
        }
    )


def _make_pairadigm_instance(n: int = 10) -> pairadigm.Pairadigm:
    """Instantiate Pairadigm with a mocked LLM client."""
    df  = _make_item_df(n)
    cli = _mock_llm_client()
    return pairadigm.Pairadigm(
        data=df,
        item_id_name="id",
        text_name="text",
        cgcot_prompts=["Analyse {text}. Previous: {previous_answers}"],
        target_concept="clarity",
        llm_clients=cli,
    )


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

class TestPackageMetadata:
    def test_version_is_string(self):
        assert isinstance(pairadigm.__version__, str)

    def test_version_is_1_0_0(self):
        assert pairadigm.__version__ == "1.0.0"


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class TestLLMClient:
    def test_provider_inference_gemini(self):
        with patch.object(pairadigm.LLMClient, "_initialize_client", return_value=MagicMock()), \
             patch.object(pairadigm.LLMClient, "_get_api_key",       return_value="key"):
            c = pairadigm.LLMClient(model_name="gemini-2.0-flash-exp")
        assert c.provider == "google"

    def test_provider_inference_gpt(self):
        with patch.object(pairadigm.LLMClient, "_initialize_client", return_value=MagicMock()), \
             patch.object(pairadigm.LLMClient, "_get_api_key",       return_value="key"):
            c = pairadigm.LLMClient(model_name="gpt-4o")
        assert c.provider == "openai"

    def test_provider_inference_claude(self):
        with patch.object(pairadigm.LLMClient, "_initialize_client", return_value=MagicMock()), \
             patch.object(pairadigm.LLMClient, "_get_api_key",       return_value="key"):
            c = pairadigm.LLMClient(model_name="claude-sonnet-4")
        assert c.provider == "anthropic"

    def test_unknown_model_defaults_to_ollama(self):
        """Unknown model names should default to the ollama provider (local)."""
        c = pairadigm.LLMClient(model_name="my-totally-unknown-model-xyz")
        assert c.provider == "ollama"

    def test_huggingface_slash_pattern(self):
        c = pairadigm.LLMClient(model_name="meta-llama/Llama-3.3-70B-Instruct")
        assert c.provider == "huggingface"


# ---------------------------------------------------------------------------
# Pairadigm init
# ---------------------------------------------------------------------------

class TestPairadigmInit:
    def test_basic_unpaired_init(self):
        obj = _make_pairadigm_instance()
        assert obj.item_id_name == "id"
        assert obj.text_name == "text"
        assert obj.paired is False
        assert obj.target_concept == "clarity"

    def test_cgcot_prompts_none_raises(self):
        df  = _make_item_df()
        cli = _mock_llm_client()
        with pytest.raises(ValueError, match="cgcot_prompts"):
            pairadigm.Pairadigm(
                data=df, item_id_name="id", text_name="text",
                cgcot_prompts=None, target_concept="X", llm_clients=cli,
            )

    def test_target_concept_none_raises(self):
        df  = _make_item_df()
        cli = _mock_llm_client()
        with pytest.raises(ValueError, match="target_concept"):
            pairadigm.Pairadigm(
                data=df, item_id_name="id", text_name="text",
                cgcot_prompts=["Test {text}"], target_concept=None, llm_clients=cli,
            )

    def test_missing_id_column_raises(self):
        df  = pd.DataFrame({"wrong": ["a"], "text": ["t"]})
        cli = _mock_llm_client()
        with pytest.raises(ValueError, match="'id' not found"):
            pairadigm.Pairadigm(
                data=df, item_id_name="id", text_name="text",
                cgcot_prompts=["Test {text}"], target_concept="X", llm_clients=cli,
            )

    def test_client_property(self):
        obj = _make_pairadigm_instance()
        assert obj.client is obj.clients[0]

    def test_column_renames_stored(self):
        """column_renames dict should be present (may be empty for unpaired)."""
        obj = _make_pairadigm_instance()
        assert isinstance(obj.column_renames, dict)

    def test_paired_init_renames_id_cols(self):
        df = pd.DataFrame({
            "resp_a": ["x", "y"], "resp_b": ["y", "z"],
            "text_a": ["ta", "tb"], "text_b": ["tb", "tc"],
        })
        cli = _mock_llm_client()
        obj = pairadigm.Pairadigm(
            data=df, item_id_name=None, item_id_cols=["resp_a", "resp_b"],
            item_text_cols=["text_a", "text_b"],
            paired=True, cgcot_prompts=["Test {text}"], target_concept="X",
            llm_clients=cli,
        )
        assert "item1" in obj.data.columns
        assert "item2" in obj.data.columns
        assert obj.column_renames.get("resp_a") == "item1"


# ---------------------------------------------------------------------------
# generate_pairings
# ---------------------------------------------------------------------------

class TestGeneratePairings:
    def test_returns_dataframe(self):
        obj = _make_pairadigm_instance(n=10)
        result = obj.generate_pairings(num_pairs_per_item=3, random_seed=0)
        assert isinstance(result, pd.DataFrame)
        assert "item1" in result.columns
        assert "item2" in result.columns

    def test_update_classObject(self):
        obj = _make_pairadigm_instance(n=10)
        obj.generate_pairings(update_classObject=True)
        assert obj.pairwise_df is not None

    def test_make_splits_adds_split_columns(self):
        obj = _make_pairadigm_instance(n=20)
        result = obj.generate_pairings(make_splits=True, num_pairs_per_item=3)
        assert "item1_split" in result.columns
        assert "item2_split" in result.columns

    def test_no_self_pairs(self):
        obj = _make_pairadigm_instance(n=10)
        result = obj.generate_pairings()
        assert (result["item1"] != result["item2"]).all()


# ---------------------------------------------------------------------------
# score_items (with mock pairwise_df)
# ---------------------------------------------------------------------------

class TestScoreItems:
    def setup_method(self):
        self.obj = _make_pairadigm_instance(n=8)
        self.obj.pairwise_df = _make_pairwise_df(n_pairs=20)

    def test_returns_dataframe(self):
        result = self.obj.score_items(summarize=False)
        assert isinstance(result, pd.DataFrame)

    def test_score_column_present(self):
        result = self.obj.score_items(summarize=False)
        score_cols = [c for c in result.columns if "Score" in c]
        assert len(score_cols) >= 1

    def test_scores_in_zero_one_range(self):
        result = self.obj.score_items(summarize=False)
        col = [c for c in result.columns if "Score" in c][0]
        assert result[col].between(0.0, 1.0).all()

    def test_updates_scored_df(self):
        self.obj.score_items(update_classObject=True, summarize=False)
        assert self.obj.scored_df is not None


# ---------------------------------------------------------------------------
# append_human_annotations (fix 1f vectorised)
# ---------------------------------------------------------------------------

class TestAppendHumanAnnotations:
    def setup_method(self):
        self.obj = _make_pairadigm_instance(n=8)
        self.obj.pairwise_df = _make_pairwise_df(n_pairs=10)
        # Simple annotation: item_0 > item_1 for all pairs where they appear
        self.anns = pd.DataFrame({
            "item1":    self.obj.pairwise_df["item1"].tolist(),
            "item2":    self.obj.pairwise_df["item2"].tolist(),
            "ann_col":  ["Text1"] * 10,
        })

    def test_adds_annotator_column(self):
        self.obj.append_human_annotations(
            self.anns, annotator_names="human1", decision_cols="ann_col"
        )
        assert "human1" in self.obj.pairwise_df.columns

    def test_sets_annotated_flag(self):
        self.obj.append_human_annotations(
            self.anns, annotator_names="human1", decision_cols="ann_col"
        )
        assert self.obj.annotated is True

    def test_normalises_numeric_decisions(self):
        anns_numeric = self.anns.copy()
        anns_numeric["ann_col"] = [0, 1] * 5  # 0 = Text1, 1 = Text2
        self.obj.append_human_annotations(
            anns_numeric, annotator_names="human_num", decision_cols="ann_col"
        )
        vals = self.obj.pairwise_df["human_num"].dropna().unique().tolist()
        # Should be normalised to string 'Text1'/'Text2'
        assert all(v in ("Text1", "Text2") for v in vals)

    def test_overwrite_false_raises_on_duplicate(self):
        self.obj.append_human_annotations(
            self.anns, annotator_names="human1", decision_cols="ann_col"
        )
        with pytest.raises(ValueError, match="already exists"):
            self.obj.append_human_annotations(
                self.anns, annotator_names="human1",
                decision_cols="ann_col", overwrite=False,
            )


# ---------------------------------------------------------------------------
# check_transitivity (fix 1d dedup)
# ---------------------------------------------------------------------------

class TestCheckTransitivity:
    def test_perfect_transitivity(self):
        """A→B, B→C, A→C — no violations."""
        df = pd.DataFrame({
            "item1":    ["A", "B", "A"],
            "item2":    ["B", "C", "C"],
            "decision": ["Text1", "Text1", "Text1"],
        })
        obj = _make_pairadigm_instance(n=3)
        obj.pairwise_df = df
        res = obj.check_transitivity()
        score, violations, total = res["decision"]
        assert violations == 0
        assert score == 1.0

    def test_violation_detected(self):
        """A→B, B→C, but C→A — cycle = violation."""
        df = pd.DataFrame({
            "item1":    ["A", "B", "C"],
            "item2":    ["B", "C", "A"],
            "decision": ["Text1", "Text1", "Text1"],
        })
        obj = _make_pairadigm_instance(n=3)
        obj.pairwise_df = df
        res = obj.check_transitivity()
        _, violations, _ = res["decision"]
        assert violations > 0


# ---------------------------------------------------------------------------
# Save / Load round-trip (new structured format)
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def _make_obj_with_data(self) -> pairadigm.Pairadigm:
        obj = _make_pairadigm_instance(n=6)
        obj.pairwise_df = _make_pairwise_df(n_pairs=10)
        obj.scored_df   = obj.data.copy()
        obj.scored_df["Bradley_Terry_Score"] = np.linspace(0, 1, len(obj.data))
        return obj

    def test_save_creates_files(self):
        obj = self._make_obj_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            obj.save(tmpdir)
            assert (os.path.join(tmpdir, "metadata.json")) in [
                os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
            ] or os.path.exists(os.path.join(tmpdir, "metadata.json"))
            assert os.path.exists(os.path.join(tmpdir, "data.parquet"))
            assert os.path.exists(os.path.join(tmpdir, "pairwise_df.parquet"))
            assert os.path.exists(os.path.join(tmpdir, "scored_df.parquet"))

    def test_metadata_json_has_version(self):
        obj = self._make_obj_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            obj.save(tmpdir)
            meta = json.loads(
                open(os.path.join(tmpdir, "metadata.json"), encoding="utf-8").read()
            )
        assert meta["version"] == "1.0"
        assert meta["target_concept"] == "clarity"

    def test_load_round_trip_preserves_data(self):
        obj = self._make_obj_with_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            obj.save(tmpdir)
            loaded = pairadigm.Pairadigm.load(tmpdir)
        assert loaded.target_concept == obj.target_concept
        assert list(loaded.data.columns) == list(obj.data.columns)
        assert len(loaded.pairwise_df) == len(obj.pairwise_df)
        assert len(loaded.scored_df) == len(obj.scored_df)

    def test_load_pickle_raises_value_error(self):
        from pairadigm.persistence import load_pairadigm
        with pytest.raises(ValueError, match="Legacy pickle"):
            load_pairadigm("some_old_file.pkl")


# ---------------------------------------------------------------------------
# dawid_skene EM — M-step guard (fix 1e)
# ---------------------------------------------------------------------------

class TestDawidSkeneEM:
    def test_missing_annotations_do_not_crash(self):
        """EM should handle -1 sentinel (missing) without dividing by zero."""
        from pairadigm.validation import _dawid_skene_em
        # n_instances=5, n_annotators=3; two missing values
        labels = np.array([
            [0, 1, 0],
            [1, -1, 1],  # annotator 1 missing for instance 1
            [0, 0, -1],  # annotator 2 missing for instance 2
            [1, 1, 0],
            [0, 0, 1],
        ])
        probs, reliability, conv_iter = _dawid_skene_em(
            labels, num_classes=2, max_iter=10, tol=1e-4, random_seed=42
        )
        assert probs.shape == (5, 2)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isnan(reliability))


# ---------------------------------------------------------------------------
# plot_comparison_network — centrality guard (fix 1b)
# ---------------------------------------------------------------------------

class TestPlotComparisonNetwork:
    def test_invalid_centrality_raises_value_error(self):
        """Should raise ValueError BEFORE any dict lookup."""
        obj = _make_pairadigm_instance(n=6)
        obj.pairwise_df = _make_pairwise_df(n_pairs=10)
        with pytest.raises(ValueError, match="Unknown centrality measure"):
            obj.plot_comparison_network(
                centrality_measure="not_a_real_metric", return_fig=True
            )


# ---------------------------------------------------------------------------
# get_score_col_name helper (fix 7b)
# ---------------------------------------------------------------------------

class TestGetScoreColName:
    def test_default(self):
        obj = _make_pairadigm_instance()
        assert obj.get_score_col_name() == "Bradley_Terry_Score"

    def test_with_split(self):
        obj = _make_pairadigm_instance()
        assert obj.get_score_col_name(split="full") == "Bradley_Terry_Score_full"

    def test_multi_model_decision_col(self):
        obj = _make_pairadigm_instance()
        col = obj.get_score_col_name(decision_col="decision_gpt-4o")
        assert col == "Bradley_Terry_Score_gpt-4o"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
