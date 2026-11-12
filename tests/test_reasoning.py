"""Unit tests. Run with:  python -m unittest discover -s tests -v"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from algorithms.algorithm1_paraconsistent import algorithm1_paraconsistent_reasoning  # noqa: E402
from algorithms.algorithm2_learning import algorithm2_learning_reasoning  # noqa: E402
from algorithms.paraconsistent import Conclusion, ParaconsistentKB, decide, naive_maximal_consistent_subsets  # noqa: E402
from experiments.evaluation_metrics import evaluate  # noqa: E402
from kg.data import count_conflicts, inject_noise  # noqa: E402
from kg.llm import LLMBackend  # noqa: E402
from kg.llm_extraction import LLMExtractor, RuleExtractor, evaluate_extraction, gold_triples, make_documents  # noqa: E402
from kg.ontology import Ontology, solar_ontology  # noqa: E402

P, PROD = "hasSolarPotential", "produces"


class TableScorer:
    def __init__(self, table):
        self.table = table

    def prob(self, triples):
        import numpy as np
        return np.array([self.table.get(t, 0.5) for t in triples])

    def predict(self, keys, pool):
        return {}


class TestOntology(unittest.TestCase):
    def test_range_and_domain(self):
        o = solar_ontology((0, 255))
        self.assertIsNone(o.violation("location_1_2", P, "231.0"))
        self.assertIsNotNone(o.violation("location_1_2", P, "300.0"))
        self.assertIsNotNone(o.violation("location_1_2", PROD, "150.0"))
        self.assertIsNone(o.violation("location_1_2", PROD, "solar_energy"))
        self.assertIsNotNone(o.violation("solar_energy", P, "10.0"))

    def test_phi(self):
        self.assertIn(("solar_energy", "type", "EnergySource"), solar_ontology().to_triples())


class TestFragments(unittest.TestCase):
    def test_local_decomposition_equals_brute_force(self):
        """Proposition 1: credulous/skeptical entailment from per-key clusters
        coincides with brute-force enumeration of MC(K)."""
        rng = random.Random(0)
        onto = Ontology(functional={"f": 1.0, "g": 1.0})
        for _ in range(40):
            triples = [(f"e{rng.randint(0, 2)}", rng.choice(["f", "g", "n"]), f"v{rng.randint(0, 2)}")
                       for _ in range(rng.randint(1, 9))]
            mcs = naive_maximal_consistent_subsets(triples, onto)
            kb = ParaconsistentKB(triples, onto)
            for t in set(triples):
                self.assertEqual(kb.entails(*t, mode="credulous"), any(t in F for F in mcs))
                self.assertEqual(kb.entails(*t, mode="skeptical"), all(t in F for F in mcs))

    def test_range_violations_excluded(self):
        onto = solar_ontology((0, 255))
        triples = [("location_1_1", P, "200.0"), ("location_1_1", P, "280.0")]
        kb = ParaconsistentKB(triples, onto)
        self.assertEqual(kb.credulous(("location_1_1", P)), {"200.0"})
        self.assertFalse(kb.is_conflict(("location_1_1", P)))

    def test_decide(self):
        self.assertEqual(decide({"a": 0.9, "b": 0.2}, 0.1, True), ["a"])
        self.assertEqual(sorted(decide({"a": 0.5, "b": 0.45}, 0.1, True)), ["a", "b"])
        self.assertEqual(sorted(decide({"a": 0.9, "b": 0.2}, 0.1, False)), ["a", "b"])


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.onto = solar_ontology((0, 255))
        self.triples = [("location_1_1", P, "255.0"), ("location_1_1", P, "120.0"),
                        ("location_2_2", P, "240.0"), ("location_2_2", PROD, "solar_energy"),
                        ("location_2_2", PROD, "180.0")]
        self.kb = ParaconsistentKB(self.triples, self.onto)
        self.keys = [("location_1_1", P), ("location_2_2", P), ("location_2_2", PROD)]

    def test_algorithm1_keeps_conflict(self):
        out = algorithm1_paraconsistent_reasoning(self.kb, self.keys)
        self.assertEqual(sorted(out[("location_1_1", P)].tails), ["120.0", "255.0"])
        self.assertEqual(out[("location_2_2", PROD)].tails, ["solar_energy"])   # range axiom

    def test_algorithm2_resolves_with_scores(self):
        scorer = TableScorer({("location_1_1", P, "255.0"): 0.9, ("location_1_1", P, "120.0"): 0.1})
        out = algorithm2_learning_reasoning(self.kb, scorer, self.keys, lam=0.5, delta=0.1)
        self.assertEqual(out[("location_1_1", P)].tails, ["255.0"])

    def test_metrics(self):
        truth = {("location_1_1", P): {"255.0"}, ("location_2_2", P): {"240.0"},
                 ("location_2_2", PROD): {"solar_energy"}}
        asserted = {("location_1_1", P): {"255.0": 1, "120.0": 1}, ("location_2_2", P): {"240.0": 1},
                    ("location_2_2", PROD): {"solar_energy": 1, "180.0": 1}}
        out = {("location_1_1", P): Conclusion(["255.0", "120.0"], {"255.0": 1, "120.0": 1}),
               ("location_2_2", P): Conclusion(["240.0"], {"240.0": 2}),
               ("location_2_2", PROD): Conclusion(["solar_energy"], {"solar_energy": 1})}
        m = evaluate(out, truth, asserted, self.onto)
        self.assertAlmostEqual(m["correctness"], 2 / 3)
        self.assertAlmostEqual(m["accuracy"], 3 / 4)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["support_ratio"], 5 / 4)
        self.assertEqual(m["conflicts"], 2)
        self.assertAlmostEqual(m["conflict_mitigation"], 0.5)
        self.assertAlmostEqual(m["conflict_resolution_acc"], 0.5)


class TestData(unittest.TestCase):
    def test_noise_is_seeded_and_labelled(self):
        df = pd.DataFrame({"head": [f"location_{i}_0" for i in range(50)], "relation": P,
                           "tail": "255.0", "split": "train"})
        a = inject_noise(df, 10, 100, 300, seed=3)
        b = inject_noise(df, 10, 100, 300, seed=3)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 60)
        self.assertEqual(int((a["label"] == 0).sum()), int((a["tail"] != "255.0").sum()))
        self.assertGreater(count_conflicts(a), 0)


class FakeBackend(LLMBackend):
    """Pretends to be an LLM: answers extraction prompts with the rule extractor."""
    name = "fake"

    def _raw(self, system, user, schema):
        import json
        import re
        docs = [{"id": int(m.group(1)), "text": m.group(2)} for m in re.finditer(r"\[(\d+)\] (.*)", user)]
        rows = RuleExtractor().extract(docs)
        triples = [{"doc_id": int(d), "head": h, "relation": r, "tail": t, "confidence": 0.9}
                   for d, h, r, t, _ in rows.itertuples(index=False)]
        return json.dumps({"triples": triples}), {"input_tokens": 10, "output_tokens": 5}


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.samples = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [9, 8, 7, 6, 5],
                                     "pv_potential": [255.0, 231.0, 198.0, 240.0, 12.0]})

    def test_rule_extractor_recovers_gold(self):
        docs = make_documents(self.samples, seed=1)
        m = evaluate_extraction(RuleExtractor().extract(docs), gold_triples(self.samples))
        self.assertEqual((m["precision"], m["recall"]), (1.0, 1.0))

    def test_llm_extractor_interface(self):
        backend = FakeBackend(model="fake")
        pred = LLMExtractor(backend, batch_size=2).extract(make_documents(self.samples))
        self.assertEqual(evaluate_extraction(pred, gold_triples(self.samples))["f1"], 1.0)
        self.assertEqual(backend.usage.calls, 3)
        self.assertTrue((pred["weight"] == 0.9).all())


class TestModels(unittest.TestCase):
    def _toy(self):
        rows = []
        for i in range(12):
            rows.append((f"e{i}", "r1", f"e{(i + 1) % 12}", "train" if i < 8 else ("val" if i < 10 else "test")))
            rows.append((f"e{i}", "r2", f"e{(i * 5) % 12}", "train"))
        df = pd.DataFrame(rows, columns=["head", "relation", "tail", "split"])
        df["label"], df["source"] = 1, "data"
        from kg.data import KGDataset
        return KGDataset(df)

    def test_score_all_matches_forward(self):
        import torch
        from models.kge import Decoder
        torch.manual_seed(0)
        E, x, r = torch.randn(7, 8), torch.randn(3, 8), torch.tensor([0, 1, 1])
        for kind in ("bilinear", "distmult", "complex", "transe", "rotate"):
            dec = Decoder(kind, 2, 8)
            for side in ("tail", "head"):
                fast = dec.score_all(x, r, E, side)
                xs, es = x.unsqueeze(1).expand(-1, 7, -1), E.unsqueeze(0).expand(3, -1, -1)
                slow = dec(xs, r, es) if side == "tail" else dec(es, r, xs)
                self.assertTrue(torch.allclose(fast, slow, atol=1e-5), (kind, side))

    def test_kvsall_training_and_ranking(self):
        from config import TrainConfig
        from models.kge import KGModel, rank_metrics, train_model
        ds = self._toy()
        cfg = TrainConfig(dim=16, epochs=3, objective="kvsall", select_metric="mrr", device="cpu")
        for enc in ("id", "rgcn"):
            graph = ds.build_graph() if enc == "rgcn" else None
            model = KGModel(enc, "distmult", ds, graph, cfg)
            info = train_model(model, ds, ds.split("train"), cfg, seed=0)
            self.assertEqual(info["select_metric"], "mrr")
            m = rank_metrics(model, ds, ds.split("test"), ds.truth)
            self.assertTrue(0 < m["MRR"] <= 1 and m["Hits@1"] <= m["Hits@10"])


if __name__ == "__main__":
    unittest.main()
