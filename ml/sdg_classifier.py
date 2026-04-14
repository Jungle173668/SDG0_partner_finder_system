"""
SDG multi-label classifier.

Predicts which UN Sustainable Development Goals (SDGs) a company contributes to,
based on its text description. Supports 5 methods, all tracked via MLflow + Dagshub.

Methods
-------
1. zero_shot   Cosine similarity between company embedding and SDG keyword embeddings.
               No training required. Model: all-MiniLM-L6-v2. F1=0.238.

2. logreg      OneVsRest LogisticRegression trained on 384-dim pgvector embeddings
               + synthetic SDG keyword embeddings. F1=0.474 (overfit — train==eval set).

3. setfit      Two-stage few-shot fine-tuning: contrastive learning on all-MiniLM-L6-v2
               + logistic regression head. Model saved to ml/models/sdg_setfit_v2. F1=0.308.

4. nli         Natural Language Inference via cross-encoder/nli-deberta-v3-small.
               High recall (0.93) but too slow for production (17 inference calls/company).
               F1=0.347.

5. llm         Few-shot prompting via Groq (llama-3.1-8b-instant). 4 labeled examples
               in prompt. Best method: F1=0.810. ✅ Current production method.

Production
----------
LLM few-shot is the current production method (registered as sdg-classifier v1 in
MLflow Model Registry). All 492 companies backfilled with predicted_sdg_tags.

Usage
-----
# Train models (one-time):
    python -m ml.sdg_classifier train            # LogReg
    python -m ml.sdg_classifier train_setfit     # SetFit

# Backfill all businesses with LLM predictions:
    python -m ml.sdg_classifier backfill_llm
    python -m ml.sdg_classifier backfill_llm --overwrite   # re-run all

# Evaluate a method against ground-truth sdg_tags (20 labeled companies):
    python -m ml.sdg_classifier evaluate --method zero_shot
    python -m ml.sdg_classifier evaluate --method logreg
    python -m ml.sdg_classifier evaluate --method setfit
    python -m ml.sdg_classifier evaluate --method nli
    python -m ml.sdg_classifier evaluate --method llm

# Threshold sweep across [0.2, 0.3, 0.4, 0.5, 0.6]:
    python -m ml.sdg_classifier sweep --method llm

# DB coverage stats:
    python -m ml.sdg_classifier stats
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDG label definitions
# IMPORTANT: these names MUST match what's stored in sdg_tags metadata exactly
# (verified against ChromaDB — Title Case, "And" capitalised, no commas in names)
# ---------------------------------------------------------------------------
SDG_LABELS: list[str] = [
    "No Poverty",                               # SDG 1
    "Zero Hunger",                              # SDG 2
    "Good Health And Well-Being",               # SDG 3
    "Quality Education",                        # SDG 4
    "Gender Equality",                          # SDG 5
    "Clean Water And Sanitation",               # SDG 6
    "Affordable And Clean Energy",              # SDG 7
    "Decent Work And Economic Growth",          # SDG 8
    "Industry Innovation And Infrastructure",  # SDG 9 (SDGZero naming)
    "Reduced Inequalities",                     # SDG 10
    "Sustainable Cities And Communities",       # SDG 11
    "Responsible Consumption And Production",   # SDG 12
    "Climate Action",                           # SDG 13
    "Life Below Water",                         # SDG 14
    "Life On Land",                             # SDG 15
    "Peace Justice And Strong Institutions",    # SDG 16
    "Partnerships For The Goals",               # SDG 17
]

# Official UN SDG descriptions — used to generate synthetic zero-shot training data.
# Richer descriptions = better training signal.
SDG_DESCRIPTIONS: dict[str, str] = {
    "No Poverty": (
        "End poverty in all its forms everywhere. Providing economic resources, "
        "social protection systems, basic services and safety nets for the poor and vulnerable. "
        "Microfinance, financial inclusion, anti-poverty programmes, welfare, "
        "affordable housing, income support, unemployment benefits."
    ),
    "Zero Hunger": (
        "End hunger, achieve food security and improved nutrition, promote sustainable agriculture. "
        "Food banks, nutrition programmes, sustainable farming, agri-tech, food waste reduction, "
        "smallholder farmers, food supply chains, agricultural development."
    ),
    "Good Health And Well-Being": (
        "Ensure healthy lives and promote well-being for all at all ages. "
        "Healthcare services, mental health, medical technology, pharmaceuticals, "
        "preventive health, fitness, wellness, disease prevention, NHS, "
        "public health, telemedicine, health education."
    ),
    "Quality Education": (
        "Ensure inclusive and equitable quality education and promote lifelong learning. "
        "Schools, universities, vocational training, e-learning, EdTech, tutoring, "
        "skills development, workforce training, STEM education, literacy, scholarships."
    ),
    "Gender Equality": (
        "Achieve gender equality and empower all women and girls. "
        "Women's rights, diversity and inclusion, equal pay, female leadership, "
        "domestic violence prevention, girls' education, women in STEM, "
        "maternity support, feminist organisations."
    ),
    "Clean Water And Sanitation": (
        "Ensure availability and sustainable management of water and sanitation for all. "
        "Water treatment, sanitation systems, clean drinking water, wastewater management, "
        "water conservation, irrigation technology, flood management, WASH programmes."
    ),
    "Affordable And Clean Energy": (
        "Ensure access to affordable, reliable, sustainable and modern energy for all. "
        "Renewable energy, solar power, wind energy, energy efficiency, battery storage, "
        "electric vehicles, green electricity, carbon-neutral energy, heat pumps, "
        "smart grids, energy transition."
    ),
    "Decent Work And Economic Growth": (
        "Promote sustained, inclusive and sustainable economic growth, full and productive "
        "employment and decent work for all. Job creation, fair wages, workers' rights, "
        "entrepreneurship, SME support, economic development, employment services, HR, "
        "recruitment, business growth, professional development."
    ),
    "Industry Innovation And Infrastructure": (
        "Build resilient infrastructure, promote inclusive and sustainable industrialisation "
        "and foster innovation. Make cities inclusive, safe, resilient and sustainable. "
        "R&D, deep tech, manufacturing, engineering, smart cities, urban planning, "
        "transport infrastructure, broadband connectivity, innovation hubs, startups."
    ),
    "Reduced Inequalities": (
        "Reduce inequality within and among countries. "
        "Social justice, racial equality, disability inclusion, migrant support, "
        "income redistribution, community development, underserved populations, "
        "anti-discrimination, socioeconomic mobility, diversity equity inclusion."
    ),
    "Sustainable Cities And Communities": (
        "Make cities and human settlements inclusive, safe, resilient and sustainable. "
        "Urban sustainability, green buildings, sustainable architecture, public transport, "
        "waste management, city planning, housing, community spaces, net zero buildings."
    ),
    "Responsible Consumption And Production": (
        "Ensure sustainable consumption and production patterns. "
        "Circular economy, recycling, waste reduction, sustainable supply chains, "
        "eco-friendly products, green procurement, life cycle assessment, "
        "sustainable packaging, upcycling, zero waste."
    ),
    "Climate Action": (
        "Take urgent action to combat climate change and its impacts. "
        "Carbon emissions reduction, net zero, carbon footprint, climate tech, "
        "sustainability consulting, ESG reporting, carbon offsetting, "
        "greenhouse gas measurement, decarbonisation, Scope 1 2 3 emissions."
    ),
    "Life Below Water": (
        "Conserve and sustainably use the oceans, seas and marine resources. "
        "Marine conservation, ocean plastics, sustainable fishing, coral reef protection, "
        "blue economy, maritime sustainability, water pollution prevention."
    ),
    "Life On Land": (
        "Protect and restore terrestrial ecosystems, sustainably manage forests, "
        "combat desertification, halt biodiversity loss. Reforestation, nature conservation, "
        "biodiversity, wildlife protection, sustainable land use, agroforestry, "
        "rewilding, ecological restoration."
    ),
    "Peace Justice And Strong Institutions": (
        "Promote peaceful and inclusive societies, provide access to justice for all, "
        "build effective and accountable institutions. Legal services, governance, "
        "anti-corruption, human rights, conflict resolution, civic engagement, "
        "public administration, rule of law, transparency."
    ),
    "Partnerships For The Goals": (
        "Strengthen means of implementation and revitalise global partnership for sustainable development. "
        "Cross-sector collaboration, impact investment, public-private partnerships, "
        "B Corp, social enterprise, stakeholder engagement, SDG alignment, "
        "sustainability networks, corporate social responsibility."
    ),
}

# Where to save the trained model
MODEL_DIR = Path(__file__).parent / "models" / "sdg_setfit"
THRESHOLD = 0.5   # probability threshold for multi-label classification (SetFit)
COSINE_THRESHOLD = 0.20  # cosine similarity threshold for zero-shot prediction

# Short keyword-focused SDG descriptions used for zero-shot cosine similarity.
# Deliberately concise so the embedding matches business text better.
SDG_KEYWORDS: dict[str, str] = {
    "No Poverty": "poverty microfinance financial inclusion welfare housing income support benefits",
    "Zero Hunger": "hunger food security agriculture food banks nutrition food supply farming",
    "Good Health And Well-Being": "healthcare mental health medical pharmaceutical wellness disease prevention NHS telemedicine",
    "Quality Education": "education schools universities training e-learning EdTech skills STEM literacy tutoring",
    "Gender Equality": "women empowerment gender diversity equal pay female leadership domestic violence girls",
    "Clean Water And Sanitation": "water treatment sanitation drinking water wastewater conservation irrigation flood",
    "Affordable And Clean Energy": "renewable energy solar wind energy efficiency electric vehicles green electricity carbon neutral heat pumps",
    "Decent Work And Economic Growth": "employment job creation entrepreneurship SME business growth HR recruitment professional development workers rights",
    "Industry Innovation And Infrastructure": "R&D technology manufacturing engineering smart cities infrastructure innovation startups deep tech",
    "Reduced Inequalities": "social justice racial equality disability inclusion migrant anti-discrimination diversity equity underserved",
    "Sustainable Cities And Communities": "urban sustainability green buildings public transport waste management net zero housing city planning",
    "Responsible Consumption And Production": "circular economy recycling waste reduction sustainable supply chain eco-friendly packaging zero waste upcycling",
    "Climate Action": "carbon emissions net zero climate ESG sustainability decarbonisation greenhouse gas carbon footprint",
    "Life Below Water": "marine ocean plastics sustainable fishing coral reef blue economy maritime water pollution",
    "Life On Land": "reforestation biodiversity wildlife conservation land use agroforestry rewilding ecological restoration",
    "Peace Justice And Strong Institutions": "legal governance anti-corruption human rights conflict resolution civic public administration transparency",
    "Partnerships For The Goals": "collaboration impact investment B Corp social enterprise stakeholder SDG alignment CSR partnership",
}


# ---------------------------------------------------------------------------
# Core functions (DB-agnostic)
# ---------------------------------------------------------------------------

def train(
    save_path: Optional[Path] = None,
) -> dict:
    """
    Train a multi-label SDG classifier using pre-computed PostgreSQL embeddings.

    Strategy:
      1. Pull stored 384-dim embeddings from ChromaDB for businesses that have
         real sdg_tags — no transformer inference needed, takes milliseconds.
      2. Encode 17 short SDG keyword descriptions (~0.1s) as synthetic fallback
         for any SDG label that has no real examples.
      3. Fit an sklearn OneVsRest LogisticRegression (~1s).
      4. Save to MODEL_DIR/logreg.joblib.

    Total training time: a few seconds on any machine.

    PostgreSQL migration note:
        Step 1 changes to a SQL query:
            SELECT embedding, sdg_tags FROM businesses WHERE sdg_tags IS NOT NULL
        Everything else (steps 2-4) stays the same.

    Args:
        save_path:  Where to save the model. Defaults to MODEL_DIR.

    Returns:
        Loaded model dict (same format as load_model()).
    """
    import joblib
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    save_path = Path(save_path) if save_path else MODEL_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    label2id = {label: i for i, label in enumerate(SDG_LABELS)}
    X, y = [], []

    # ------------------------------------------------------------------
    # Step 1: Real labeled examples — use stored PostgreSQL embeddings
    # No transformer call needed; embeddings are already computed.
    # ------------------------------------------------------------------
    from db.pg_store import PGStore
    store = PGStore()
    with store._cursor(dict_rows=True) as cur:
        cur.execute(
            "SELECT embedding, sdg_tags FROM businesses "
            "WHERE sdg_tags IS NOT NULL AND sdg_tags != ''"
        )
        pg_rows = list(cur.fetchall())

    real_count = 0
    for row in pg_rows:
        sdg_str = (row["sdg_tags"] or "").strip()
        if not sdg_str:
            continue
        raw_tags = [t.strip() for t in sdg_str.split(",")]
        known_tags = [t for t in raw_tags if t in label2id]
        if not known_tags:
            continue

        vec = [0] * len(SDG_LABELS)
        for tag in known_tags:
            vec[label2id[tag]] = 1
        X.append(list(row["embedding"]))
        y.append(vec)
        real_count += 1

    print(f"Real labeled examples from PostgreSQL : {real_count}")

    # ------------------------------------------------------------------
    # Step 2: Synthetic examples — encode 17 SDG keyword descriptions.
    # Only 17 short strings, takes ~0.1s. Covers labels with no real data.
    # ------------------------------------------------------------------
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    synth_texts = [SDG_KEYWORDS[label] for label in SDG_LABELS]
    synth_embs = encoder.encode(synth_texts, normalize_embeddings=True)

    for i, _ in enumerate(SDG_LABELS):
        vec = [0] * len(SDG_LABELS)
        vec[i] = 1
        X.append(synth_embs[i])
        y.append(vec)

    print(f"Synthetic examples (SDG keywords)   : {len(SDG_LABELS)}")
    print(f"Total training examples              : {len(X)}")

    X = np.array(X)   # (N, 384)
    y = np.array(y)   # (N, 17)

    # ------------------------------------------------------------------
    # Step 3: Fit OneVsRest LogisticRegression
    # ------------------------------------------------------------------
    print("Training LogisticRegression... ", end="", flush=True)
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    )
    clf.fit(X, y)
    print("done.")

    # ------------------------------------------------------------------
    # Step 4: Save
    # ------------------------------------------------------------------
    model_obj = {"classifier": clf, "labels": SDG_LABELS}
    out_path = save_path / "logreg.joblib"
    joblib.dump(model_obj, out_path)
    print(f"Model saved to {out_path}")
    return model_obj


def load_model(model_path: Optional[Path] = None) -> dict:
    """
    Load the trained LogisticRegression SDG classifier.

    Returns a dict: {"classifier": OneVsRestClassifier, "labels": list[str]}
    """
    import joblib

    model_path = Path(model_path) if model_path else MODEL_DIR
    logreg_path = model_path / "logreg.joblib"
    if not logreg_path.exists():
        raise FileNotFoundError(
            f"No trained model at {logreg_path}. "
            "Run: python -m ml.sdg_classifier train"
        )
    logger.info(f"Loading model from {logreg_path}")
    return joblib.load(logreg_path)


def predict(text: str, model: dict, threshold: float = THRESHOLD) -> list[str]:
    """
    Predict SDG tags for a single business text using the trained LogReg model.

    Args:
        text:      Business embedding text.
        model:     Loaded model dict from load_model().
        threshold: Probability threshold (default 0.5).

    Returns:
        List of matching SDG label names.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = encoder.encode([text], normalize_embeddings=True)          # (1, 384)
    probs = model["classifier"].predict_proba(emb)[0]                # (17,)
    return [model["labels"][i] for i, p in enumerate(probs) if p >= threshold]


def predict_batch(
    texts: list[str],
    model: dict,
    threshold: float = THRESHOLD,
    batch_size: int = 64,
) -> list[list[str]]:
    """
    Predict SDG tags for multiple texts using the trained LogReg model.

    Args:
        texts:      List of business embedding texts.
        model:      Loaded model dict from load_model().
        threshold:  Probability threshold.
        batch_size: Encoding batch size for SentenceTransformer.

    Returns:
        List of lists of SDG label names, one per input text.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk_embs = encoder.encode(texts[i: i + batch_size], normalize_embeddings=True)
        all_embs.append(chunk_embs)

    X = np.vstack(all_embs)                                          # (N, 384)
    probs = model["classifier"].predict_proba(X)                     # (N, 17)
    labels = model["labels"]
    return [
        [labels[j] for j, p in enumerate(row) if p >= threshold]
        for row in probs
    ]


def predict_from_embeddings(
    embeddings,
    model: dict,
    threshold: float = THRESHOLD,
) -> list[list[str]]:
    """Predict SDG tags from pre-computed 384-dim embedding vectors (from pgvector)."""
    X = np.array(embeddings)
    probs = model["classifier"].predict_proba(X)
    labels = model["labels"]
    return [
        [labels[j] for j, p in enumerate(row) if p >= threshold]
        for row in probs
    ]


# ---------------------------------------------------------------------------
# Zero-shot cosine similarity prediction (no training required)
# This is used for backfill since the SetFit zero-shot training produces
# unreliable probabilities with synthetic single-example data.
# ---------------------------------------------------------------------------

def _get_sdg_embeddings(encoder):
    """Embed SDG keyword descriptions once, normalised for cosine similarity."""
    keyword_texts = [SDG_KEYWORDS[label] for label in SDG_LABELS]
    return encoder.encode(keyword_texts, normalize_embeddings=True)


def predict_zero_shot(
    text: str,
    encoder,
    sdg_embeddings=None,
    threshold: float = COSINE_THRESHOLD,
) -> list[str]:
    """
    Predict SDG tags via cosine similarity between the business text and SDG
    keyword descriptions. No training required.

    Args:
        text:           Business embedding text.
        encoder:        SentenceTransformer instance.
        sdg_embeddings: Pre-computed SDG embeddings (pass to avoid recomputing).
        threshold:      Cosine similarity threshold (default 0.20).

    Returns:
        List of matching SDG label names.
    """
    if sdg_embeddings is None:
        sdg_embeddings = _get_sdg_embeddings(encoder)
    emb = encoder.encode([text], normalize_embeddings=True)[0]
    sims = sdg_embeddings @ emb
    return [SDG_LABELS[i] for i, s in enumerate(sims) if s >= threshold]


def predict_zero_shot_batch(
    texts: list[str],
    encoder,
    threshold: float = COSINE_THRESHOLD,
    batch_size: int = 64,
) -> list[list[str]]:
    """
    Batch zero-shot SDG prediction via cosine similarity. DB-agnostic.

    Args:
        texts:      List of business embedding texts.
        encoder:    SentenceTransformer instance.
        threshold:  Cosine similarity threshold.
        batch_size: Encoding batch size.

    Returns:
        List of lists of SDG label names, one per input text.
    """
    sdg_embeddings = _get_sdg_embeddings(encoder)   # (17, 384)

    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = encoder.encode(texts[i: i + batch_size], normalize_embeddings=True)
        all_embs.append(chunk)

    all_embs = np.vstack(all_embs)                  # (total, 384)
    sims = all_embs @ sdg_embeddings.T              # (total, 17)
    return [
        [SDG_LABELS[j] for j, s in enumerate(row) if s >= threshold]
        for row in sims
    ]


# ---------------------------------------------------------------------------
# SetFit multi-label classifier (parallel to LogReg)
# Combines real ChromaDB labels + synthetic template data from get_templated_dataset()
# ---------------------------------------------------------------------------

SETFIT_MODEL_DIR = Path(__file__).parent / "models" / "sdg_setfit_v2"


def train_setfit(
    save_path: Optional[Path] = None,
    num_iterations: int = 10,
    num_epochs: int = 1,
    batch_size: int = 16,
):
    """
    Train a SetFit multi-label SDG classifier.

    Combines:
      - Real labeled businesses from PostgreSQL (full text + real SDG tags)
      - Synthetic template sentences via get_templated_dataset()
        e.g. "This sentence is about Climate Action"

    Estimated training time: 5-10 minutes on CPU.

    Args:
        save_path:      Save directory. Defaults to SETFIT_MODEL_DIR.
        num_iterations: Contrastive pairs per class (lower = faster, default 10).
        num_epochs:     Fine-tuning epochs (default 1).
        batch_size:     Batch size for contrastive training (default 16).

    Returns:
        Trained SetFitModel.
    """
    import time
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments, get_templated_dataset

    save_path = Path(save_path) if save_path else SETFIT_MODEL_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    label2id = {label: i for i, label in enumerate(SDG_LABELS)}
    n = len(SDG_LABELS)

    # ------------------------------------------------------------------
    # Step 1: Real labeled data — full business text from PostgreSQL
    # ------------------------------------------------------------------
    from db.pg_store import PGStore
    store = PGStore()
    with store._cursor(dict_rows=True) as cur:
        cur.execute(
            "SELECT document, sdg_tags FROM businesses "
            "WHERE sdg_tags IS NOT NULL AND sdg_tags != ''"
        )
        pg_rows = list(cur.fetchall())

    real_texts, real_labels = [], []
    for row in pg_rows:
        sdg_str = (row["sdg_tags"] or "").strip()
        if not sdg_str:
            continue
        raw_tags = [t.strip() for t in sdg_str.split(",")]
        known_tags = [t for t in raw_tags if t in label2id]
        if not known_tags:
            continue
        vec = [0] * n
        for tag in known_tags:
            vec[label2id[tag]] = 1
        real_texts.append(row["document"] or "")
        real_labels.append(vec)

    print(f"Real labeled examples from PostgreSQL : {len(real_texts)}")

    # ------------------------------------------------------------------
    # Step 2: Synthetic template sentences via get_templated_dataset()
    # Each sentence is single-label → convert to multi-hot vector.
    # sample_size=8 → 17 labels × 8 = 136 short template sentences.
    # ------------------------------------------------------------------
    synth_ds = get_templated_dataset(candidate_labels=SDG_LABELS, sample_size=8)
    synth_texts = list(synth_ds["text"])
    print(synth_texts[:5])
    synth_labels = []
    for label_int in synth_ds["label"]:
        vec = [0] * n
        vec[int(label_int)] = 1
        synth_labels.append(vec)

    print(f"Synthetic template sentences        : {len(synth_texts)}")
    print(f"Total training examples             : {len(real_texts) + len(synth_texts)}")

    # ------------------------------------------------------------------
    # Step 3: Combine into HuggingFace Dataset
    # ------------------------------------------------------------------
    train_ds = Dataset.from_dict({
        "text": real_texts + synth_texts,
        "label": real_labels + synth_labels,
    })

    # ------------------------------------------------------------------
    # Step 4: Train SetFit with one-vs-rest multi-label head
    # ------------------------------------------------------------------
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        multi_target_strategy="one-vs-rest",
        labels=SDG_LABELS,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_iterations=num_iterations,
        ),
        train_dataset=train_ds,
    )

    print(f"\nTraining SetFit (num_iterations={num_iterations}, num_epochs={num_epochs})...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.0f}s  ({elapsed / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Step 5: Save
    # ------------------------------------------------------------------
    model.save_pretrained(str(save_path))
    print(f"SetFit model saved to {save_path}")
    return model


def load_setfit_model(model_path: Optional[Path] = None):
    """
    Load the trained SetFit model.

    Returns:
        SetFitModel instance.
    """
    from setfit import SetFitModel

    model_path = Path(model_path) if model_path else SETFIT_MODEL_DIR
    if not model_path.exists():
        raise FileNotFoundError(
            f"No SetFit model at {model_path}. "
            "Run: python -m ml.sdg_classifier train_setfit"
        )
    return SetFitModel.from_pretrained(str(model_path))


def predict_setfit(text: str, model, threshold: float = THRESHOLD) -> list[str]:
    """
    Predict SDG tags for a single text using the trained SetFit model.

    Args:
        text:      Business description text.
        model:     SetFitModel instance from load_setfit_model().
        threshold: Probability threshold (default 0.5).

    Returns:
        List of matching SDG label names.
    """
    probs = model.predict_proba([text])[0]
    if hasattr(probs, "numpy"):
        probs = probs.numpy()
    return [SDG_LABELS[i] for i, p in enumerate(probs) if float(p) >= threshold]


def predict_setfit_batch(
    texts: list[str],
    model,
    threshold: float = THRESHOLD,
) -> list[list[str]]:
    """
    Predict SDG tags for multiple texts using the trained SetFit model.

    Args:
        texts:     List of business description texts.
        model:     SetFitModel instance from load_setfit_model().
        threshold: Probability threshold.

    Returns:
        List of lists of SDG label names, one per input text.
    """
    probs = model.predict_proba(texts)
    if hasattr(probs, "numpy"):
        probs = probs.numpy()
    return [
        [SDG_LABELS[j] for j, p in enumerate(row) if float(p) >= threshold]
        for row in probs
    ]


# ---------------------------------------------------------------------------
# Helpers for evaluation analysis
# ---------------------------------------------------------------------------

def _plot_confusion(y_true, y_pred, method: str, threshold: float) -> str:
    """
    Generate a per-label confusion heatmap (TP/FP/FN counts per SDG).
    Saves to a temp PNG and returns the file path.
    """
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_labels = [
        lbl.replace("And", "&").replace("Sustainable", "Sust.")
           .replace("Responsible", "Resp.").replace("Communities", "Comm.")
        for lbl in SDG_LABELS
    ]

    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    data = np.stack([tp, fp, fn], axis=0)  # shape (3, 17)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(SDG_LABELS)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["TP", "FP", "FN"])
    ax.set_title(f"Per-SDG confusion — {method} (threshold={threshold})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False,
                                      prefix=f"confusion_{method}_")
    plt.savefig(tmp.name, dpi=120)
    plt.close(fig)
    return tmp.name


def _per_label_analysis(y_true, y_pred, rows) -> dict:
    """
    Compute per-SDG F1 and per-category precision to surface error patterns.
    Returns a dict summary for MLflow logging.
    """
    from sklearn.metrics import f1_score as _f1

    per_label_f1 = _f1(y_true, y_pred, average=None, zero_division=0)
    best_sdg  = SDG_LABELS[per_label_f1.argmax()]
    worst_sdg = SDG_LABELS[per_label_f1.argmin()]

    # Per-category precision
    cat_stats: dict = {}
    for i, row in enumerate(rows):
        cat = (row.get("categories") or "Unknown").split(",")[0].strip()
        if cat not in cat_stats:
            cat_stats[cat] = {"tp": 0, "fp": 0, "fn": 0}
        tp = int((y_true[i] * y_pred[i]).sum())
        fp = int(((1 - y_true[i]) * y_pred[i]).sum())
        fn = int((y_true[i] * (1 - y_pred[i])).sum())
        cat_stats[cat]["tp"] += tp
        cat_stats[cat]["fp"] += fp
        cat_stats[cat]["fn"] += fn

    print("\n  Per-SDG F1 (top 3 best / worst):")
    ranked = sorted(zip(SDG_LABELS, per_label_f1), key=lambda x: -x[1])
    for lbl, f in ranked[:3]:
        print(f"    ✓ {lbl}: {f:.2f}")
    for lbl, f in ranked[-3:]:
        print(f"    ✗ {lbl}: {f:.2f}")

    print("\n  Per-category error pattern:")
    for cat, s in sorted(cat_stats.items()):
        total = s["tp"] + s["fp"] + s["fn"]
        if total == 0:
            continue
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        print(f"    {cat}: precision={prec:.2f}  (TP={s['tp']} FP={s['fp']} FN={s['fn']})")

    return {
        "best_sdg": best_sdg,
        "worst_sdg": worst_sdg,
        "best_sdg_f1": round(float(per_label_f1.max()), 4),
        "worst_sdg_f1": round(float(per_label_f1.min()), 4),
    }


# ---------------------------------------------------------------------------
# LLM (Groq) backfill
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
You are an expert in UN Sustainable Development Goals (SDGs).

Given a company description, identify which SDGs this company EXPLICITLY or CLEARLY contributes to.

Rules:
1. If the description explicitly mentions SDG numbers or sustainability focus areas, prioritise those.
2. Only include SDGs with clear evidence in the description. Do not guess.
3. Return at most 4 SDGs. If none clearly apply, return an empty list.
4. Use ONLY the exact SDG names from this list:
{sdg_list}

Here are examples of correct SDG assignments:

Example 1:
Description: Heat Engineer Software Ltd provides advanced heating design software and compliance tools for UK heating engineers, renewable installers and low-carbon consultants. Their tools support the transition to heat pumps and other low-carbon heating systems.
Output: {{"sdg_tags": ["Affordable And Clean Energy", "Climate Action", "Responsible Consumption And Production", "Industry Innovation And Infrastructure"]}}

Example 2:
Description: Lancashire Women is a registered charity supporting women across Lancashire, providing trauma-informed services that help women improve their wellbeing, build resilience and move towards independence. Services include mental health support, domestic abuse recovery and employment skills.
Output: {{"sdg_tags": ["Gender Equality", "Good Health And Well-Being", "Decent Work And Economic Growth", "Reduced Inequalities"]}}

Example 3:
Description: Zuri Adventures is a UK based luxury travel consultancy specialising in tailor made wildlife, adventure and milestone journeys. They partner with conservation projects and local communities across Africa, with a focus on responsible tourism and protecting natural habitats.
Output: {{"sdg_tags": ["Life On Land", "Life Below Water", "Responsible Consumption And Production", "Climate Action"]}}

Example 4:
Description: SLART is a UK-based Outsider Artist who transforms themes of mortality, identity and human vulnerability into bold paintings that challenge social norms and advocate for marginalised communities.
Output: {{"sdg_tags": ["Gender Equality", "Reduced Inequalities", "Peace Justice And Strong Institutions"]}}

Now classify this company:

Company description:
{description}

Respond with valid JSON only, no explanation:
{{"sdg_tags": ["SDG name1", "SDG name2"]}}
"""

def backfill_llm(
    dry_run: bool = False,
    batch_size: int = 5,
    skip_existing: bool = True,
) -> dict:
    """
    Use LLM (Groq via existing agent.llm) to predict SDG tags for all businesses.

    Strategy:
    - First pass: companies whose description explicitly mentions SDGs → high confidence
    - Second pass: remaining companies → LLM infers from context
    - Writes results to predicted_sdg_tags in PostgreSQL

    Args:
        dry_run:       If True, print predictions but don't write to DB.
        batch_size:    Requests per second throttle (Groq rate limit).
        skip_existing: Skip companies that already have predicted_sdg_tags.

    Returns:
        Summary dict with counts.
    """
    import json
    import re
    import time
    from db.pg_store import PGStore
    from agent.llm import get_llm
    from psycopg2.extras import execute_batch

    store = PGStore()
    llm = get_llm("groq")

    sdg_list = "\n".join(f"- {lbl}" for lbl in SDG_LABELS)

    with store._cursor(dict_rows=True) as cur:
        if skip_existing:
            cur.execute(
                "SELECT id, name, document FROM businesses "
                "WHERE (predicted_sdg_tags IS NULL OR predicted_sdg_tags = '') "
                "AND document IS NOT NULL AND document != ''"
            )
        else:
            cur.execute(
                "SELECT id, name, document FROM businesses "
                "WHERE document IS NOT NULL AND document != ''"
            )
        rows = list(cur.fetchall())

    total = len(rows)
    print(f"Running LLM backfill on {total} businesses "
          f"({'skip existing' if skip_existing else 'overwrite all'})...")

    updates = []
    errors = 0

    for i, row in enumerate(rows):
        doc = (row["document"] or "")[:800]  # truncate to avoid token overflow
        prompt = _LLM_PROMPT.format(sdg_list=sdg_list, description=doc)

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Extract JSON robustly
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in response: {content[:100]}")
            parsed = json.loads(match.group())
            tags = parsed.get("sdg_tags", [])

            # Validate against known SDG labels
            valid_tags = [t for t in tags if t in SDG_LABELS]
            tag_str = ", ".join(valid_tags)
            updates.append((tag_str, row["id"]))

            print(f"  [{i+1}/{total}] {row['name'][:40]}: {valid_tags or '(none)'}")

        except Exception as e:
            print(f"  [{i+1}/{total}] {row['name'][:40]}: ERROR — {e}")
            updates.append(("", row["id"]))
            errors += 1

        # Rate limit: Groq free tier ~30 req/min
        if (i + 1) % batch_size == 0:
            time.sleep(1)

    with_pred = sum(1 for tag_str, _ in updates if tag_str)

    print(f"\nLLM prediction complete:")
    print(f"  Processed  : {total}")
    print(f"  With tags  : {with_pred}  ({with_pred * 100 // total if total else 0}%)")
    print(f"  Errors     : {errors}")

    if dry_run:
        print("\n[DRY RUN] Not writing to DB.")
        return {"total": total, "with_predictions": with_pred, "errors": errors, "dry_run": True}

    with store._cursor(dict_rows=False) as cur:
        execute_batch(
            cur,
            "UPDATE businesses SET predicted_sdg_tags = %s WHERE id = %s",
            updates,
            page_size=100,
        )
    print(f"Done. {total} records updated.")

    # MLflow logging
    try:
        import dagshub
        import mlflow
        dagshub.init(
            repo_owner="jungle173770",
            repo_name="SDG0_partner_finder_system",
            mlflow=True,
        )
        with mlflow.start_run(run_name="backfill_llm"):
            mlflow.log_param("method", "groq_llm")
            mlflow.log_param("model", "llama-3.1-8b-instant")
            mlflow.log_param("total_businesses", total)
            mlflow.log_param("skip_existing", skip_existing)
            mlflow.log_metric("coverage_rate", round(with_pred / total, 4) if total else 0)
            mlflow.log_metric("with_predictions", with_pred)
            mlflow.log_metric("errors", errors)
        print("MLflow run logged to Dagshub.")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")

    return {"total": total, "with_predictions": with_pred, "errors": errors, "dry_run": False}


# ---------------------------------------------------------------------------
# Evaluation against ground-truth sdg_tags
# ---------------------------------------------------------------------------

def evaluate(method: str = "zero_shot", threshold: float = THRESHOLD) -> dict:
    """
    Evaluate prediction quality against ground-truth sdg_tags.

    Uses the 20 businesses that have real sdg_tags as labeled eval set.
    Computes micro-averaged precision, recall, F1 across all SDG labels.

    Args:
        method:    "zero_shot" | "nli" | "setfit" | "logreg" | "llm"
        threshold: prediction threshold

    Returns:
        dict with precision, recall, f1, support, and per-label breakdown
    """
    from db.pg_store import PGStore
    from sklearn.metrics import precision_score, recall_score, f1_score
    import numpy as np

    store = PGStore()
    with store._cursor(dict_rows=True) as cur:
        cur.execute(
            "SELECT id, name, document, sdg_tags, categories FROM businesses "
            "WHERE sdg_tags IS NOT NULL AND sdg_tags != ''"
        )
        rows = list(cur.fetchall())

    if not rows:
        print("No labeled examples found.")
        return {}

    n_labels = len(SDG_LABELS)

    y_true, y_pred = [], []

    if method == "nli":
        from sentence_transformers import CrossEncoder
        nli_model_name = "cross-encoder/nli-deberta-v3-small"
        print(f"Loading NLI model: {nli_model_name}...")
        nli_model = CrossEncoder(nli_model_name)
        texts = [row["document"] or "" for row in rows]
        preds_list = []
        for text in texts:
            pairs = [(text, f"This organisation contributes to: {SDG_DESCRIPTIONS[lbl]}") for lbl in SDG_LABELS]
            scores = nli_model.predict(pairs, apply_softmax=True)
            # scores shape: (n_labels, 3) — [contradiction, neutral, entailment]
            entailment_scores = [float(s[2]) for s in scores]
            predicted = [SDG_LABELS[i] for i, s in enumerate(entailment_scores) if s >= threshold]
            preds_list.append(predicted)
        print("NLI predictions done.")
    elif method == "setfit":
        from setfit import SetFitModel
        setfit_path = MODEL_DIR.parent / "sdg_setfit_v2"
        if not setfit_path.exists():
            raise FileNotFoundError(f"SetFit model not found at {setfit_path}")
        print(f"Loading SetFit model from {setfit_path}...")
        sf_model = SetFitModel.from_pretrained(str(setfit_path))
        texts = [row["document"] or "" for row in rows]
        preds_list = [predict_setfit(t, sf_model, threshold=threshold) for t in texts]
    elif method == "llm":
        import json as _json
        import re as _re
        import time as _time
        from agent.llm import get_llm
        llm = get_llm("groq")
        texts = [row["document"] or "" for row in rows]
        sdg_list = "\n".join(f"- {lbl}" for lbl in SDG_LABELS)
        preds_list = []
        print(f"Running LLM (Groq) on {len(texts)} labeled businesses...")
        for i, text in enumerate(texts):
            prompt = _LLM_PROMPT.format(sdg_list=sdg_list, description=text[:800])
            try:
                response = llm.invoke(prompt)
                match = _re.search(r'\{.*\}', response.content.strip(), _re.DOTALL)
                parsed = _json.loads(match.group()) if match else {}
                tags = [t for t in parsed.get("sdg_tags", []) if t in SDG_LABELS]
            except Exception as e:
                print(f"  [{i+1}] ERROR: {e}")
                tags = []
            preds_list.append(tags)
            if (i + 1) % 5 == 0:
                _time.sleep(1)
        print("LLM predictions done.")
    elif method == "logreg":
        logreg_model = load_model()  # raises FileNotFoundError if not trained yet
        with store._cursor(dict_rows=True) as cur:
            cur.execute(
                "SELECT id, embedding FROM businesses "
                "WHERE sdg_tags IS NOT NULL AND sdg_tags != ''"
            )
            emb_rows = list(cur.fetchall())
        id_order = [r["id"] for r in emb_rows]
        embeddings = [list(r["embedding"]) for r in emb_rows]
        # reorder rows to match emb_rows order
        id_to_row = {r["id"]: r for r in rows}
        rows = [id_to_row[i] for i in id_order if i in id_to_row]
        print(f"Using LogReg on {len(rows)} labeled businesses (threshold={threshold})...")
        print("  ⚠️  WARNING: training set = eval set — F1 will be inflated")
        preds_list = predict_from_embeddings(embeddings, logreg_model, threshold=threshold)
    else:
        from sentence_transformers import SentenceTransformer
        print("Running zero-shot cosine similarity...")
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [row["document"] or "" for row in rows]
        preds_list = predict_zero_shot_batch(texts, encoder, threshold=threshold)

    for row, pred_tags in zip(rows, preds_list):
        true_tags = [t.strip() for t in (row["sdg_tags"] or "").split(",") if t.strip()]
        true_vec = [1 if SDG_LABELS[i] in true_tags else 0 for i in range(n_labels)]
        pred_vec = [1 if SDG_LABELS[i] in pred_tags else 0 for i in range(n_labels)]
        y_true.append(true_vec)
        y_pred.append(pred_vec)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="micro", zero_division=0)
    support   = int(y_true.sum())

    print(f"\n=== Evaluation: {method} (threshold={threshold}) ===")
    print(f"  Samples  : {len(rows)}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall   : {recall:.3f}")
    print(f"  F1       : {f1:.3f}")
    print(f"  Support  : {support} true labels total")

    # Per-label + per-category analysis
    label_summary = _per_label_analysis(y_true, y_pred, rows)

    # MLflow logging
    try:
        import dagshub
        import mlflow
        dagshub.init(
            repo_owner="jungle173770",
            repo_name="SDG0_partner_finder_system",
            mlflow=True,
        )
        heatmap_path = _plot_confusion(y_true, y_pred, method, threshold)
        with mlflow.start_run(run_name=f"eval_{method}"):
            mlflow.log_param("method", method)
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("n_samples", len(rows))
            mlflow.log_param("best_sdg", label_summary["best_sdg"])
            mlflow.log_param("worst_sdg", label_summary["worst_sdg"])
            mlflow.log_metric("precision", round(precision, 4))
            mlflow.log_metric("recall", round(recall, 4))
            mlflow.log_metric("f1", round(f1, 4))
            mlflow.log_metric("support", support)
            mlflow.log_metric("best_sdg_f1", label_summary["best_sdg_f1"])
            mlflow.log_metric("worst_sdg_f1", label_summary["worst_sdg_f1"])
            mlflow.log_artifact(heatmap_path, artifact_path="confusion")
            if method == "llm":
                mlflow.log_text(_LLM_PROMPT, "prompt_v1.txt")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")

    return {
        "method": method,
        "threshold": threshold,
        "n_samples": len(rows),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": support,
        **label_summary,
    }


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep(method: str = "llm", thresholds: list = None) -> list:
    """
    Evaluate a method across multiple thresholds to find the optimal F1.
    Each run is logged separately to MLflow for comparison.

    Args:
        method:     Method to sweep (same choices as evaluate).
        thresholds: List of thresholds to try. Defaults to [0.2, 0.3, 0.4, 0.5, 0.6].

    Returns:
        List of result dicts sorted by F1.
    """
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

    results = []
    for t in thresholds:
        print(f"\n{'='*40}\nSweeping {method} @ threshold={t}\n{'='*40}")
        result = evaluate(method=method, threshold=t)
        results.append(result)

    results.sort(key=lambda r: -r["f1"])
    print(f"\n=== Sweep complete: best threshold={results[0]['threshold']} "
          f"(F1={results[0]['f1']}) ===")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SDG classifier (LogReg + SetFit)")
    parser.add_argument(
        "command",
        choices=["train", "train_setfit", "backfill_llm", "predict", "stats", "evaluate", "sweep"],
        help="Command to run",
    )
    parser.add_argument("--text", type=str, help="Text to predict (for 'predict' command)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Probability threshold (default: {THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for backfill")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing predicted_sdg_tags (for backfill_llm)")
    parser.add_argument("--method", type=str, default="zero_shot",
                        choices=["zero_shot", "nli", "setfit", "logreg", "llm"],
                        help="Method to evaluate (for 'evaluate' command)")
    args = parser.parse_args()

    if args.command == "train":
        print("=" * 50)
        print("Training SDG LogReg classifier (PostgreSQL embeddings + synthetic)")
        print("=" * 50)
        train()

    elif args.command == "train_setfit":
        print("=" * 50)
        print("Training SDG SetFit classifier (real text + template sentences)")
        print("Estimated time: 5-10 minutes on CPU")
        print("=" * 50)
        train_setfit()

    elif args.command == "evaluate":
        _method = getattr(args, "method", "zero_shot")
        print("=" * 50)
        print(f"Evaluating SDG classifier: {_method}")
        print("=" * 50)
        result = evaluate(method=_method, threshold=args.threshold)
        print(f"\nResult: {json.dumps(result, indent=2)}")

    elif args.command == "backfill_llm":
        print("=" * 50)
        print("Backfilling predicted_sdg_tags using Groq LLM")
        print("=" * 50)
        skip = not getattr(args, "overwrite", False)
        result = backfill_llm(dry_run=args.dry_run, skip_existing=skip)
        print(f"\nResult: {json.dumps(result, indent=2)}")

    elif args.command == "sweep":
        _method = getattr(args, "method", "llm")
        print("=" * 50)
        print(f"Threshold sweep: {_method}")
        print("=" * 50)
        results = sweep(method=_method)
        print(f"\nAll results:\n{json.dumps(results, indent=2)}")

    elif args.command == "predict":
        if not args.text:
            print("Error: --text required for predict command")
            sys.exit(1)
        model = load_model()
        tags = predict(args.text, model, threshold=args.threshold)
        print(f"Predicted SDG tags: {tags}")

    elif args.command == "stats":
        # Show current coverage in PostgreSQL
        from db.pg_store import PGStore
        store = PGStore()
        with store._cursor(dict_rows=True) as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(CASE WHEN sdg_tags != '' THEN 1 END) AS with_orig,
                    COUNT(CASE WHEN predicted_sdg_tags != '' THEN 1 END) AS with_pred,
                    COUNT(CASE WHEN sdg_tags != '' OR predicted_sdg_tags != '' THEN 1 END) AS with_any
                FROM businesses
            """)
            stats = dict(cur.fetchone())
        total = stats["total"]
        with_orig = stats["with_orig"]
        with_pred = stats["with_pred"]
        with_any = stats["with_any"]
        print(f"\nSDG coverage in PostgreSQL ({total} businesses):")
        print(f"  Original sdg_tags          : {with_orig}  ({with_orig * 100 // total}%)")
        print(f"  Predicted sdg_tags         : {with_pred}  ({with_pred * 100 // total}%)")
        print(f"  Either field filled        : {with_any}  ({with_any * 100 // total}%)")
