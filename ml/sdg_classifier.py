"""
SetFit-based SDG multi-label classifier.

Zero-shot training: uses official SDG descriptions to generate synthetic
training data — no manual labelling required.

Usage
-----
# One-time setup (run once, takes ~5 min on CPU):
    python -m ml.sdg_classifier train
    python -m ml.sdg_classifier backfill       # write predicted_sdg_tags into ChromaDB

# Predict for a single business (called by pipeline):
    from ml.sdg_classifier import load_model, predict
    model = load_model()
    tags = predict("Company description text...", model)

PostgreSQL migration
--------------------
Only `backfill_chroma()` is ChromaDB-specific.
`predict()` and `predict_batch()` are DB-agnostic — they just take text and return lists.
When migrating to PostgreSQL, replace `backfill_chroma()` with a function that runs
    UPDATE businesses SET predicted_sdg_tags = ? WHERE id = ?
Everything else stays the same.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

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
    "Industry Innovation Cities And Communities",  # SDG 9 (SDGZero naming)
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
    "Industry Innovation Cities And Communities": (
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
    "Industry Innovation Cities And Communities": "R&D technology manufacturing engineering smart cities infrastructure innovation startups deep tech",
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
    chroma_dir: str = "./chroma_db",
) -> dict:
    """
    Train a multi-label SDG classifier using pre-computed ChromaDB embeddings.

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
        chroma_dir: Path to ChromaDB directory.

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
    # Step 1: Real labeled examples — use stored ChromaDB embeddings
    # No transformer call needed; embeddings are already computed.
    # ------------------------------------------------------------------
    import os as _os
    _os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    from db.chroma_store import BusinessStore

    store = BusinessStore(persist_dir=chroma_dir)
    all_data = store.collection.get(include=["embeddings", "metadatas"])

    real_count = 0
    for emb, meta in zip(all_data["embeddings"], all_data["metadatas"]):
        sdg_str = meta.get("sdg_tags", "").strip()
        if not sdg_str:
            continue
        raw_tags = [t.strip() for t in sdg_str.split(",")]
        known_tags = [t for t in raw_tags if t in label2id]
        if not known_tags:
            continue

        vec = [0] * len(SDG_LABELS)
        for tag in known_tags:
            vec[label2id[tag]] = 1
        X.append(emb)
        y.append(vec)
        real_count += 1

    print(f"Real labeled examples from ChromaDB : {real_count}")

    # ------------------------------------------------------------------
    # Step 2: Synthetic examples — encode 17 SDG keyword descriptions.
    # Only 17 short strings, takes ~0.1s. Covers labels with no real data.
    # ------------------------------------------------------------------
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    synth_texts = [SDG_KEYWORDS[label] for label in SDG_LABELS]
    synth_embs = encoder.encode(synth_texts, normalize_embeddings=True)

    for i, label in enumerate(SDG_LABELS):
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
    """
    Predict SDG tags directly from pre-computed embedding vectors.
    Use this with ChromaDB stored embeddings to avoid any transformer inference.

    Args:
        embeddings: Array-like of shape (N, 384) — e.g. from ChromaDB get().
        model:      Loaded model dict from load_model().
        threshold:  Probability threshold.

    Returns:
        List of lists of SDG label names, one per embedding.
    """
    X = np.array(embeddings)
    probs = model["classifier"].predict_proba(X)                     # (N, 17)
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
    chroma_dir: str = "./chroma_db",
    num_iterations: int = 10,
    num_epochs: int = 1,
    batch_size: int = 16,
):
    """
    Train a SetFit multi-label SDG classifier.

    Combines:
      - Real labeled businesses from ChromaDB (full text + real SDG tags)
      - Synthetic template sentences via get_templated_dataset()
        e.g. "This sentence is about Climate Action"

    Estimated training time: 5-10 minutes on CPU.

    Args:
        save_path:      Save directory. Defaults to SETFIT_MODEL_DIR.
        chroma_dir:     Path to ChromaDB.
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
    # Step 1: Real labeled data — full business text from ChromaDB
    # ------------------------------------------------------------------
    import os as _os
    _os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    from db.chroma_store import BusinessStore

    store = BusinessStore(persist_dir=chroma_dir)
    all_data = store.collection.get(include=["documents", "metadatas"])

    real_texts, real_labels = [], []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        sdg_str = meta.get("sdg_tags", "").strip()
        if not sdg_str:
            continue
        raw_tags = [t.strip() for t in sdg_str.split(",")]
        known_tags = [t for t in raw_tags if t in label2id]
        if not known_tags:
            continue
        vec = [0] * n
        for tag in known_tags:
            vec[label2id[tag]] = 1
        real_texts.append(doc)
        real_labels.append(vec)

    print(f"Real labeled examples from ChromaDB : {len(real_texts)}")

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


def compare_models(
    chroma_dir: str = "./chroma_db",
    threshold: float = THRESHOLD,
    n_samples: int = 8,
) -> dict:
    """
    Compare LogReg vs SetFit side-by-side on all ChromaDB businesses.

    Prints:
      - Overall coverage for both models
      - Exact-match agreement rate
      - n_samples sampled businesses showing real tags + both predictions

    Args:
        chroma_dir: Path to ChromaDB.
        threshold:  Probability threshold for both models.
        n_samples:  How many example businesses to print (half labeled, half random).

    Returns:
        Summary dict with counts.
    """
    import random

    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    from db.chroma_store import BusinessStore

    store = BusinessStore(persist_dir=chroma_dir)
    all_data = store.collection.get(include=["documents", "embeddings", "metadatas"])
    ids = all_data["ids"]
    docs = all_data["documents"]
    embeddings = all_data["embeddings"]
    metadatas = all_data["metadatas"]
    total = len(ids)

    # --- LogReg: use pre-stored embeddings (fast, no transformer) ---
    print("Loading LogReg model...")
    logreg = load_model()
    logreg_preds = predict_from_embeddings(embeddings, logreg, threshold=threshold)
    print(f"LogReg predictions done.")

    # --- SetFit: encode texts with fine-tuned transformer ---
    print("Loading SetFit model...")
    try:
        sf_model = load_setfit_model()
    except FileNotFoundError:
        print("SetFit model not found. Run: python -m ml.sdg_classifier train_setfit")
        return {}
    print(f"Running SetFit on {total} texts (this may take a minute)...")
    sf_preds = predict_setfit_batch(docs, sf_model, threshold=threshold)
    print("SetFit predictions done.")

    # --- Summary stats ---
    lr_with = sum(1 for p in logreg_preds if p)
    sf_with = sum(1 for p in sf_preds if p)
    exact_agree = sum(1 for a, b in zip(logreg_preds, sf_preds) if set(a) == set(b))

    print(f"\n{'=' * 62}")
    print(f"  MODEL COMPARISON  |  {total} businesses  |  threshold={threshold}")
    print(f"{'=' * 62}")
    print(f"  LogReg  coverage  :  {lr_with}/{total}  ({lr_with * 100 // total}%)")
    print(f"  SetFit  coverage  :  {sf_with}/{total}  ({sf_with * 100 // total}%)")
    print(f"  Exact agreement   :  {exact_agree}/{total}  ({exact_agree * 100 // total}%)")
    print(f"{'=' * 62}")

    # --- Sample output ---
    labeled_idx = [i for i, m in enumerate(metadatas) if m.get("sdg_tags", "").strip()]
    unlabeled_idx = [i for i in range(total) if i not in set(labeled_idx)]
    half = n_samples // 2
    sample_idx = labeled_idx[:min(half, len(labeled_idx))]
    remainder = n_samples - len(sample_idx)
    if remainder > 0 and unlabeled_idx:
        sample_idx += random.sample(unlabeled_idx, min(remainder, len(unlabeled_idx)))

    print(f"\n  SAMPLE PREDICTIONS  ({len(sample_idx)} businesses)\n")
    for i in sample_idx:
        name = metadatas[i].get("name", ids[i])[:45]
        real = metadatas[i].get("sdg_tags", "")
        lr = logreg_preds[i]
        sf = sf_preds[i]
        status = "agree" if set(lr) == set(sf) else "DIFFER"
        print(f"  [{name}]")
        if real:
            print(f"    Real   : {real}")
        print(f"    LogReg : {lr if lr else '(none)'}")
        print(f"    SetFit : {sf if sf else '(none)'}")
        print(f"    Status : {status}\n")

    return {
        "total": total,
        "logreg_coverage": lr_with,
        "setfit_coverage": sf_with,
        "exact_agreement": exact_agree,
    }


# ---------------------------------------------------------------------------
# ChromaDB-specific backfill
# (Replace this function with a PostgreSQL UPDATE when migrating)
# ---------------------------------------------------------------------------

def backfill_chroma(
    store=None,
    model=None,
    threshold: float = THRESHOLD,
    dry_run: bool = False,
) -> dict:
    """
    Predict SDG tags for all businesses and write `predicted_sdg_tags`
    into ChromaDB metadata.

    Uses the trained LogReg model when available (fastest: uses stored
    ChromaDB embeddings, no transformer inference at all).
    Falls back to zero-shot cosine similarity if no model is trained.

    Args:
        store:     BusinessStore instance. Created automatically if None.
        model:     Loaded model dict. Loaded automatically if None.
        threshold: Probability threshold (LogReg) or cosine threshold (fallback).
        dry_run:   If True, print predictions but don't write to DB.

    Returns:
        Summary dict with counts.

    PostgreSQL migration note:
        Replace this function with:
            SELECT id, embedding FROM businesses
            predictions = predict_from_embeddings(embeddings, model)
            UPDATE businesses SET predicted_sdg_tags = ? WHERE id = ?
    """
    from db.chroma_store import BusinessStore

    if store is None:
        store = BusinessStore(persist_dir="./chroma_db")

    # Try to load trained LogReg model; fall back to cosine similarity
    use_logreg = False
    if model is None:
        try:
            model = load_model()
            use_logreg = True
        except FileNotFoundError:
            pass
    elif isinstance(model, dict) and "classifier" in model:
        use_logreg = True

    print("Fetching all businesses from ChromaDB...")
    if use_logreg:
        # Pull stored embeddings — no transformer inference needed
        all_data = store.collection.get(include=["embeddings", "metadatas"])
        ids = all_data["ids"]
        metadatas = all_data["metadatas"]
        embeddings = all_data["embeddings"]
        total = len(ids)
        print(f"Using trained LogReg model on {total} businesses "
              f"(threshold={threshold}, embeddings from ChromaDB)...")
        predictions = predict_from_embeddings(embeddings, model, threshold=threshold)
    else:
        # Fallback: zero-shot cosine similarity (no model required)
        from sentence_transformers import SentenceTransformer
        all_data = store.collection.get(include=["metadatas", "documents"])
        ids = all_data["ids"]
        metadatas = all_data["metadatas"]
        documents = all_data["documents"]
        total = len(ids)
        cos_threshold = threshold if threshold <= 0.5 else COSINE_THRESHOLD
        print(f"No trained model found — using cosine similarity on {total} businesses "
              f"(threshold={cos_threshold})...")
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        predictions = predict_zero_shot_batch(documents, encoder, threshold=cos_threshold)

    # Stats
    with_pred = sum(1 for p in predictions if p)
    print(f"\nPrediction complete:")
    print(f"  Total businesses : {total}")
    print(f"  With predictions : {with_pred}  ({with_pred * 100 // total}%)")

    if dry_run:
        print("\n[DRY RUN] Sample predictions (first 5 with results):")
        shown = 0
        for meta, pred in zip(metadatas, predictions):
            if pred and shown < 5:
                print(f"  {meta.get('name')}: {pred}")
                shown += 1
        return {"total": total, "with_predictions": with_pred, "dry_run": True}

    # Write back to ChromaDB in batches
    batch_size = 100
    updated = 0
    for i in range(0, total, batch_size):
        chunk_ids = ids[i: i + batch_size]
        chunk_metas = metadatas[i: i + batch_size]
        chunk_preds = predictions[i: i + batch_size]

        new_metas = []
        for meta, pred in zip(chunk_metas, chunk_preds):
            new_meta = dict(meta)
            new_meta["predicted_sdg_tags"] = ", ".join(pred) if pred else ""
            new_metas.append(new_meta)

        store.collection.update(ids=chunk_ids, metadatas=new_metas)
        updated += len(chunk_ids)
        print(f"  Updated {updated}/{total}...")

    print(f"\nDone. {total} records updated with predicted_sdg_tags.")
    return {"total": total, "with_predictions": with_pred, "dry_run": False}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SDG classifier (LogReg + SetFit)")
    parser.add_argument(
        "command",
        choices=["train", "train_setfit", "compare", "backfill", "predict", "stats"],
        help="Command to run",
    )
    parser.add_argument("--text", type=str, help="Text to predict (for 'predict' command)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Probability threshold (default: {THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for backfill")
    args = parser.parse_args()

    if args.command == "train":
        print("=" * 50)
        print("Training SDG LogReg classifier (ChromaDB embeddings + synthetic)")
        print("=" * 50)
        train()

    elif args.command == "train_setfit":
        print("=" * 50)
        print("Training SDG SetFit classifier (real text + template sentences)")
        print("Estimated time: 5-10 minutes on CPU")
        print("=" * 50)
        train_setfit()

    elif args.command == "compare":
        print("=" * 50)
        print("Comparing LogReg vs SetFit on all ChromaDB businesses")
        print("=" * 50)
        compare_models(threshold=args.threshold)

    elif args.command == "backfill":
        print("=" * 50)
        print("Backfilling predicted_sdg_tags into ChromaDB")
        print("=" * 50)
        result = backfill_chroma(threshold=args.threshold, dry_run=args.dry_run)
        print(f"\nResult: {json.dumps(result, indent=2)}")

    elif args.command == "predict":
        if not args.text:
            print("Error: --text required for predict command")
            sys.exit(1)
        model = load_model()
        tags = predict(args.text, model, threshold=args.threshold)
        print(f"Predicted SDG tags: {tags}")

    elif args.command == "stats":
        # Show current coverage in ChromaDB
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        from db.chroma_store import BusinessStore
        store = BusinessStore(persist_dir="./chroma_db")
        all_data = store.collection.get(include=["metadatas"])
        total = len(all_data["metadatas"])
        with_orig = sum(1 for m in all_data["metadatas"] if m.get("sdg_tags", "").strip())
        with_pred = sum(1 for m in all_data["metadatas"] if m.get("predicted_sdg_tags", "").strip())
        with_any = sum(
            1 for m in all_data["metadatas"]
            if m.get("sdg_tags", "").strip() or m.get("predicted_sdg_tags", "").strip()
        )
        print(f"\nSDG coverage in ChromaDB ({total} businesses):")
        print(f"  Original sdg_tags          : {with_orig}  ({with_orig * 100 // total}%)")
        print(f"  Predicted sdg_tags (SetFit) : {with_pred}  ({with_pred * 100 // total}%)")
        print(f"  Either field filled         : {with_any}  ({with_any * 100 // total}%)")
