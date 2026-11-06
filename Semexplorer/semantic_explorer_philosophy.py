import hashlib
import inspect
import os
import re
import sqlite3
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Semantic Explorer for Philosophy Texts
# Created by Brian Guo
#
# Philosophy texts from 120+ nations -> multilingual Transformer
# -> 768-dimensional embeddings -> UMAP -> interactive 3D semantic space.
#
# If your network cannot reach huggingface.co directly, use the mirror:
#   macOS/Linux: HF_ENDPOINT=https://hf-mirror.com streamlit run ...
#   or enable "Use hf-mirror.com mirror" in the app sidebar.
# ------------------------------------------------------------------

HF_ENDPOINT_ENV_KEY = "HF_ENDPOINT"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

try:
    import torch
except ImportError:
    torch = None

try:
    import umap.umap_ as umap
except ImportError:
    umap = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

try:
    import safetensors  # noqa: F401

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import jieba
except ImportError:
    jieba = None


# multilingual-e5-base covers ~100 languages and produces 768-dim vectors,
# which makes cross-country / cross-language comparison meaningful while
# keeping the "768D" description accurate. For pure English token-level
# demos, google-bert/bert-base-uncased remains a good choice.
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DB_PATH = os.path.join(APP_DIR, "philosophy_corpus.db")

st.set_page_config(
    page_title="Semantic Explorer for Philosophy Texts",
    page_icon="🧠",
    layout="wide",
)


def inject_aesthetic_background():
    """
    Calm, slowly flowing backdrop: a warm ivory gradient with three soft
    blurred orbs (indigo / sage / muted gold) drifting like idea clusters
    in semantic space. Purely decorative; honors reduced-motion settings.
    """
    st.markdown(
        """
<style>
/* Slow-flowing warm academic gradient */
.stApp {
    background: linear-gradient(
        115deg, #f8f5ef 0%, #edf0f5 45%, #f4efe4 100%
    );
    background-size: 280% 280%;
    animation: se-gradient-flow 55s ease-in-out infinite;
}
@keyframes se-gradient-flow {
    0%   { background-position: 0% 40%; }
    50%  { background-position: 100% 60%; }
    100% { background-position: 0% 40%; }
}

/* Drifting "idea clusters" — soft blurred orbs */
.se-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
}
.se-orb-indigo {
    width: 46vmax; height: 46vmax;
    top: -14vmax; left: -10vmax;
    background: #7286ad;
    opacity: 0.16;
    animation: se-drift-a 48s ease-in-out infinite;
}
.se-orb-sage {
    width: 38vmax; height: 38vmax;
    bottom: -12vmax; right: -8vmax;
    background: #8ba888;
    opacity: 0.14;
    animation: se-drift-b 62s ease-in-out infinite;
}
.se-orb-gold {
    width: 30vmax; height: 30vmax;
    top: 38%; left: 52%;
    background: #d3ab4e;
    opacity: 0.10;
    animation: se-drift-c 70s ease-in-out infinite;
}
@keyframes se-drift-a {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50%      { transform: translate(6vmax, 4vmax) scale(1.08); }
}
@keyframes se-drift-b {
    0%, 100% { transform: translate(0, 0) scale(1.05); }
    50%      { transform: translate(-5vmax, -4vmax) scale(0.96); }
}
@keyframes se-drift-c {
    0%, 100% { transform: translate(0, 0) scale(0.98); }
    50%      { transform: translate(-4vmax, 3vmax) scale(1.06); }
}

/* Keep app content above the orbs */
[data-testid="stMain"],
section.main,
.main .block-container {
    position: relative;
    z-index: 1;
}

/* Soft translucent sidebar */
[data-testid="stSidebar"] > div:first-child {
    background-color: rgba(250, 248, 243, 0.82);
    backdrop-filter: blur(10px);
}

/* Respect reduced-motion preferences */
@media (prefers-reduced-motion: reduce) {
    .stApp, .se-orb { animation: none; }
}
</style>

<div class="se-orb se-orb-indigo"></div>
<div class="se-orb se-orb-sage"></div>
<div class="se-orb se-orb-gold"></div>
        """,
        unsafe_allow_html=True,
    )


inject_aesthetic_background()


# ============================================================
# Compatibility and model helpers
# ============================================================


def show_missing_dependencies():
    """Stop early with an actionable message when required packages are absent."""
    missing = []

    if torch is None:
        missing.append("torch")

    if umap is None:
        missing.append("umap-learn")

    if AutoTokenizer is None or AutoModel is None:
        missing.append("transformers")

    if missing:
        st.error(
            "This environment is missing the Python packages required "
            "to run the app."
        )
        st.code(
            f"pip install {' '.join(missing)}",
            language="bash",
        )
        st.stop()


def get_device_and_dtype():
    """Choose a practical device and model precision for the current machine."""
    if torch.cuda.is_available():
        device = torch.device("cuda")

        if (
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        ):
            return device, torch.bfloat16

        return device, torch.float16

    mps_backend = getattr(torch.backends, "mps", None)

    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps"), torch.float16

    return torch.device("cpu"), torch.float32


# ============================================================
# Model download retry / mirror helpers
# ============================================================


def is_transient_network_error(exc):
    """Detect connection-level failures that are worth retrying."""
    transient_names = (
        "RemoteProtocolError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "ReadError",
        "PoolTimeout",
        "ConnectionError",
        "TimeoutError",
        "IncompleteRead",
        "ChunkedEncodingError",
        "LocalEntryNotFoundError",
    )

    seen = set()
    current = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if type(current).__name__ in transient_names:
            return True

        if isinstance(current, OSError):
            message = str(exc).lower()
        else:
            message = str(current).lower()

        if any(
            keyword in message
            for keyword in (
                "server disconnected",
                "connection reset",
                "connection aborted",
                "timed out",
                "name resolution",
                "temporarily unavailable",
                "does not appear to have a file named",
                "model.safetensors",
                "pytorch_model.bin",
            )
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


def clear_broken_model_cache(model_name):
    """Remove cached files for a model after a failed download."""
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return False

    removed = False

    try:
        cache_info = scan_cache_dir()

        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                cache_info.delete_revisions(
                    *(revision.commit_hash for revision in repo.revisions)
                ).execute()
                removed = True
    except Exception:
        pass

    return removed


def clear_model_cache_and_retry(model_name):
    """Delete the cached copy of a model, then load it again."""
    clear_broken_model_cache(model_name)
    load_model.clear()

    return load_model_with_retry(model_name)


def configure_hf_endpoint(endpoint):
    """Switch the Hugging Face endpoint at runtime."""
    os.environ[HF_ENDPOINT_ENV_KEY] = endpoint

    try:
        import huggingface_hub.constants as hub_constants

        hub_constants.ENDPOINT = endpoint
    except Exception:
        pass

    try:
        from huggingface_hub import HfApi

        HfApi().endpoint = endpoint
    except Exception:
        pass


def is_local_model_path(model_name):
    expanded = os.path.expanduser(model_name.strip())

    return (
        os.path.isdir(expanded)
        or expanded.startswith((".", "/", "~"))
    )


@st.cache_resource(show_spinner=False)
def load_model(model_name):
    """Load and cache the tokenizer/model so reruns do not reload them."""
    device, model_dtype = get_device_and_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
    }

    if not HAS_SAFETENSORS:
        load_kwargs["use_safetensors"] = False

    try:
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
    except OSError:
        if clear_broken_model_cache(model_name):
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
        else:
            raise

    model.to(device)
    model.eval()

    return tokenizer, model, device, model_dtype


def load_model_with_retry(model_name, max_attempts=3):
    """Load the model with a few automatic retries for network glitches."""
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            return load_model(model_name), attempt
        except Exception as exc:
            last_exc = exc

            if "does not appear to have a file named" in str(exc):
                raise

            if (
                is_local_model_path(model_name)
                or not is_transient_network_error(exc)
                or attempt == max_attempts
            ):
                raise

            time.sleep(min(2 ** attempt, 8))

    raise last_exc


def apply_hf_mirror_and_reload(model_name):
    """Switch to the hf-mirror.com endpoint and reload the model once."""
    configure_hf_endpoint(HF_MIRROR_ENDPOINT)
    load_model.clear()

    return load_model_with_retry(model_name)


def uses_e5_prefix(model_name):
    """E5 models expect 'query: '/'passage: ' prefixes on their inputs."""
    return "e5" in model_name.lower()


# ============================================================
# Corpus database
# ============================================================


@st.cache_data(show_spinner=False)
def load_corpus(db_path):
    """Load the philosophy-text corpus from SQLite into a DataFrame."""
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT id, text, author, work, country, region, tradition,"
        " language, era, source FROM texts",
        conn,
    )
    conn.close()
    return df


def model_cache_slug(model_name):
    return hashlib.md5(model_name.encode("utf-8")).hexdigest()[:12]


# ============================================================
# East / West sphere — a simpler lens on top of region
# ============================================================

# Coarse civilizational spheres, following the usual philosophy-survey
# convention: Islamic, Persian, Indic and Sinitic traditions count as
# East; Latin America inherits the European canon and counts as West.
REGION_SPHERE = {
    "East Asia": "East",
    "South Asia": "East",
    "Southeast Asia": "East",
    "Central Asia": "East",
    "Middle East & North Africa": "East",
    "Sub-Saharan Africa": "East",
    "Europe": "West",
    "North America": "West",
    "Latin America & Caribbean": "West",
    "Oceania": "West",
}

CURATED_PER_SPHERE = 10000

# Authors treated as "unnamed" for curation priority.
_UNNAMED_AUTHORS = {
    "", "anonymous", "traditional", "unknown", "proverb",
    "folk saying", "oral tradition",
}


def assign_sphere(df):
    """Attach a coarse 'sphere' column (East / West) derived from region."""
    df = df.copy()
    df["sphere"] = df["region"].map(REGION_SPHERE).fillna("")
    return df


def curated_corpus_mask(df, per_sphere=CURATED_PER_SPHERE):
    """
    Boolean mask selecting the most representative texts in each sphere.

    Transparent curation rule (not a quality judgment): texts attributed
    to a named author outrank anonymous ones, texts naming a source work
    outrank those without, and readable lengths (30–1200 chars) outrank
    fragments or oversized passages. Within each sphere the highest
    priority entries win; ties break by corpus id for determinism.
    """
    author = df["author"].fillna("").str.strip().str.lower()
    has_author = ~author.isin(_UNNAMED_AUTHORS)
    has_work = df["work"].fillna("").str.strip() != ""
    length = df["text"].fillna("").str.len()
    good_length = (length >= 30) & (length <= 1200)

    score = (
        has_author.astype(int) * 2
        + has_work.astype(int)
        + good_length.astype(int)
    )

    order = (
        pd.DataFrame(
            {"sphere": df["sphere"], "_score": score, "_id": df["id"]}
        )
        .sort_values(
            ["sphere", "_score", "_id"],
            ascending=[True, False, True],
        )
    )
    keep_ids = (
        order[order["sphere"] != ""]
        .groupby("sphere")
        .head(per_sphere)["_id"]
    )
    return df["id"].isin(set(keep_ids.tolist()))


def embedding_cache_path(model_name):
    return os.path.join(
        APP_DIR, f"corpus_embeddings_{model_cache_slug(model_name)}.npz"
    )


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden.float() * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts, tokenizer, model, device, model_name,
                batch_size=32, progress=None):
    """Batch-embed texts into sentence vectors via mean pooling."""
    prefix = "passage: " if uses_e5_prefix(model_name) else ""
    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start:start + batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        tensor_keys = {"input_ids", "attention_mask", "token_type_ids"}
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in tensor_keys
        }

        with torch.inference_mode():
            outputs = model(**inputs, return_dict=True)

        pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        vectors.append(pooled.cpu().numpy().astype(np.float32))

        if progress is not None:
            progress.progress(
                min(1.0, (start + len(batch)) / max(1, len(texts))),
                text=f"Embedding corpus… {start + len(batch)}/{len(texts)}",
            )

    return np.vstack(vectors)


def get_corpus_embeddings(corpus_df, tokenizer, model, device, model_name):
    """
    Return corpus embeddings, computing and caching only what is missing.

    The cache file lives next to this script, so the second launch of the
    app starts instantly. Rows are matched to cache entries by corpus id.
    """
    cache_path = embedding_cache_path(model_name)
    cached_ids, cached_vecs = None, None

    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            cached_ids = data["ids"]
            cached_vecs = data["vectors"]
        except Exception:
            cached_ids, cached_vecs = None, None

    ids = corpus_df["id"].to_numpy(dtype=np.int64)

    if cached_ids is not None:
        pos = {int(cid): i for i, cid in enumerate(cached_ids)}
        have = np.array([int(i) in pos for i in ids])
    else:
        pos = {}
        have = np.zeros(len(ids), dtype=bool)

    vectors = np.zeros(
        (len(ids), 0 if cached_vecs is None else cached_vecs.shape[1]),
        dtype=np.float32,
    )

    for row, idx in enumerate(ids):
        if have[row]:
            vectors[row] = cached_vecs[pos[int(idx)]]

    missing_rows = np.where(~have)[0]

    if len(missing_rows) > 0:
        progress = st.progress(
            0.0, text=f"Embedding corpus… 0/{len(missing_rows)}"
        )
        new_vecs = embed_texts(
            corpus_df.iloc[missing_rows]["text"].tolist(),
            tokenizer,
            model,
            device,
            model_name,
            progress=progress,
        )
        progress.empty()

        if vectors.shape[1] == 0:
            vectors = np.zeros(
                (len(ids), new_vecs.shape[1]), dtype=np.float32
            )
            for row, idx in enumerate(ids):
                if have[row]:
                    vectors[row] = cached_vecs[pos[int(idx)]]

        vectors[missing_rows] = new_vecs

        all_ids = ids
        all_vecs = vectors
        np.savez_compressed(cache_path, ids=all_ids, vectors=all_vecs)

    return vectors


# ============================================================
# Token-level helpers (from the original app)
# ============================================================


def readable_token(tokenizer, token_id):
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if text == "":
        return "[EMPTY]"

    special_tokens = {
        token: label
        for token, label in [
            (tokenizer.bos_token, "[BOS]"),
            (tokenizer.eos_token, "[EOS]"),
            (tokenizer.pad_token, "[PAD]"),
            (tokenizer.unk_token, "[UNK]"),
        ]
        if token is not None
    }

    if text in special_tokens:
        return special_tokens[text]

    if text == "\n":
        return "[NEWLINE]"

    if text == "\t":
        return "[TAB]"

    text = text.replace("\n", "\\n")
    text = text.replace("\t", "\\t")

    if text.startswith(" "):
        text = "␠" + text[1:]

    return text


def get_final_layer_vectors(text, tokenizer, model, device, max_tokens):
    """Tokenize, run the model, and return final-layer token vectors."""
    original_token_count = len(
        tokenizer(text, add_special_tokens=True)["input_ids"]
    )

    was_truncated = original_token_count > max_tokens

    encode_kwargs = {
        "return_tensors": "pt",
        "add_special_tokens": True,
    }

    if getattr(tokenizer, "is_fast", False):
        encode_kwargs["return_offsets_mapping"] = True

    if was_truncated:
        encode_kwargs.update(
            {"truncation": True, "max_length": max_tokens}
        )

    encoded = tokenizer(text, **encode_kwargs)

    tensor_keys = {"input_ids", "attention_mask", "token_type_ids"}
    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in tensor_keys
    }

    offset_mapping = encoded.get("offset_mapping")
    offsets = (
        None
        if offset_mapping is None
        else offset_mapping[0].cpu().tolist()
    )

    with torch.inference_mode():
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
        raise RuntimeError(
            "The model did not return hidden states. "
            "Please check whether the selected model supports "
            "output_hidden_states=True."
        )

    final_hidden = outputs.hidden_states[-1]

    vectors = np.ascontiguousarray(
        final_hidden[0].float().cpu().numpy(),
        dtype=np.float32,
    )
    token_ids = model_inputs["input_ids"][0].cpu().tolist()

    tokens = [
        readable_token(tokenizer, token_id)
        for token_id in token_ids
    ]

    return (
        vectors,
        tokens,
        token_ids,
        offsets,
        original_token_count,
        was_truncated,
    )


# ============================================================
# Word-level merging (jieba segmentation + mean pooling)
# ============================================================


def parse_custom_words(raw_text):
    """Parse a comma-separated custom word list from the sidebar."""
    if not raw_text:
        return []

    return [
        word.strip()
        for word in re.split(r"[,，、;；\n]", raw_text)
        if word.strip()
    ]


def segment_text_to_spans(text, custom_words=()):
    """Segment text into (start, end, word) character spans with jieba."""
    if jieba is None:
        return []

    for word in custom_words:
        jieba.add_word(word)

    return [
        (start, end, word)
        for word, start, end in jieba.tokenize(text)
    ]


def _pool_groups(groups, vectors):
    merged_vectors = np.stack(
        [vectors[g["indices"]].mean(axis=0) for g in groups]
    ).astype(np.float32)

    merged_tokens = [g["display"] for g in groups]

    return merged_vectors, merged_tokens


def merge_tokens_by_words(tokens, token_ids, vectors, offsets, word_spans):
    """Group subword tokens into segmented words, then mean-pool."""
    groups = []
    span_to_group = {}

    for index, (offset, token, _token_id) in enumerate(
        zip(offsets, tokens, token_ids)
    ):
        start, end = offset
        target_span = None
        target_word = None

        if start is not None and end is not None and end > start:
            best_overlap = 0

            for span_start, span_end, word in word_spans:
                overlap = min(end, span_end) - max(start, span_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    target_span = (span_start, span_end)
                    target_word = word

        if target_span is None:
            groups.append({"indices": [index], "display": token})
            continue

        if target_span in span_to_group:
            groups[span_to_group[target_span]]["indices"].append(index)
        else:
            span_to_group[target_span] = len(groups)
            groups.append({"indices": [index], "display": target_word})

    merged_vectors, merged_tokens = _pool_groups(groups, vectors)

    merged_token_ids = [
        ", ".join(str(token_ids[i]) for i in g["indices"])
        for g in groups
    ]

    merged_subtokens = [
        " + ".join(tokens[i] for i in g["indices"])
        for g in groups
    ]

    return merged_vectors, merged_tokens, merged_token_ids, merged_subtokens


def _group_is_special(group, offsets):
    """True if a group's first token is a special token (offset (0,0))."""
    start, end = offsets[group["indices"][0]]
    return not (start is not None and end is not None and end > start)


WORDPIECE_CONTINUATION = "##"  # BERT-style subword continuation
SENTENCEPIECE_START = "▁"      # SentencePiece word-start (XLM-R, E5)
BYTE_BPE_START = "Ġ"           # byte-level BPE word-start (GPT-style)
VISIBLE_SPACE = "␠"            # readable_token renders a leading space so

# Punctuation that always forms its own group, even when flush against a
# word. Apostrophes are excluded so contractions ("don't", "it’s") stay
# whole; hyphens are excluded so "well-known" stays whole.
HARD_PUNCTUATION = set(".,;:!?()[]{}\"“”«»—–…")


def _clean_piece(piece):
    """Strip subword markers from a piece for display."""
    for marker in (
        WORDPIECE_CONTINUATION,
        VISIBLE_SPACE,
        SENTENCEPIECE_START,
        BYTE_BPE_START,
    ):
        if piece.startswith(marker):
            piece = piece[len(marker):]
    return piece


def merge_tokens_by_wordpiece(tokens, token_ids, vectors, offsets):
    """Merge subword tokens back into whole words.

    Works for every tokenizer convention: BERT WordPiece (##),
    SentencePiece (▁ — used by the default multilingual-e5 model), and
    byte-level BPE (Ġ). A universal offset-contiguity fallback catches
    the rest: a token that begins exactly where the previous one ended
    belongs to the same word, so e.g. E5 splitting "seek" into
    "se" + "ek" is stitched back together.
    """
    groups = []
    prev_end = None
    prev_was_hard_punct = False

    for index, (offset, token, _token_id) in enumerate(
        zip(offsets, tokens, token_ids)
    ):
        start, end = offset
        is_special = not (
            start is not None and end is not None and end > start
        )

        if is_special:
            groups.append({"indices": [index], "display": token})
            prev_end = None
            prev_was_hard_punct = False
            continue

        cleaned = _clean_piece(token)
        is_hard_punct = (
            cleaned != ""
            and all(ch in HARD_PUNCTUATION for ch in cleaned)
        )

        if token.startswith(WORDPIECE_CONTINUATION):
            starts_new_word = False
        elif (
            token.startswith(VISIBLE_SPACE)
            or token.startswith(SENTENCEPIECE_START)
            or token.startswith(BYTE_BPE_START)
        ):
            starts_new_word = True
        elif is_hard_punct:
            starts_new_word = True
        elif (
            prev_end is not None
            and start == prev_end
            and not prev_was_hard_punct
            and groups
            and not _group_is_special(groups[-1], offsets)
        ):
            # No marker, but flush against the previous token — this is
            # a continuation piece (e.g. SentencePiece "se" + "ek").
            starts_new_word = False
        else:
            starts_new_word = True

        prev_end = end
        prev_was_hard_punct = is_hard_punct

        if (
            not starts_new_word
            and groups
            and not _group_is_special(groups[-1], offsets)
        ):
            groups[-1]["indices"].append(index)
        else:
            groups.append({"indices": [index], "display": cleaned})

    for group in groups:
        indices = group["indices"]

        if len(indices) == 1 and _group_is_special(group, offsets):
            group["display"] = tokens[indices[0]]
        else:
            group["display"] = "".join(
                _clean_piece(tokens[i]) for i in indices
            )

    merged_vectors, merged_tokens = _pool_groups(groups, vectors)

    merged_token_ids = [
        ", ".join(str(token_ids[i]) for i in g["indices"])
        for g in groups
    ]

    merged_subtokens = [
        " + ".join(tokens[i] for i in g["indices"])
        for g in groups
    ]

    return merged_vectors, merged_tokens, merged_token_ids, merged_subtokens


# ============================================================
# Vector helpers
# ============================================================


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def top_k_similar(target, vectors, k=5, exclude=()):
    """Return [(index, similarity)] for the k vectors nearest to target."""
    target = np.asarray(target, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)

    target_norm = np.linalg.norm(target)

    if target_norm == 0 or len(vectors) == 0:
        return []

    excluded = set(exclude)
    vector_norms = np.linalg.norm(vectors, axis=1)

    valid = vector_norms > 0
    for index in excluded:
        if 0 <= index < len(valid):
            valid[index] = False

    if not np.any(valid):
        return []

    similarities = np.full(len(vectors), -np.inf, dtype=np.float64)
    similarities[valid] = (
        vectors[valid] @ target / (vector_norms[valid] * target_norm)
    )

    order = np.argsort(similarities)[::-1]
    results = []

    for index in order:
        if not np.isfinite(similarities[index]):
            break
        results.append((int(index), float(similarities[index])))
        if len(results) >= k:
            break

    return results


def find_nearest_vector(target, vectors, exclude=()):
    """Find the nearest non-excluded vector using cosine similarity."""
    results = top_k_similar(target, vectors, k=1, exclude=exclude)

    if not results:
        return None, 0.0

    return results[0]


# ============================================================
# Streamlit / Plotly compatibility helpers
# ============================================================


def get_field(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]

        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def get_selected_indices(event, allowed_traces=(0,)):
    """Read selected point indices across Streamlit/Plotly versions."""
    if event is None:
        return []

    selection = get_field(event, "selection", default=event)
    points = get_field(selection, "points", default=[]) or []

    indices = []

    for point in points:
        trace_index = get_field(
            point, "trace_index", "curve_number", default=0
        )

        try:
            trace_index = 0 if trace_index is None else int(trace_index)
        except (TypeError, ValueError):
            trace_index = 0

        if trace_index not in allowed_traces:
            continue

        point_index = get_field(
            point, "point_index", "point_number", default=None
        )

        if point_index is None:
            customdata = get_field(point, "customdata", default=None)

            if isinstance(customdata, (list, tuple, np.ndarray)):
                if len(customdata) > 0:
                    point_index = customdata[0]

        if point_index is None:
            continue

        try:
            indices.append(int(point_index))
        except (TypeError, ValueError):
            continue

    if not indices:
        point_indices = (
            get_field(selection, "point_indices", default=[]) or []
        )

        for point_index in point_indices:
            try:
                indices.append(int(point_index))
            except (TypeError, ValueError):
                continue

    return list(dict.fromkeys(indices))


def streamlit_version_at_least(major, minor):
    version_match = re.match(
        r"(\d+)\.(\d+)",
        getattr(st, "__version__", "0.0"),
    )

    if version_match is None:
        return False

    current_major, current_minor = map(int, version_match.groups())

    return (current_major, current_minor) >= (major, minor)


def plotly_selection_supported():
    """Return True only when this Streamlit build supports chart selections."""
    try:
        parameters = inspect.signature(st.plotly_chart).parameters

        if "on_select" in parameters and "selection_mode" in parameters:
            return True

        has_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

        if has_kwargs:
            return streamlit_version_at_least(1, 35)
    except (TypeError, ValueError):
        return streamlit_version_at_least(1, 35)

    return False


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ============================================================
# Session-state helpers
# ============================================================


def initialize_session_state():
    st.session_state.setdefault("run_id", 0)
    st.session_state.setdefault("selection_version", 0)
    st.session_state.setdefault("selected_points", [])
    st.session_state.setdefault("arithmetic_result", None)
    st.session_state.setdefault("arithmetic_nearest", None)
    st.session_state.setdefault("manual_cosine_result", None)
    st.session_state.setdefault("corpus_selected", None)
    st.session_state.setdefault("concept_result", None)


def clear_generated_state():
    keys_to_clear = [
        "vectors", "tokens", "token_ids", "df", "reducer",
        "source_text", "source_model_name", "source_device",
        "source_dtype", "original_token_count", "was_truncated",
        "subtokens", "granularity", "selected_points",
        "arithmetic_result", "arithmetic_nearest",
        "manual_cosine_result", "manual_cosine_a", "manual_cosine_b",
        "arithmetic_A", "arithmetic_B",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    st.session_state["selected_points"] = []
    st.session_state["arithmetic_result"] = None
    st.session_state["arithmetic_nearest"] = None
    st.session_state["manual_cosine_result"] = None
    st.session_state["selection_version"] = 0
    st.session_state["run_id"] += 1


# ============================================================
# Main UI
# ============================================================


show_missing_dependencies()
initialize_session_state()

st.title("🧠 Semantic Explorer for Philosophy Texts")

st.markdown("**Created by Brian Guo**")

st.markdown(
    """
**An interactive map of the world's philosophical thought.**
Philosophy texts from **120+ nations** are encoded by a multilingual
Transformer into 768-dimensional embeddings and projected with UMAP
into an interactive 3D semantic space — revealing how ideas cluster
across languages and borders.
"""
)

with st.sidebar:
    st.header("Settings")

    model_name = st.text_input(
        "Model name or local path",
        value=DEFAULT_MODEL_NAME,
        help=(
            "Hugging Face model ID or a local model directory. "
            "Default: intfloat/multilingual-e5-base (768-dim, ~100 "
            "languages — required for cross-country comparison). "
            "For English token-level demos: google-bert/bert-base-uncased. "
            "For Chinese: Qwen/Qwen3-0.6B."
        ),
    )

    max_tokens = st.slider(
        "Maximum tokens processed (Text Explorer)",
        min_value=16,
        max_value=2048,
        value=512,
        step=16,
    )

    st.caption(
        "The first launch downloads the model and computes the corpus "
        "embeddings; later launches reuse the cache."
    )

    st.divider()
    st.subheader("Model Download")

    use_hf_mirror = st.toggle(
        "Use hf-mirror.com mirror",
        value=(
            os.environ.get(HF_ENDPOINT_ENV_KEY, "").rstrip("/")
            == HF_MIRROR_ENDPOINT
        ),
        help=(
            "Enable this if model downloads fail with 'Server "
            "disconnected without sending a response' or stall for "
            "a long time (e.g. on networks that cannot reach "
            "huggingface.co directly)."
        ),
    )

    st.divider()
    st.caption("Created by Brian Guo")


def load_selected_model():
    """Central model-loading flow with mirror / retry handling."""
    if use_hf_mirror:
        configure_hf_endpoint(HF_MIRROR_ENDPOINT)

    try:
        with st.spinner(f"Loading {model_name}..."):
            result, attempts = load_model_with_retry(model_name)

            if attempts > 1:
                st.toast(
                    f"Network hiccup — download succeeded on attempt "
                    f"{attempts}.",
                    icon="🔁",
                )
        return result
    except Exception as exc:
        message = str(exc)

        if (
            "does not appear to have a file named" in message
            or "model.safetensors" in message
        ):
            st.warning(
                "The model cache looks incomplete — clearing it and "
                "downloading again…"
            )
            try:
                with st.spinner(f"Re-downloading {model_name}..."):
                    result, _ = clear_model_cache_and_retry(model_name)
                return result
            except Exception as retry_exc:
                if not HAS_SAFETENSORS:
                    st.error(
                        "Could not load the model weights. This model "
                        "is distributed in safetensors format, but "
                        "safetensors is not installed in this "
                        "environment."
                    )
                    st.code("pip install safetensors", language="bash")
                st.exception(retry_exc)
                st.stop()
        elif (
            not use_hf_mirror
            and not is_local_model_path(model_name)
            and is_transient_network_error(exc)
        ):
            st.warning(
                "Download from huggingface.co failed (connection "
                "interrupted). Automatically trying the hf-mirror.com "
                "mirror…"
            )
            try:
                with st.spinner(
                    f"Loading {model_name} from hf-mirror.com..."
                ):
                    result, _ = apply_hf_mirror_and_reload(model_name)
                return result
            except Exception as mirror_exc:
                st.error(
                    "The mirror download failed as well. Try the "
                    "following:"
                )
                st.markdown(
                    """
1. Enable **Use hf-mirror.com mirror** in the sidebar and retry;
2. Check your proxy settings — if a system proxy is on, make sure
   the terminal uses it too (or run
   `unset http_proxy https_proxy all_proxy` and restart the app);
3. Launch with
   `HF_ENDPOINT=https://hf-mirror.com streamlit run ...`;
4. Enter a **local model path** in the sidebar's Model name field.
"""
                )
                st.exception(mirror_exc)
                st.stop()
        else:
            st.error(
                "Failed to load the model. Please check the model name "
                "or local path, and make sure this machine can reach "
                "Hugging Face."
            )

            if is_transient_network_error(exc):
                st.info(
                    "A network interruption was detected. Try enabling "
                    "**Use hf-mirror.com mirror** in the sidebar."
                )

            if not HAS_SAFETENSORS:
                st.info(
                    "Tip: pip install safetensors avoids most "
                    "weight-loading issues."
                )

            st.exception(exc)
            st.stop()


tokenizer, model, device, model_dtype = load_selected_model()

corpus_df = load_corpus(CORPUS_DB_PATH)

if corpus_df is not None and len(corpus_df) > 0:
    corpus_df = assign_sphere(corpus_df)

tab_corpus, tab_text, tab_about = st.tabs(
    ["🌍 Corpus Map", "🔬 Text Explorer", "ℹ️ About"]
)


# ============================================================
# TAB 1: Corpus Map — the 120+ nations view
# ============================================================

with tab_corpus:
    if corpus_df is None or len(corpus_df) == 0:
        st.warning(
            "Corpus database not found. Run this in the app directory "
            "first:\n\n`python3 build_corpus.py`"
        )
    else:
        st.subheader("Corpus-Level Semantic Map")
        st.caption(
            "Each point is a philosophy text. Texts that think alike "
            "sit close together—across languages and borders."
        )

        # ---------------- Filters ----------------
        filter_cols = st.columns(4)

        regions = sorted(corpus_df["region"].dropna().unique().tolist())
        countries = sorted(corpus_df["country"].dropna().unique().tolist())
        traditions = sorted(
            corpus_df["tradition"].dropna().unique().tolist()
        )

        with filter_cols[0]:
            selected_regions = st.multiselect(
                "Region", regions, default=regions, key="corpus_regions"
            )
        with filter_cols[1]:
            selected_countries = st.multiselect(
                "Country / nation", countries, key="corpus_countries"
            )
        with filter_cols[2]:
            selected_traditions = st.multiselect(
                "Tradition", traditions, key="corpus_traditions"
            )
        with filter_cols[3]:
            keyword = st.text_input(
                "Keyword in text", "", key="corpus_keyword"
            )

        sphere_cols = st.columns([2, 3])
        with sphere_cols[0]:
            sphere_choice = st.radio(
                "Civilization sphere",
                ["All", "East", "West"],
                horizontal=True,
                key="corpus_sphere",
            )
        with sphere_cols[1]:
            st.write("")
            curated = st.checkbox(
                f"Curated view — up to {CURATED_PER_SPHERE:,} most "
                f"representative texts per sphere",
                value=False,
                key="corpus_curated",
                help=(
                    "Keeps the top texts per sphere, prioritizing "
                    "named authors, attributed works and readable "
                    "lengths."
                ),
            )

        mask = corpus_df["region"].isin(selected_regions)

        if sphere_choice != "All":
            mask &= corpus_df["sphere"] == sphere_choice

        if selected_countries:
            mask &= corpus_df["country"].isin(selected_countries)

        if selected_traditions:
            mask &= corpus_df["tradition"].isin(selected_traditions)

        if keyword.strip():
            mask &= corpus_df["text"].str.contains(
                keyword.strip(), case=False, regex=False
            )

        if curated:
            mask &= curated_corpus_mask(corpus_df)

        filtered = corpus_df[mask].reset_index(drop=True)

        if len(filtered) < 4:
            st.info(
                "Fewer than 4 texts match the current filters. "
                "Please broaden the selection."
            )
        else:
            # ---------------- Embeddings ----------------
            corpus_vectors = get_corpus_embeddings(
                corpus_df, tokenizer, model, device, model_name
            )
            filtered_vectors = corpus_vectors[
                np.where(mask.to_numpy())[0]
            ]

            color_options = {
                "Region": "region",
                "Tradition": "tradition",
                "Language": "language",
                "East / West": "sphere",
            }
            color_label = st.radio(
                "Color points by",
                list(color_options.keys()),
                horizontal=True,
                key="corpus_color_by",
            )
            color_col = color_options[color_label]

            # ---------------- Display sampling ----------------
            # Browsers choke on very large 3D scatters, so plot a fixed
            # random sample. The FULL filtered_vectors are still used for
            # concept search and nearest-text features below.
            MAX_DISPLAY_POINTS = 20000
            sampled = len(filtered) > MAX_DISPLAY_POINTS
            if sampled:
                st.info(
                    f"Displaying a random sample of "
                    f"{MAX_DISPLAY_POINTS:,} of {len(filtered):,} "
                    f"filtered texts (search still covers all of them)."
                )
                sampled_idx = np.random.RandomState(42).choice(
                    len(filtered), MAX_DISPLAY_POINTS, replace=False
                )
                sampled_idx.sort()
                display_vectors = filtered_vectors[sampled_idx]
                display_df = filtered.iloc[sampled_idx].copy()
            else:
                display_vectors = filtered_vectors
                display_df = filtered.copy()

            # ---------------- UMAP projection ----------------
            cache_key = (
                f"corpus_proj_{model_cache_slug(model_name)}_"
                f"{len(display_df)}_"
                f"{hashlib.md5(display_df['id'].to_numpy().tobytes()).hexdigest()[:8]}"
            )

            if cache_key in st.session_state:
                coords = st.session_state[cache_key]
            else:
                with st.spinner("Projecting corpus into 3D..."):
                    reducer = umap.UMAP(
                        n_components=3,
                        n_neighbors=max(2, min(15, len(display_df) - 1)),
                        min_dist=0.1,
                        metric="cosine",
                        init="random",
                        random_state=42,
                    )
                    coords = reducer.fit_transform(display_vectors)
                st.session_state[cache_key] = coords

            plot_df = display_df
            plot_df["x"] = coords[:, 0]
            plot_df["y"] = coords[:, 1]
            plot_df["z"] = coords[:, 2]

            # ---------------- 3D scatter ----------------
            groups = list(plot_df[color_col].unique())
            palette = (
                "Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Light24"
            )
            import plotly.colors as pc

            color_pool = []
            for scale in palette:
                color_pool.extend(getattr(pc.qualitative, scale))

            fig = go.Figure()

            for gi, group in enumerate(groups):
                gdf = plot_df[plot_df[color_col] == group]
                customdata = np.column_stack(
                    [
                        gdf.index.values,
                        gdf["country"].values,
                        gdf["author"].values,
                        gdf["text"].str.slice(0, 120).values,
                    ]
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=gdf["x"], y=gdf["y"], z=gdf["z"],
                        mode="markers",
                        marker=dict(
                            size=6,
                            opacity=0.85,
                            color=color_pool[gi % len(color_pool)],
                        ),
                        name=str(group),
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{customdata[3]}…</b><br>"
                            "Author: %{customdata[2]}<br>"
                            "Country: %{customdata[1]}<br>"
                            "<extra>" + str(group) + "</extra>"
                        ),
                    )
                )

            shown_nations = min(filtered["country"].nunique(), 128)
            title_text = (
                f"Corpus semantic map — showing {len(plot_df):,} of "
                f"{len(filtered):,} texts from "
                f"{shown_nations} nations "
                f"({model_name})"
                if sampled
                else f"Corpus semantic map — {len(filtered):,} texts from "
                f"{shown_nations} nations "
                f"({model_name})"
            )
            fig.update_layout(
                title=title_text,
                paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2",
                    zaxis_title="UMAP 3",
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                ),
                height=720,
                margin=dict(l=0, r=0, b=0, t=50),
                showlegend=True,
                uirevision="corpus-map",
            )

            event = None

            if plotly_selection_supported():
                event = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="corpus_map_chart",
                    on_select="rerun",
                    selection_mode="points",
                    config={"scrollZoom": True, "displaylogo": False},
                )
            else:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="corpus_map_chart",
                    config={"scrollZoom": True, "displaylogo": False},
                )

            # ---------------- Click → full record ----------------
            clicked = get_selected_indices(
                event, allowed_traces=tuple(range(len(groups)))
            )

            if clicked:
                # Map the clicked point (trace, point) back to a row.
                selection = get_field(event, "selection", default=event)
                points = get_field(selection, "points", default=[]) or []
                for p in points:
                    trace_i = get_field(
                        p, "trace_index", "curve_number", default=0
                    )
                    point_i = get_field(
                        p, "point_index", "point_number", default=None
                    )
                    if point_i is None:
                        continue
                    try:
                        group = groups[int(trace_i)]
                        row = plot_df[plot_df[color_col] == group].iloc[
                            int(point_i)
                        ]
                        st.session_state["corpus_selected"] = row["id"]
                        break
                    except Exception:
                        continue

            selected_id = st.session_state.get("corpus_selected")

            if selected_id is not None:
                match = corpus_df[corpus_df["id"] == selected_id]
                if len(match) > 0:
                    row = match.iloc[0]
                    with st.container(border=True):
                        st.markdown(f"**{row['text']}**")
                        meta = []
                        if row["author"]:
                            meta.append(f"👤 {row['author']}")
                        if row["work"]:
                            meta.append(f"📖 {row['work']}")
                        meta.append(f"🌍 {row['country']} · {row['region']}")
                        if row["tradition"]:
                            meta.append(f"🏛️ {row['tradition']}")
                        if row["language"]:
                            meta.append(f"🗣️ {row['language']}")
                        if row["era"]:
                            meta.append(f"🕰️ {row['era']}")
                        st.caption(" ｜ ".join(meta))

                        # Top-k nearest texts = word/concept correlation
                        idx_all = corpus_df.index[
                            corpus_df["id"] == selected_id
                        ][0]
                        neighbors = top_k_similar(
                            corpus_vectors[idx_all],
                            corpus_vectors,
                            k=5,
                            exclude=(idx_all,),
                        )
                        st.markdown("**Most related texts in the corpus:**")
                        for ni, sim in neighbors:
                            nrow = corpus_df.iloc[ni]
                            st.write(
                                f"- {nrow['text'][:100]}… "
                                f"({nrow['country']}, "
                                f"similarity {sim:.3f})"
                            )

            st.divider()

            # ---------------- Concept search ----------------
            st.subheader("Concept Search — semantic correlation across nations")
            st.caption(
                "Type a concept (e.g. justice, freedom, the good life). "
                "It is embedded into the same space and matched against "
                "the whole corpus."
            )

            concept_cols = st.columns([4, 1])
            with concept_cols[0]:
                concept = st.text_input(
                    "Concept", key="concept_input",
                    placeholder="e.g. justice / freedom / the soul",
                )
            with concept_cols[1]:
                st.write("")
                st.write("")
                run_concept = st.button("Search", type="primary")

            if run_concept and concept.strip():
                prefix = "query: " if uses_e5_prefix(model_name) else ""
                qvec = embed_texts(
                    [prefix + concept.strip()],
                    tokenizer, model, device, model_name,
                )[0]
                results = top_k_similar(qvec, corpus_vectors, k=10)
                st.session_state["concept_result"] = (concept, results)

            if st.session_state.get("concept_result") is not None:
                q, results = st.session_state["concept_result"]
                st.markdown(f"**Top matches for “{q}”:**")
                rows = []
                for ni, sim in results:
                    nrow = corpus_df.iloc[ni]
                    rows.append(
                        {
                            "similarity": round(sim, 4),
                            "text": nrow["text"][:110],
                            "author": nrow["author"],
                            "country": nrow["country"],
                            "tradition": nrow["tradition"],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# TAB 2: Text Explorer — token-level analysis (original app)
# ============================================================

with tab_text:
    with st.expander("Tokenization settings", expanded=False):
        granularity = st.selectbox(
            "Token granularity",
            [
                "Raw subwords",
                "Merge subwords into whole words (recommended for English)",
                "Merge by word (jieba segmentation, for CJK text)",
            ],
            index=0,
            help=(
                "'Merge subwords into whole words' stitches subword "
                "pieces back into complete words — works with BERT (##), "
                "SentencePiece (E5/XLM-R) and GPT-style tokenizers; "
                "'Merge by word' uses jieba segmentation for "
                "Chinese/Japanese text."
            ),
        )

        custom_words_input = ""

        if granularity.startswith("Merge by word"):
            custom_words_input = st.text_input(
                "Custom words (optional)",
                value="",
                placeholder="e.g. enlightenment, dao",
            )

    text = st.text_area(
        "Enter text (any language the model supports):",
        height=160,
        placeholder=(
            "e.g. I seek wisdom, and I try to understand life, "
            "freedom, and justice."
        ),
    )

    generate = st.button(
        "Generate 3D Representation",
        type="primary",
    )

    if generate:
        if not text.strip():
            st.warning("Please enter some text first.")
            st.stop()

        clear_generated_state()

        try:
            with st.spinner(
                "Running Transformer and extracting final-layer "
                "representations..."
            ):
                (
                    vectors,
                    tokens,
                    token_ids,
                    offsets,
                    original_token_count,
                    was_truncated,
                ) = get_final_layer_vectors(
                    text, tokenizer, model, device, max_tokens
                )
        except Exception as exc:
            st.error("Model inference failed.")
            st.exception(exc)
            st.stop()

        subtokens = None

        if granularity.startswith("Merge subwords"):
            if offsets is None:
                st.warning(
                    "This tokenizer does not support offset mapping "
                    "(not a fast tokenizer) — falling back to raw "
                    "subword granularity."
                )
            else:
                vectors, tokens, token_ids, subtokens = (
                    merge_tokens_by_wordpiece(
                        tokens, token_ids, vectors, offsets
                    )
                )
        elif granularity.startswith("Merge by word"):
            if jieba is None:
                st.warning(
                    "jieba is not installed — falling back to raw "
                    "subword granularity. Install: pip install jieba"
                )
            elif offsets is None:
                st.warning(
                    "This tokenizer does not support offset mapping — "
                    "falling back to raw subword granularity."
                )
            else:
                word_spans = segment_text_to_spans(
                    text, parse_custom_words(custom_words_input)
                )

                if word_spans:
                    vectors, tokens, token_ids, subtokens = (
                        merge_tokens_by_words(
                            tokens, token_ids, vectors, offsets, word_spans
                        )
                    )

        if len(vectors) < 4:
            st.error(
                "Please enter a longer text. "
                "At least 4 tokens/words are required for 3D UMAP."
            )
            st.stop()

        try:
            with st.spinner(
                "Projecting high-dimensional representations into 3D..."
            ):
                reducer = umap.UMAP(
                    n_components=3,
                    n_neighbors=max(2, min(15, len(vectors) - 1)),
                    min_dist=0.1,
                    metric="cosine",
                    init="random",
                    random_state=42,
                )
                coordinates = reducer.fit_transform(vectors)
        except Exception as exc:
            st.error("UMAP projection failed.")
            st.exception(exc)
            st.stop()

        df_data = {
            "token": tokens,
            "token_id": token_ids,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "z": coordinates[:, 2],
        }

        if subtokens is not None:
            df_data["subtokens"] = subtokens

        st.session_state["vectors"] = vectors
        st.session_state["tokens"] = tokens
        st.session_state["token_ids"] = token_ids
        st.session_state["df"] = pd.DataFrame(df_data)
        st.session_state["reducer"] = reducer
        st.session_state["subtokens"] = subtokens
        st.session_state["granularity"] = granularity
        st.session_state["source_text"] = text
        st.session_state["source_device"] = str(device)
        st.session_state["source_dtype"] = str(model_dtype).replace(
            "torch.", ""
        )
        st.session_state["original_token_count"] = original_token_count
        st.session_state["was_truncated"] = was_truncated

    # ---------------- Display ----------------

    if "vectors" in st.session_state:
        vectors = st.session_state["vectors"]
        tokens = st.session_state["tokens"]
        df = st.session_state["df"]
        reducer = st.session_state["reducer"]
        subtokens = st.session_state.get("subtokens")
        run_id = st.session_state["run_id"]
        selection_version = st.session_state["selection_version"]

        st.subheader("Model Information")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            f"Number of {'Words' if subtokens is not None else 'Tokens'}",
            len(tokens),
        )
        col2.metric("Hidden Dimension", vectors.shape[1])
        col3.metric("Device", st.session_state.get("source_device", "?"))
        col4.metric("Model Dtype", st.session_state.get("source_dtype", "?"))
        st.caption(f"Model: `{model_name}`")

        if st.session_state.get("was_truncated"):
            st.warning(
                f"Input contained "
                f"{st.session_state.get('original_token_count')} tokens; "
                f"only the first {len(tokens)} tokens were processed."
            )

        st.subheader("Final-Layer Representation Space")

        show_token_labels = st.checkbox(
            "Show token labels on the chart",
            value=len(tokens) <= 120,
            key=f"show_token_labels_{run_id}",
        )

        fig = go.Figure()

        if subtokens is not None:
            customdata = np.column_stack(
                [
                    np.arange(len(df)),
                    df["token_id"].astype(str).values,
                    df["token"].values,
                    df["subtokens"].values,
                ]
            )
            hovertemplate = (
                "<b>Word:</b> %{customdata[2]}<br>"
                "<b>Token IDs:</b> %{customdata[1]}<br>"
                "<b>Merged Subwords:</b> %{customdata[3]}<br>"
                "<b>UMAP X:</b> %{x:.6f}<br>"
                "<b>UMAP Y:</b> %{y:.6f}<br>"
                "<b>UMAP Z:</b> %{z:.6f}<br>"
                "<extra></extra>"
            )
        else:
            customdata = np.column_stack(
                [
                    np.arange(len(df)),
                    df["token_id"].values,
                    df["token"].values,
                ]
            )
            hovertemplate = (
                "<b>Token:</b> %{customdata[2]}<br>"
                "<b>Token ID:</b> %{customdata[1]}<br>"
                "<b>UMAP X:</b> %{x:.6f}<br>"
                "<b>UMAP Y:</b> %{y:.6f}<br>"
                "<b>UMAP Z:</b> %{z:.6f}<br>"
                "<extra></extra>"
            )

        fig.add_trace(
            go.Scatter3d(
                x=df["x"], y=df["y"], z=df["z"],
                mode="markers+text" if show_token_labels else "markers",
                text=df["token"] if show_token_labels else None,
                textposition="top center",
                marker=dict(
                    size=6, opacity=0.85,
                    color=np.arange(len(df)),
                    colorscale="Viridis", showscale=False,
                ),
                customdata=customdata,
                hovertemplate=hovertemplate,
                name="Token Vectors",
            )
        )

        arithmetic_result = st.session_state.get("arithmetic_result")

        if arithmetic_result is not None:
            try:
                result_3d = reducer.transform(
                    np.asarray(arithmetic_result).reshape(1, -1)
                )[0]

                fig.add_trace(
                    go.Scatter3d(
                        x=[result_3d[0]], y=[result_3d[1]],
                        z=[result_3d[2]],
                        mode="markers+text",
                        text=["A + B RESULT"],
                        textposition="top center",
                        marker=dict(
                            size=13, symbol="diamond", color="red"
                        ),
                        name="A + B Result",
                    )
                )
            except Exception as exc:
                st.warning(
                    "The calculated vector could not be projected into "
                    "the existing UMAP space."
                )
                st.caption(str(exc))

        fig.update_layout(
            title=f"{model_name} Final-Layer Representation Space",
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                zaxis_title="UMAP Dimension 3",
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=50),
            showlegend=True,
            uirevision=f"representation-{run_id}",
        )

        chart_key = f"representation_space_{run_id}_{selection_version}"
        chart_config = {"scrollZoom": True, "displaylogo": False}

        event = None

        if plotly_selection_supported():
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                key=chart_key,
                on_select="rerun",
                selection_mode="points",
                config=chart_config,
            )
        else:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=chart_key,
                config=chart_config,
            )
            st.info(
                "This Streamlit version does not support point "
                "selection on the chart. Please use Manual Vector "
                "Selection below."
            )

        clicked_indices = [
            index
            for index in get_selected_indices(event)
            if 0 <= index < len(tokens)
        ]

        if clicked_indices:
            st.session_state["selected_points"] = clicked_indices[-2:]

        # ---------------- Cosine similarity ----------------

        st.divider()
        st.subheader("1. Cosine Similarity & Word Correlation")

        selected = [
            index
            for index in st.session_state.get("selected_points", [])
            if 0 <= index < len(tokens)
        ]

        if len(selected) == 0:
            st.info("Click one point in the 3D chart.")
        elif len(selected) == 1:
            index = selected[0]
            st.info(
                f"Selected: **{tokens[index]}** "
                f"(token index {index}). Click a second point."
            )

            # Word correlation: top-5 nearest tokens for the selection
            neighbors = top_k_similar(
                vectors[index], vectors, k=5, exclude=(index,)
            )
            if neighbors:
                st.markdown(
                    f"**Top correlated tokens for “{tokens[index]}”:**"
                )
                corr_rows = [
                    {
                        "token": tokens[ni],
                        "cosine similarity": round(sim, 4),
                    }
                    for ni, sim in neighbors
                ]
                st.dataframe(
                    pd.DataFrame(corr_rows),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            i, j = selected[-2:]
            similarity = cosine_similarity(vectors[i], vectors[j])

            c1, c2, c3 = st.columns(3)
            c1.metric("Vector A", tokens[i])
            c2.metric("Vector B", tokens[j])
            c3.metric("Cosine Similarity", f"{similarity:.6f}")

            st.latex(
                r"""
                \cos(\theta)
                =
                \frac{A\cdot B}
                {\|A\|\|B\|}
                """
            )

            st.caption(
                "The calculation uses the original high-dimensional "
                "final-layer vectors, not the 3D UMAP coordinates."
            )

        if st.button(
            "Clear Selected Vectors",
            key=f"clear_selected_{run_id}_{selection_version}",
        ):
            st.session_state["selected_points"] = []
            st.session_state["selection_version"] += 1
            rerun_app()

        # ---------------- Manual cosine fallback ----------------

        st.markdown("### Manual Vector Selection")

        options = list(range(len(tokens)))
        c1, c2, c3 = st.columns([3, 3, 1])

        with c1:
            manual_a = st.selectbox(
                "Vector A", options,
                format_func=lambda i: f"{tokens[i]} | token {i}",
                key="manual_cosine_a",
            )
        with c2:
            manual_b = st.selectbox(
                "Vector B", options,
                format_func=lambda i: f"{tokens[i]} | token {i}",
                key="manual_cosine_b",
            )
        with c3:
            st.write("")
            st.write("")
            compare = st.button("Compare")

        if compare:
            st.session_state["manual_cosine_result"] = (
                manual_a,
                manual_b,
                cosine_similarity(vectors[manual_a], vectors[manual_b]),
            )

        manual_result = st.session_state.get("manual_cosine_result")

        if manual_result is not None:
            result_a, result_b, result_similarity = manual_result
            st.success(
                f"**{tokens[result_a]} ↔ {tokens[result_b]}**\n\n"
                f"Cosine similarity = **{result_similarity:.6f}**"
            )

        # ---------------- Vector arithmetic ----------------

        st.divider()
        st.subheader("2. Vector Arithmetic")

        c1, c2 = st.columns(2)

        with c1:
            arithmetic_a = st.selectbox(
                "Vector A", options,
                format_func=lambda i: f"{tokens[i]} | token {i}",
                key="arithmetic_A",
            )
        with c2:
            arithmetic_b = st.selectbox(
                "Vector B", options,
                format_func=lambda i: f"{tokens[i]} | token {i}",
                key="arithmetic_B",
            )

        st.markdown(
            f"### Formula\n**{tokens[arithmetic_a]} + "
            f"{tokens[arithmetic_b]}**"
        )

        if st.button("Calculate A + B", type="primary"):
            result_vector = vectors[arithmetic_a] + vectors[arithmetic_b]
            st.session_state["arithmetic_result"] = result_vector

            nearest_index, nearest_similarity = find_nearest_vector(
                result_vector, vectors,
                exclude=(arithmetic_a, arithmetic_b),
            )

            st.session_state["arithmetic_nearest"] = (
                nearest_index, nearest_similarity,
                arithmetic_a, arithmetic_b,
            )
            rerun_app()

        arithmetic_nearest = st.session_state.get("arithmetic_nearest")

        if arithmetic_nearest is not None:
            (
                nearest_index, nearest_similarity,
                result_a, result_b,
            ) = arithmetic_nearest

            if nearest_index is not None:
                st.success(
                    f"**{tokens[result_a]} + {tokens[result_b]}**\n\n"
                    f"Nearest token: **{tokens[nearest_index]}**\n\n"
                    f"Cosine similarity: **{nearest_similarity:.6f}**"
                )

            if st.button(
                "Clear Arithmetic Result", key=f"clear_arith_{run_id}"
            ):
                st.session_state["arithmetic_result"] = None
                st.session_state["arithmetic_nearest"] = None
                rerun_app()

        # ---------------- Token table ----------------

        st.divider()
        st.subheader("Tokenization & 3D Coordinates")

        display_df = df.copy()

        for column in ["x", "y", "z"]:
            display_df[column] = display_df[column].round(6)

        table_columns = ["token", "token_id"]

        if "subtokens" in display_df.columns:
            table_columns.append("subtokens")

        table_columns += ["x", "y", "z"]

        st.dataframe(
            display_df[table_columns],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 3: About
# ============================================================

with tab_about:
    st.subheader("About this project")
    st.markdown("**Created by Brian Guo**")

    if corpus_df is not None and len(corpus_df) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Philosophy texts", len(corpus_df))
        c2.metric("Nations covered", min(corpus_df["country"].nunique(), 128))
        c3.metric("Languages", corpus_df["language"].nunique())
        c4.metric("Traditions", corpus_df["tradition"].nunique())

    st.markdown(
        """
**Pipeline.** Each text is encoded by a multilingual Transformer
(default: `intfloat/multilingual-e5-base`, 768 dimensions, ~100
languages). Sentence vectors are mean-pooled from the final layer,
then projected to 3D with UMAP (cosine metric). Texts that express
similar ideas cluster together regardless of language.

**Why 120+ nations — and not more?** The corpus is bounded by two
objective filters, not by selection bias:

1. **Availability** — only openly licensed, digitized philosophical
   texts can be included. Many traditions survive in manuscripts,
   oral transmission, or copyrighted editions that cannot be used.
2. **Model language coverage** — the embedding model supports about
   100 languages, so nations whose texts exist only outside that set
   cannot yet be represented.

The 120+ nations are essentially the full set of traditions that meet
both criteria. This limitation is documented in the accompanying
research paper as a direction for future work.

**East / West spheres.** Beyond region and tradition, every text is
assigned to one of two broad civilizational spheres: **East** (East,
South, Southeast & Central Asia; Middle East & North Africa; Sub-Saharan
Africa) and **West** (Europe, North America, Latin America & Caribbean,
Oceania). The curated view shows up to 10,000 representative texts per
sphere, prioritizing texts with a named author, an attributed source
work, and a readable length.

**Data.** The corpus lives in `philosophy_corpus.db` (SQLite), built by
`build_corpus.py`. The bundled seed data is a demonstration set of
public-domain quotes and traditional proverbs; import a research corpus
with `python3 build_corpus.py --import your_corpus.csv`.

**Embeddings cache.** After the first run, corpus vectors are stored in
`corpus_embeddings_<model>.npz` next to this script, so subsequent
launches are instant.
"""
    )
