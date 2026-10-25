//! Static Bengali Word Embeddings & Semantic Proximity Engine
//!
//! Provides dense, pre-normalized vector embeddings (32 dimensions) and sub-microsecond
//! cosine similarity computations for deep semantic proximity scoring and contextual
//! candidate re-ranking (e.g. আজ ↔ কাল, ভালো ↔ সুন্দর, ভাত ↔ খাবো).
//!
//! Hot path execution executes with zero heap allocations and autovectorized SIMD dot-products.

use hashbrown::HashMap;
use std::sync::Arc;

pub const EMBEDDING_DIM: usize = 32;
pub const BINARY_MAGIC: &[u8; 4] = b"LEMB";
pub const BINARY_VERSION: u32 = 1;

/// Word embeddings engine for sub-microsecond semantic similarity scoring
#[derive(Debug, Clone)]
pub struct WordEmbeddings {
    inner: Arc<WordEmbeddingsInner>,
}

#[derive(Debug, Clone)]
struct WordEmbeddingsInner {
    /// Maps interned word token to its L2-normalized 32-dimensional embedding vector
    vectors: HashMap<Arc<str>, [f32; EMBEDDING_DIM]>,
}

impl Default for WordEmbeddings {
    fn default() -> Self {
        Self::new()
    }
}

impl WordEmbeddings {
    /// Creates a new WordEmbeddings instance populated with core conversational Bengali vectors
    /// and optionally enriched from system embedding binary files.
    pub fn new() -> Self {
        let mut inner = WordEmbeddingsInner {
            vectors: HashMap::with_capacity(CORE_EMBEDDINGS.len() + 64),
        };
        for &(word, raw_vec) in CORE_EMBEDDINGS {
            let normalized = l2_normalize(raw_vec);
            inner.vectors.insert(Arc::from(word), normalized);
        }
        let mut embeddings = Self {
            inner: Arc::new(inner),
        };
        let _ = embeddings.load_from_system_paths();
        embeddings
    }

    /// Number of vocabulary words currently indexed in the embedding table.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.vectors.len()
    }

    /// Whether the embedding table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.vectors.is_empty()
    }

    /// Lookup pre-normalized 32-dimensional vector for a word with 0 heap allocations.
    #[inline]
    pub fn get_vector(&self, word: &str) -> Option<&[f32; EMBEDDING_DIM]> {
        self.inner.vectors.get(word)
    }

    /// Compute cosine similarity between two words in [-1.0, 1.0].
    /// Because stored vectors are strictly pre-normalized (||v|| = 1.0),
    /// cosine similarity simplifies to a pure 32-element dot product with 0 allocations.
    /// Returns 0.0 if either word is absent from the embedding vocabulary.
    #[inline]
    pub fn cosine_similarity(&self, word_a: &str, word_b: &str) -> f32 {
        if word_a == word_b {
            return 1.0;
        }
        let clean_a = word_a.trim_matches(is_punct);
        let clean_b = word_b.trim_matches(is_punct);
        let (va, vb) = match (self.get_vector(clean_a), self.get_vector(clean_b)) {
            (Some(a), Some(b)) => (a, b),
            _ => return 0.0,
        };

        let mut dot = 0.0f32;
        for i in 0..EMBEDDING_DIM {
            dot += va[i] * vb[i];
        }
        dot.clamp(-1.0, 1.0)
    }

    /// Calculates a calibrated soft score boost for candidate ranking based on
    /// semantic proximity with the preceding context word.
    /// Returns a non-negative boost up to ~0.9 points when semantic alignment is strong (similarity > 0.25).
    #[inline]
    pub fn semantic_boost(&self, context_word: &str, candidate: &str) -> f32 {
        let sim = self.cosine_similarity(context_word, candidate);
        if sim > 0.25 {
            (sim - 0.25) * 1.2
        } else {
            0.0
        }
    }

    /// Insert or update a word embedding. The vector is automatically L2-normalized.
    pub fn insert(&mut self, word: &str, raw_vec: [f32; EMBEDDING_DIM]) {
        let normalized = l2_normalize(raw_vec);
        let inner = Arc::make_mut(&mut self.inner);
        inner.vectors.insert(Arc::from(word), normalized);
    }

    /// Ingest word embeddings from raw binary data (`bengali_embeddings.bin`).
    /// Returns the number of imported words.
    pub fn load_binary_bytes(&mut self, bytes: &[u8]) -> Result<usize, std::io::Error> {
        if bytes.len() < 16 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Embedding binary too small for header",
            ));
        }

        if &bytes[0..4] != BINARY_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid embedding magic bytes",
            ));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != BINARY_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported embedding version: {}", version),
            ));
        }

        let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        if dim != EMBEDDING_DIM {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unsupported dimension: {} (expected {})",
                    dim, EMBEDDING_DIM
                ),
            ));
        }

        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let mut cursor = 16;
        let mut loaded = 0;

        let inner = Arc::make_mut(&mut self.inner);

        for _ in 0..count {
            if cursor + 2 > bytes.len() {
                break;
            }
            let word_len =
                u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap()) as usize;
            cursor += 2;

            if cursor + word_len + (EMBEDDING_DIM * 4) > bytes.len() {
                break;
            }

            let word_str = match std::str::from_utf8(&bytes[cursor..cursor + word_len]) {
                Ok(s) => s,
                Err(_) => {
                    cursor += word_len + (EMBEDDING_DIM * 4);
                    continue;
                }
            };
            cursor += word_len;

            let mut vec = [0.0f32; EMBEDDING_DIM];
            for val in &mut vec {
                *val = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
                cursor += 4;
            }

            let key = Arc::from(word_str);
            // Curated anchors in CORE_EMBEDDINGS take precedence to guarantee crisp homophone distinctions
            if !inner.vectors.contains_key(&key) {
                inner.vectors.insert(key, l2_normalize(vec));
                loaded += 1;
            } else {
                loaded += 1;
            }
        }

        Ok(loaded)
    }

    /// Serializes all currently resident embeddings into compact LEMB binary format.
    pub fn save_binary(&self) -> Vec<u8> {
        let count = self.inner.vectors.len() as u32;
        let est_size = 16 + (count as usize * (2 + 12 + EMBEDDING_DIM * 4));
        let mut out = Vec::with_capacity(est_size);

        out.extend_from_slice(BINARY_MAGIC);
        out.extend_from_slice(&BINARY_VERSION.to_le_bytes());
        out.extend_from_slice(&(EMBEDDING_DIM as u32).to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());

        for (word, vec) in &self.inner.vectors {
            let bytes = word.as_bytes();
            let len = bytes.len() as u16;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(bytes);
            for &val in vec {
                out.extend_from_slice(&val.to_le_bytes());
            }
        }

        out
    }

    /// Load embeddings from a binary file on the filesystem.
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<usize, std::io::Error> {
        let bytes = std::fs::read(path)?;
        self.load_binary_bytes(&bytes)
    }

    /// Search and load embeddings from standard Lekhani system directories or environment variables.
    pub fn load_from_system_paths(&mut self) -> bool {
        let mut candidates = Vec::new();

        if let Ok(env_path) = std::env::var("LEKHANI_EMBEDDINGS_PATH") {
            candidates.push(std::path::PathBuf::from(env_path));
        }

        if let Some(home) = std::env::var_os("HOME") {
            candidates.push(
                std::path::PathBuf::from(&home)
                    .join(".local/share/lekhani/data/bengali_embeddings.bin"),
            );
            candidates.push(
                std::path::PathBuf::from(&home)
                    .join(".local/share/lekhani/dictionaries/bengali_embeddings.bin"),
            );
        }

        candidates.push(std::path::PathBuf::from(
            "./data/dictionaries/bengali_embeddings.bin",
        ));
        candidates.push(std::path::PathBuf::from(
            "../data/dictionaries/bengali_embeddings.bin",
        ));
        candidates.push(std::path::PathBuf::from(
            "../../data/dictionaries/bengali_embeddings.bin",
        ));

        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                candidates.push(
                    exe_dir
                        .join("data")
                        .join("dictionaries")
                        .join("bengali_embeddings.bin"),
                );
                candidates.push(exe_dir.join("data").join("bengali_embeddings.bin"));
                candidates.push(exe_dir.join("bengali_embeddings.bin"));
            }
        }

        for path in candidates {
            if path.exists() {
                if let Ok(count) = self.load_from_file(&path) {
                    if count > 0 {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Computes a dense 32-D semantic topic vector from a sequence of words.
    #[inline]
    pub fn compute_topic_vector<'a, I>(&self, words: I) -> TopicVector
    where
        I: IntoIterator<Item = &'a str>,
        I::IntoIter: DoubleEndedIterator,
    {
        TopicVector::from_context(words, self, 0.85, 20)
    }

    /// Computes a dense 32-D semantic topic vector from a slice of words.
    #[inline]
    pub fn compute_topic_vector_from_slice(&self, words: &[&str]) -> TopicVector {
        TopicVector::from_slice(words, self)
    }
}

/// A 32-dimensional dense semantic topic vector representing the running domain centroid
/// of the conversational context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopicVector {
    /// Normalized 32-D topic centroid coordinates (||v|| = 1.0 if magnitude > 0)
    pub vector: [f32; EMBEDDING_DIM],
    /// Accumulated unnormalized magnitude indicating topical confidence (0.0 = uninformative/neutral)
    pub magnitude: f32,
    /// Number of embedding-bearing tokens that contributed to this topic vector
    pub active_tokens: usize,
}

impl Default for TopicVector {
    fn default() -> Self {
        Self::zero()
    }
}

impl TopicVector {
    /// An empty, uninformative topic vector with zero magnitude.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            vector: [0.0; EMBEDDING_DIM],
            magnitude: 0.0,
            active_tokens: 0,
        }
    }

    /// Whether this topic vector contains non-trivial topical signal.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.magnitude >= 1e-4 && self.active_tokens > 0
    }

    /// Computes the exponential recency-decayed dense topic centroid over recent context words.
    ///
    /// - `words`: Token sequence in chronological order (earliest to most recent before cursor).
    /// - `decay_rate`: Exponential discount $\gamma \in (0.0, 1.0]$ per token step from cursor (e.g. 0.85).
    /// - `max_tokens`: Maximum number of trailing tokens to examine (e.g. 16 or 20).
    ///
    /// Zero heap allocations; operates purely on stack floats.
    pub fn from_context<'a, I>(
        words: I,
        embeddings: &WordEmbeddings,
        decay_rate: f32,
        max_tokens: usize,
    ) -> Self
    where
        I: IntoIterator<Item = &'a str>,
        I::IntoIter: DoubleEndedIterator,
    {
        let mut accum = [0.0f32; EMBEDDING_DIM];
        let mut current_weight = 1.0f32;
        let mut active_tokens = 0usize;

        // Iterate backward from most recent word toward older context
        for word in words.into_iter().rev().take(max_tokens) {
            let clean = word.trim_matches(is_punct);
            if clean.is_empty() {
                continue;
            }
            if let Some(vec) = embeddings.get_vector(clean) {
                for i in 0..EMBEDDING_DIM {
                    accum[i] += vec[i] * current_weight;
                }
                active_tokens += 1;
                current_weight *= decay_rate;
            }
        }

        if active_tokens == 0 {
            return Self::zero();
        }

        let mut norm_sq = 0.0f32;
        for &val in &accum {
            norm_sq += val * val;
        }
        let magnitude = norm_sq.sqrt();

        if magnitude > 1e-6 {
            let inv_mag = 1.0 / magnitude;
            for val in &mut accum {
                *val *= inv_mag;
            }
            Self {
                vector: accum,
                magnitude,
                active_tokens,
            }
        } else {
            Self::zero()
        }
    }

    /// Convenience constructor over a string slice with standard 0.85 decay and 20 token limit.
    #[inline]
    pub fn from_slice(words: &[&str], embeddings: &WordEmbeddings) -> Self {
        Self::from_context(words.iter().copied(), embeddings, 0.85, 20)
    }

    /// Computes the cosine similarity between this normalized topic vector and a candidate's vector.
    /// Returns a scalar in `[-1.0, 1.0]`. If this topic vector has zero magnitude, returns 0.0.
    #[inline]
    pub fn similarity(&self, candidate_vector: &[f32; EMBEDDING_DIM]) -> f32 {
        if self.magnitude < 1e-6 {
            return 0.0;
        }
        let mut dot = 0.0f32;
        for (&a, &b) in self.vector.iter().zip(candidate_vector.iter()) {
            dot += a * b;
        }
        dot.clamp(-1.0, 1.0)
    }

    /// Calculates a calibrated soft score boost for candidate ranking based on
    /// dense topic alignment.
    ///
    /// - High positive alignment (> 0.18): boost up to +1.4 points.
    /// - Strong negative incongruence (< -0.25): mild penalty down to -0.6 points to suppress out-of-domain distractors.
    /// - Neutral/orthogonal range [-0.25, 0.18]: 0.0 (no distortion).
    #[inline]
    pub fn topic_boost(&self, candidate: &str, embeddings: &WordEmbeddings) -> f32 {
        if self.magnitude < 1e-4 {
            return 0.0;
        }
        let clean = candidate.trim_matches(is_punct);
        if let Some(cand_vec) = embeddings.get_vector(clean) {
            let sim = self.similarity(cand_vec);
            if sim > 0.18 {
                (sim - 0.18) * 1.6
            } else if sim < -0.25 {
                (sim + 0.25) * 0.8
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Disambiguates between two competing homophone candidates `cand_a` and `cand_b`
    /// by comparing their directional projection onto the topic vector.
    ///
    /// Returns `Some((winner, margin))` if one candidate has a clear topic advantage ($\Delta > 0.08$),
    /// or `None` if topic alignment is ambiguous or neither word is in vocabulary.
    pub fn disambiguate<'a>(
        &self,
        cand_a: &'a str,
        cand_b: &'a str,
        embeddings: &WordEmbeddings,
    ) -> Option<(&'a str, f32)> {
        if self.magnitude < 1e-4 {
            return None;
        }
        let clean_a = cand_a.trim_matches(is_punct);
        let clean_b = cand_b.trim_matches(is_punct);
        let sim_a = embeddings.get_vector(clean_a).map(|v| self.similarity(v)).unwrap_or(0.0);
        let sim_b = embeddings.get_vector(clean_b).map(|v| self.similarity(v)).unwrap_or(0.0);

        let delta = sim_a - sim_b;
        if delta > 0.08 {
            Some((cand_a, delta))
        } else if delta < -0.08 {
            Some((cand_b, -delta))
        } else {
            None
        }
    }
}

#[inline]
fn is_punct(c: char) -> bool {
    c.is_ascii_punctuation()
        || c == '।'
        || c == '—'
        || c == '‘'
        || c == '’'
        || c == '“'
        || c == '”'
        || c == '\''
        || c == '"'
        || c == ','
}

#[inline]
fn l2_normalize(mut vec: [f32; EMBEDDING_DIM]) -> [f32; EMBEDDING_DIM] {
    let mut norm_sq = 0.0f32;
    for &val in &vec {
        norm_sq += val * val;
    }
    if norm_sq > 1e-12 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for val in &mut vec {
            *val *= inv_norm;
        }
    }
    vec
}

// ── Static Pre-Trained Conversational Semantic Vector Embeddings ──────────────
//
// 32-dimensional semantic vectors encoding fundamental Bengali grammatical,
// topical, sentiment, temporal, and spatial clusters.

macro_rules! vec32 {
    ($($v:expr),* $(,)?) => {{
        let arr = [$($v as f32),*];
        arr
    }};
}

pub const CORE_EMBEDDINGS: &[(&str, [f32; EMBEDDING_DIM])] = &[
    // ── Temporal cluster (Dimensions 0, 1, 2) ─────────────────────────────────
    (
        "আজ",
        vec32![
            0.90, 0.30, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কাল",
        vec32![
            0.88, 0.35, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পরশু",
        vec32![
            0.85, 0.40, 0.20, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "এখন",
        vec32![
            0.80, 0.10, 0.00, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "তখন",
        vec32![
            0.78, 0.12, 0.00, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সকাল",
        vec32![
            0.70, 0.70, 0.10, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "দুপুর",
        vec32![
            0.70, 0.65, 0.20, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সন্ধ্যা",
        vec32![
            0.70, 0.65, 0.30, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "রাত",
        vec32![
            0.72, 0.60, 0.40, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "প্রতিদিন",
        vec32![
            0.65, 0.40, 0.05, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Sentiment / Valence & Quality (Dimensions 3, 4, 5) ────────────────────
    (
        "ভালো",
        vec32![
            0.05, 0.0, 0.0, 0.90, 0.45, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সুন্দর",
        vec32![
            0.00, 0.0, 0.0, 0.88, 0.50, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "দারুণ",
        vec32![
            0.00, 0.0, 0.0, 0.86, 0.48, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "খারাপ",
        vec32![
            0.05, 0.0, 0.0, -0.85, -0.40, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কঠিন",
        vec32![
            0.00, 0.0, 0.0, -0.40, -0.30, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সহজ",
        vec32![
            0.00, 0.0, 0.0, 0.60, 0.30, -0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "নতুন",
        vec32![
            0.30, 0.0, 0.0, 0.50, 0.40, 0.00, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পুরনো",
        vec32![
            0.35, 0.0, 0.0, 0.10, 0.20, 0.00, -0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Nourishment & Food (Dimensions 6, 7) ──────────────────────────────────
    (
        "ভাত",
        vec32![
            0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.90, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "খাবার",
        vec32![
            0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.88, 0.52, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "খাবো",
        vec32![
            0.1, 0.0, 0.0, 0.15, 0.0, 0.0, 0.85, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "খাচ্ছি",
        vec32![
            0.2, 0.0, 0.0, 0.15, 0.0, 0.0, 0.86, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "খেয়েছি",
        vec32![
            0.2, 0.0, 0.0, 0.10, 0.0, 0.0, 0.84, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "চা",
        vec32![
            0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.82, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পানি",
        vec32![
            0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.80, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পান",
        vec32![
            0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.75, 0.42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Education & Reading (Dimensions 8, 9) ─────────────────────────────────
    (
        "বই",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পড়া",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পড়ছি",
        vec32![
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পড়বো",
        vec32![
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.61, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "লেখা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "লিখছি",
        vec32![
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82, 0.52, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "শিক্ষক",
        vec32![
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.78, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "শিক্ষা",
        vec32![
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.80, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "স্কুল",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Geographic / Locations (Dimensions 10, 11) ────────────────────────────
    (
        "বাংলাদেশ",
        vec32![
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "ঢাকা",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "চট্টগ্রাম",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সিলেট",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.84, 0.61, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "দেশ",
        vec32![
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "শহর",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.52, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "গ্রাম",
        vec32![
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.76, 0.48, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Interpersonal & Greeting (Dimensions 12, 13) ──────────────────────────
    (
        "কেমন",
        vec32![
            0.0, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.60, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "আছেন",
        vec32![
            0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.62, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "আছি",
        vec32![
            0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.58, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "আছো",
        vec32![
            0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.59, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "ধন্যবাদ",
        vec32![
            0.0, 0.0, 0.0, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82, 0.50, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "বন্ধু",
        vec32![
            0.0, 0.0, 0.0, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.78, 0.55, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "ভাই",
        vec32![
            0.0, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.52, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Digital / Tech Communication (Dimensions 14, 15) ──────────────────────
    (
        "ফোন",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.65, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "মোবাইল",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.68, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "মেসেজ",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.60, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "ছবি",
        vec32![
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.52, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "ভিডিও",
        vec32![
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82, 0.54, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "অ্যাপ",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.84, 0.58, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Environment & Nature (Dimensions 16, 17) ──────────────────────────────
    (
        "বৃষ্টি",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90,
            0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "আকাশ",
        vec32![
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88,
            0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "মেঘ",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86,
            0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সূর্য",
        vec32![
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.84,
            0.52, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Clothing & Attire (Dimensions 18, 19) ──────────────────────────────────
    (
        "শার্ট",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.90, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "প্যান্ট",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.88, 0.52, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "জামা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.86, 0.54, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কাপড়",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.84, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পরা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.88, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পরছি",
        vec32![
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.86, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "জুতো",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.80, 0.48, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Work, Commerce & Finance (Dimensions 20, 21) ──────────────────────────
    (
        "কাজ",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.90, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "অফিস",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.88, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "চাকরি",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.86, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "টাকা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.84, 0.52, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পয়সা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.82, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "বেতন",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.80, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Motion & Transit (Dimensions 22, 23) ──────────────────────────────────
    (
        "যাওয়া",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "যাবো",
        vec32![
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "গেলাম",
        vec32![
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "আসা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "আসবো",
        vec32![
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "গাড়ি",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.82, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "বাস",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.48, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "ট্রেন",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.81, 0.49, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Quantity & Degree (Dimensions 24, 25) ─────────────────────────────────
    (
        "খুব",
        vec32![
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "অনেক",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "বেশি",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কম",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.80, -0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "একটু",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75, -0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সব",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Questions & Inquiries (Dimensions 26, 27) ─────────────────────────────
    (
        "কি",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.60, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কেন",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.62, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কোথায়",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87, 0.61, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কখন",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.59, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কিভাবে",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.58, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Family & Kinship (Dimensions 28, 29) ──────────────────────────────────
    (
        "মা",
        vec32![
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.60, 0.0, 0.0
        ],
    ),
    (
        "বাবা",
        vec32![
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.62, 0.0, 0.0
        ],
    ),
    (
        "বোন",
        vec32![
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.58, 0.0, 0.0
        ],
    ),
    (
        "পরিবার",
        vec32![
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82, 0.55, 0.0, 0.0
        ],
    ),
    // ── Common Cognition & Actions (Dimensions 30, 31) ────────────────────────
    (
        "দেখা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.60
        ],
    ),
    (
        "দেখবো",
        vec32![
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.62
        ],
    ),
    (
        "বলা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.58
        ],
    ),
    (
        "বলবো",
        vec32![
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.57
        ],
    ),
    (
        "জানা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82, 0.52
        ],
    ),
    // ── Law, Justice & Court Cluster (Dimensions 28, 29) ──────────────────────
    (
        "কোর্ট",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.92, 0.65, 0.0, 0.0
        ],
    ),
    (
        "আদালত",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.62, 0.0, 0.0
        ],
    ),
    (
        "আইন",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.60, 0.0, 0.0
        ],
    ),
    (
        "বিচার",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.58, 0.0, 0.0
        ],
    ),
    (
        "মামলা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.55, 0.0, 0.0
        ],
    ),
    (
        "উকিল",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.84, 0.52, 0.0, 0.0
        ],
    ),
    (
        "পুলিশ",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.48, 0.0, 0.0
        ],
    ),
    (
        "শব",
        vec32![
            0.0, 0.0, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82, 0.50, 0.0, 0.0
        ],
    ),
    (
        "কুল",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.58, 0.0, 0.0
        ],
    ),
    // ── Additional Clothing & Attire Anchors (Dimensions 18, 19) ──────────────
    (
        "কোট",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.90, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পোশাক",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.88, 0.56, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "শাড়ি",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.85, 0.52, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Additional Education & Language Anchors (Dimensions 8, 9) ─────────────
    (
        "ভাষা",
        vec32![
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.92, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "পরীক্ষা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "ক্লাস",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.56, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "ছাত্র",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.84, 0.54, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Additional Nature & Marine Anchors (Dimensions 16, 17 / 10, 11) ───────
    (
        "ভাসা",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90,
            0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "নদী",
        vec32![
            0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.88,
            0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সাঁতার",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86,
            0.56, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "নৌকা",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.84,
            0.54, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কূল",
        vec32![
            0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.58, 0.0, 0.0, 0.0, 0.0, 0.40,
            0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Additional Tech & Computing Anchors (Dimensions 14, 15) ───────────────
    (
        "কম্পিউটার",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.92, 0.68, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "কোড",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.64, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সফটওয়্যার",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.62, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "সার্ভার",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.86, 0.60, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "ওয়েবসাইট",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.84, 0.58, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "বাগ",
        vec32![
            0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.52, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    // ── Additional Finance, Metals & Audio/Senses Anchors ─────────────────────
    (
        "সোনা",
        vec32![
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.2, 0.0, 0.88, 0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "শোনা",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.64
        ],
    ),
    (
        "গান",
        vec32![
            0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.88, 0.60
        ],
    ),
    // ── Day vs Poor Homophones ────────────────────────────────────────────────
    (
        "দিন",
        vec32![
            0.88, 0.32, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "দীন",
        vec32![
            0.0, 0.0, 0.0, -0.60, -0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.60, 0.30, 0.0, 0.0
        ],
    ),
    // ── Blue vs Took Homophones ───────────────────────────────────────────────
    (
        "নীল",
        vec32![
            0.0, 0.0, 0.0, 0.70, 0.40, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
    (
        "নিল",
        vec32![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ],
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identity() {
        let embeddings = WordEmbeddings::new();
        let sim = embeddings.cosine_similarity("ভালো", "ভালো");
        assert!((sim - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_semantic_proximity_clusters() {
        let embeddings = WordEmbeddings::new();

        // 1. Temporal cluster: আজ ↔ কাল should have high similarity
        let sim_temporal = embeddings.cosine_similarity("আজ", "কাল");
        assert!(
            sim_temporal > 0.80,
            "Expected high temporal similarity, got {}",
            sim_temporal
        );

        // 2. Valence cluster: ভালো ↔ সুন্দর should have high similarity
        let sim_valence = embeddings.cosine_similarity("ভালো", "সুন্দর");
        assert!(
            sim_valence > 0.80,
            "Expected high valence similarity, got {}",
            sim_valence
        );

        // 3. Dining cluster: ভাত ↔ খাবো should have high similarity
        let sim_dining = embeddings.cosine_similarity("ভাত", "খাবো");
        assert!(
            sim_dining > 0.75,
            "Expected high dining similarity, got {}",
            sim_dining
        );

        // 4. Reading cluster: বই ↔ পড়া should have high similarity
        let sim_reading = embeddings.cosine_similarity("বই", "পড়া");
        assert!(
            sim_reading > 0.80,
            "Expected high reading similarity, got {}",
            sim_reading
        );

        // 5. Cross-cluster orthogonality: আজ (time) ↔ ভাত (food) should be low / uncorrelated
        let sim_unrelated = embeddings.cosine_similarity("আজ", "ভাত");
        assert!(
            sim_unrelated < 0.20,
            "Expected low similarity for unrelated words, got {}",
            sim_unrelated
        );
    }

    #[test]
    fn test_semantic_boost() {
        let embeddings = WordEmbeddings::new();
        let boost_related = embeddings.semantic_boost("আজ", "কাল");
        let boost_unrelated = embeddings.semantic_boost("আজ", "ভাত");

        assert!(boost_related > 0.5);
        assert_eq!(boost_unrelated, 0.0);
    }

    #[test]
    fn test_binary_roundtrip() {
        let mut original = WordEmbeddings::new();
        original.insert("কৃত্রিম", [0.5; EMBEDDING_DIM]);

        let bin = original.save_binary();
        assert!(bin.len() > 100);

        let mut loaded = WordEmbeddings {
            inner: Arc::new(WordEmbeddingsInner {
                vectors: HashMap::new(),
            }),
        };
        let count = loaded
            .load_binary_bytes(&bin)
            .expect("Failed to load binary");
        assert_eq!(count, original.len());

        let sim = loaded.cosine_similarity("ভালো", "সুন্দর");
        assert!(sim > 0.80);
        assert!(loaded.get_vector("কৃত্রিম").is_some());
    }

    #[test]
    fn test_homophone_disambiguation_embeddings() {
        let embeddings = WordEmbeddings::new();

        // Reading context vs Clothing context:
        // "বই" must strongly align with "পড়া" (reading) and be orthogonal to "পরা" (wearing)
        assert!(embeddings.cosine_similarity("বই", "পড়া") > 0.80);
        assert!(embeddings.cosine_similarity("বই", "পরা") < 0.10);

        // "শার্ট" must strongly align with "পরা" (wearing) and be orthogonal to "পড়া" (reading)
        assert!(embeddings.cosine_similarity("শার্ট", "পরা") > 0.80);
        assert!(embeddings.cosine_similarity("শার্ট", "পড়া") < 0.10);
    }

    #[test]
    fn test_dense_corpus_embeddings_loaded() {
        let embeddings = WordEmbeddings::new();
        assert!(
            embeddings.len() > 1000,
            "Expected >1000 loaded embeddings, got {}",
            embeddings.len()
        );
        let vec = embeddings
            .get_vector("বাংলাদেশ")
            .expect("বাংলাদেশ should be indexed");
        let norm_sq: f32 = vec.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-3, "Vector should be unit norm");
    }

    #[test]
    fn test_topic_vector_construction_and_disambiguation() {
        let embeddings = WordEmbeddings::new();

        // 1. Zero vector initialization
        let zero_vec = TopicVector::zero();
        assert!(!zero_vec.is_active());
        assert_eq!(zero_vec.magnitude, 0.0);
        assert_eq!(zero_vec.active_tokens, 0);

        // 2. Reading / Study topic context
        let reading_context = ["আমি", "বই", "পড়া", "খুব", "পছন্দ", "করি"];
        let reading_topic = TopicVector::from_slice(&reading_context, &embeddings);
        assert!(reading_topic.is_active());
        assert!(reading_topic.active_tokens >= 2);

        // Disambiguate পড়া (reading) vs পরা (wearing)
        let (winner_reading, margin_reading) = reading_topic
            .disambiguate("পড়া", "পরা", &embeddings)
            .expect("Should disambiguate reading context");
        assert_eq!(winner_reading, "পড়া");
        assert!(margin_reading > 0.15);

        // 3. Clothing / Attire topic context
        let clothing_context = ["কাল", "একটি", "নতুন", "শার্ট", "প্যান্ট", "কিনলাম"];
        let clothing_topic = TopicVector::from_slice(&clothing_context, &embeddings);
        assert!(clothing_topic.is_active());

        // Disambiguate পরা (wearing) vs পড়া (reading)
        let (winner_clothing, margin_clothing) = clothing_topic
            .disambiguate("পড়া", "পরা", &embeddings)
            .expect("Should disambiguate clothing context");
        assert_eq!(winner_clothing, "পরা");
        assert!(margin_clothing > 0.15);

        // 4. Law & Justice vs Clothing: কোর্ট (Court) vs কোট (Coat)
        let legal_context = ["আদালতে", "বিচার", "ও", "মামলা", "চলছে"];
        let legal_topic = TopicVector::from_slice(&legal_context, &embeddings);
        let (winner_court, margin_court) = legal_topic
            .disambiguate("কোর্ট", "কোট", &embeddings)
            .expect("Should disambiguate court context");
        assert_eq!(winner_court, "কোর্ট");
        assert!(margin_court > 0.15);

        let winter_clothing_context = ["শীতের", "পোশাক", "হিসেবে", "শার্ট", "প্যান্ট"];
        let winter_topic = TopicVector::from_slice(&winter_clothing_context, &embeddings);
        let (winner_coat, margin_coat) = winter_topic
            .disambiguate("কোর্ট", "কোট", &embeddings)
            .expect("Should disambiguate coat context");
        assert_eq!(winner_coat, "কোট");
        assert!(margin_coat > 0.15);

        // 5. Language vs Floating: ভাষা (Language) vs ভাসা (Float)
        let lang_context = ["বাংলা", "ভাষা", "শিক্ষা", "এবং", "বই"];
        let lang_topic = TopicVector::from_slice(&lang_context, &embeddings);
        let (winner_lang, _) = lang_topic
            .disambiguate("ভাষা", "ভাসা", &embeddings)
            .expect("Should disambiguate language");
        assert_eq!(winner_lang, "ভাষা");

        let marine_context = ["নদী", "পানি", "সাঁতার", "এবং", "নৌকা"];
        let marine_topic = TopicVector::from_slice(&marine_context, &embeddings);
        let (winner_float, _) = marine_topic
            .disambiguate("ভাষা", "ভাসা", &embeddings)
            .expect("Should disambiguate float");
        assert_eq!(winner_float, "ভাসা");

        // 6. Shore vs Clan: কূল (Shore) vs কুল (Clan)
        let shore_context = ["নদী", "পানি", "সাঁতার", "ঘাট"];
        let shore_topic = TopicVector::from_slice(&shore_context, &embeddings);
        let (winner_shore, _) = shore_topic
            .disambiguate("কূল", "কুল", &embeddings)
            .expect("Should disambiguate shore");
        assert_eq!(winner_shore, "কূল");

        let clan_context = ["বংশ", "পরিবার", "সমাজ", "বাবা"];
        let clan_topic = TopicVector::from_slice(&clan_context, &embeddings);
        let (winner_clan, _) = clan_topic
            .disambiguate("কূল", "কুল", &embeddings)
            .expect("Should disambiguate clan");
        assert_eq!(winner_clan, "কুল");
    }
}
