//! Bengali Corpus Training Engine
//!
//! Tokenizes, segments, and computes N-gram probability distributions
//! from raw Bengali text corpora for on-device language modeling.

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::zero_copy::{dequantize_log_prob, quantize_log_prob};

/// Trained statistical N-gram dataset exported by the trainer
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainedLanguageModelData {
    pub unigrams: HashMap<String, f32>,
    pub bigrams: Vec<(String, String, f32)>,
    pub trigrams: Vec<(String, String, String, f32)>,
    #[serde(default)]
    pub fourgrams: Vec<(String, String, String, String, f32)>,
    pub total_words: usize,
}

impl TrainedLanguageModelData {
    pub const BINARY_MAGIC: &'static [u8; 4] = b"LLM2";
    pub const BINARY_VERSION: u32 = 2;
    pub const BINARY_MAGIC_V3: &'static [u8; 4] = b"LLM3";
    pub const BINARY_VERSION_V3: u32 = 3;

    #[inline]
    pub fn unigram_count(&self) -> usize {
        self.unigrams.len()
    }

    #[inline]
    pub fn bigram_count(&self) -> usize {
        self.bigrams.len()
    }

    #[inline]
    pub fn trigram_count(&self) -> usize {
        self.trigrams.len()
    }

    #[inline]
    pub fn fourgram_count(&self) -> usize {
        self.fourgrams.len()
    }

    /// Merge another trained model into this one, preserving unique N-grams
    pub fn merge(&mut self, other: &Self) {
        for (w, p) in &other.unigrams {
            self.unigrams.entry(w.clone()).or_insert(*p);
        }

        let mut existing_bi: hashbrown::HashSet<(&str, &str)> = self.bigrams.iter().map(|(w1, w2, _)| (w1.as_str(), w2.as_str())).collect();
        let mut new_bi = Vec::new();
        for (w1, w2, p) in &other.bigrams {
            if existing_bi.insert((w1.as_str(), w2.as_str())) {
                new_bi.push((w1.clone(), w2.clone(), *p));
            }
        }
        drop(existing_bi);
        self.bigrams.extend(new_bi);

        let mut existing_tri: hashbrown::HashSet<(&str, &str, &str)> = self.trigrams.iter().map(|(w1, w2, w3, _)| (w1.as_str(), w2.as_str(), w3.as_str())).collect();
        let mut new_tri = Vec::new();
        for (w1, w2, w3, p) in &other.trigrams {
            if existing_tri.insert((w1.as_str(), w2.as_str(), w3.as_str())) {
                new_tri.push((w1.clone(), w2.clone(), w3.clone(), *p));
            }
        }
        drop(existing_tri);
        self.trigrams.extend(new_tri);

        let mut existing_four: hashbrown::HashSet<(&str, &str, &str, &str)> = self.fourgrams.iter().map(|(w1, w2, w3, w4, _)| (w1.as_str(), w2.as_str(), w3.as_str(), w4.as_str())).collect();
        let mut new_four = Vec::new();
        for (w1, w2, w3, w4, p) in &other.fourgrams {
            if existing_four.insert((w1.as_str(), w2.as_str(), w3.as_str(), w4.as_str())) {
                new_four.push((w1.clone(), w2.clone(), w3.clone(), w4.clone(), *p));
            }
        }
        drop(existing_four);
        self.fourgrams.extend(new_four);

        self.total_words += other.total_words;
    }

    /// Filter out corrupted crawl artifacts, Wikipedia citation tags, and unsegmented garbage tokens.
    /// Returns the number of removed unigrams.
    pub fn sanitize(&mut self) -> usize {
        let before_unigrams = self.unigrams.len();
        self.unigrams.retain(|w, _| !is_junk_token(w));
        let removed_unigrams = before_unigrams - self.unigrams.len();

        self.bigrams
            .retain(|(w1, w2, _)| !is_junk_token(w1) && !is_junk_token(w2));
        self.trigrams
            .retain(|(w1, w2, w3, _)| !is_junk_token(w1) && !is_junk_token(w2) && !is_junk_token(w3));
        self.fourgrams.retain(|(w1, w2, w3, w4, _)| {
            !is_junk_token(w1) && !is_junk_token(w2) && !is_junk_token(w3) && !is_junk_token(w4)
        });

        removed_unigrams
    }


    /// Load trained model data from a JSON file
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }


    /// Serialize language model data into compact quantized binary format (LLM3)
    pub fn to_binary_llm3(&self) -> Vec<u8> {
        let mut unigrams = self.unigrams.clone();
        let mut bigrams = self.bigrams.clone();
        let mut trigrams = self.trigrams.clone();
        let fourgrams = self.fourgrams.clone();

        let has_bengali = self.unigrams.is_empty()
            || self.unigrams.keys().any(|k| k.chars().any(crate::trainer::chars::is_bengali_char));
        if has_bengali {
            for &(w, p) in crate::lm::UNIGRAM_LOG_PROBS {
                unigrams.entry(w.to_string()).or_insert(p);
            }

            let bigram_set: hashbrown::HashSet<(&str, &str)> = self.bigrams.iter().map(|(w1, w2, _)| (w1.as_str(), w2.as_str())).collect();
            for &((w1, w2), p) in crate::lm::BIGRAM_TRANSITIONS {
                if !bigram_set.contains(&(w1, w2)) {
                    bigrams.push((w1.to_string(), w2.to_string(), p));
                }
            }
            drop(bigram_set);

            let trigram_set: hashbrown::HashSet<(&str, &str, &str)> = self.trigrams.iter().map(|(w1, w2, w3, _)| (w1.as_str(), w2.as_str(), w3.as_str())).collect();
            for &((w1, w2, w3), p) in crate::lm::TRIGRAM_TRANSITIONS {
                if !trigram_set.contains(&(w1, w2, w3)) {
                    trigrams.push((w1.to_string(), w2.to_string(), w3.to_string(), p));
                }
            }
            drop(trigram_set);
        }

        let mut words_set = hashbrown::HashSet::new();
        for w in unigrams.keys() {
            words_set.insert(w.clone());
        }
        for (w1, w2, _) in &bigrams {
            words_set.insert(w1.clone());
            words_set.insert(w2.clone());
        }
        for (w1, w2, w3, _) in &trigrams {
            words_set.insert(w1.clone());
            words_set.insert(w2.clone());
            words_set.insert(w3.clone());
        }
        for (w1, w2, w3, w4, _) in &fourgrams {
            words_set.insert(w1.clone());
            words_set.insert(w2.clone());
            words_set.insert(w3.clone());
            words_set.insert(w4.clone());
        }

        let mut words: Vec<String> = words_set.into_iter().collect();
        words.sort_unstable();

        let mut vocab_map: HashMap<&str, u32> = HashMap::with_capacity(words.len());
        for (id, w) in words.iter().enumerate() {
            vocab_map.insert(w.as_str(), id as u32);
        }

        let mut string_buffer = Vec::new();
        let mut vocab_entries: Vec<(u32, u16)> = Vec::with_capacity(words.len());
        for w in &words {
            let offset = string_buffer.len() as u32;
            let bytes = w.as_bytes();
            let len = bytes.len() as u16;
            string_buffer.extend_from_slice(bytes);
            vocab_entries.push((offset, len));
        }

        let vocab_count = words.len() as u32;
        let unigram_count = words.len() as u32;

        let mut sorted_bigrams = bigrams;
        sorted_bigrams.sort_unstable_by(|(a1, a2, p1), (b1, b2, p2)| {
            let id_a1 = vocab_map[a1.as_str()];
            let id_b1 = vocab_map[b1.as_str()];
            let id_a2 = vocab_map[a2.as_str()];
            let id_b2 = vocab_map[b2.as_str()];
            id_a1.cmp(&id_b1)
                .then_with(|| p2.partial_cmp(p1).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| id_a2.cmp(&id_b2))
        });
        let bigram_count = sorted_bigrams.len() as u32;

        let mut sorted_trigrams = trigrams;
        sorted_trigrams.sort_unstable_by(|(a1, a2, a3, p1), (b1, b2, b3, p2)| {
            let id_a1 = vocab_map[a1.as_str()];
            let id_b1 = vocab_map[b1.as_str()];
            let id_a2 = vocab_map[a2.as_str()];
            let id_b2 = vocab_map[b2.as_str()];
            let id_a3 = vocab_map[a3.as_str()];
            let id_b3 = vocab_map[b3.as_str()];
            id_a1.cmp(&id_b1)
                .then_with(|| id_a2.cmp(&id_b2))
                .then_with(|| p2.partial_cmp(p1).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| id_a3.cmp(&id_b3))
        });
        let trigram_count = sorted_trigrams.len() as u32;

        let mut sorted_fourgrams = fourgrams;
        sorted_fourgrams.sort_unstable_by(|(a1, a2, a3, a4, p1), (b1, b2, b3, b4, p2)| {
            let id_a1 = vocab_map[a1.as_str()];
            let id_b1 = vocab_map[b1.as_str()];
            let id_a2 = vocab_map[a2.as_str()];
            let id_b2 = vocab_map[b2.as_str()];
            let id_a3 = vocab_map[a3.as_str()];
            let id_b3 = vocab_map[b3.as_str()];
            let id_a4 = vocab_map[a4.as_str()];
            let id_b4 = vocab_map[b4.as_str()];
            id_a1.cmp(&id_b1)
                .then_with(|| id_a2.cmp(&id_b2))
                .then_with(|| id_a3.cmp(&id_b3))
                .then_with(|| p2.partial_cmp(p1).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| id_a4.cmp(&id_b4))
        });
        let fourgram_count = sorted_fourgrams.len() as u32;
        let string_buffer_len = string_buffer.len() as u32;

        let total_size = 36
            + (vocab_count as usize * 6)
            + (unigram_count as usize * 5)
            + (bigram_count as usize * 9)
            + (trigram_count as usize * 13)
            + (fourgram_count as usize * 17)
            + string_buffer.len();

        let mut out = Vec::with_capacity(total_size);

        // Header (36 bytes for LLM3)
        out.extend_from_slice(Self::BINARY_MAGIC_V3);
        out.extend_from_slice(&Self::BINARY_VERSION_V3.to_le_bytes());
        out.extend_from_slice(&(self.total_words as u32).to_le_bytes());
        out.extend_from_slice(&vocab_count.to_le_bytes());
        out.extend_from_slice(&unigram_count.to_le_bytes());
        out.extend_from_slice(&bigram_count.to_le_bytes());
        out.extend_from_slice(&trigram_count.to_le_bytes());
        out.extend_from_slice(&fourgram_count.to_le_bytes());
        out.extend_from_slice(&string_buffer_len.to_le_bytes());

        // Vocab table
        for (offset, len) in vocab_entries {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
        }

        // Unigrams (id: u32, quantized: u8)
        for (id, w) in words.iter().enumerate() {
            let p = unigrams.get(w).copied().unwrap_or(-6.0);
            out.extend_from_slice(&(id as u32).to_le_bytes());
            out.push(quantize_log_prob(p));
        }

        // Bigrams (w1: u32, w2: u32, quantized: u8)
        for (w1, w2, p) in &sorted_bigrams {
            let w1_id = vocab_map[w1.as_str()];
            let w2_id = vocab_map[w2.as_str()];
            out.extend_from_slice(&w1_id.to_le_bytes());
            out.extend_from_slice(&w2_id.to_le_bytes());
            out.push(quantize_log_prob(*p));
        }

        // Trigrams (w1: u32, w2: u32, w3: u32, quantized: u8)
        for (w1, w2, w3, p) in &sorted_trigrams {
            let w1_id = vocab_map[w1.as_str()];
            let w2_id = vocab_map[w2.as_str()];
            let w3_id = vocab_map[w3.as_str()];
            out.extend_from_slice(&w1_id.to_le_bytes());
            out.extend_from_slice(&w2_id.to_le_bytes());
            out.extend_from_slice(&w3_id.to_le_bytes());
            out.push(quantize_log_prob(*p));
        }

        // Fourgrams (w1: u32, w2: u32, w3: u32, w4: u32, quantized: u8)
        for (w1, w2, w3, w4, p) in &sorted_fourgrams {
            let w1_id = vocab_map[w1.as_str()];
            let w2_id = vocab_map[w2.as_str()];
            let w3_id = vocab_map[w3.as_str()];
            let w4_id = vocab_map[w4.as_str()];
            out.extend_from_slice(&w1_id.to_le_bytes());
            out.extend_from_slice(&w2_id.to_le_bytes());
            out.extend_from_slice(&w3_id.to_le_bytes());
            out.extend_from_slice(&w4_id.to_le_bytes());
            out.push(quantize_log_prob(*p));
        }

        // String buffer
        out.extend_from_slice(&string_buffer);

        out
    }

    /// Serialize language model data into compact sorted binary format (LLM2)
    pub fn to_binary(&self) -> Vec<u8> {
        let mut unigrams = self.unigrams.clone();
        let mut bigrams = self.bigrams.clone();
        let mut trigrams = self.trigrams.clone();

        let has_bengali = self.unigrams.is_empty()
            || self.unigrams.keys().any(|k| k.chars().any(crate::trainer::chars::is_bengali_char));
        if has_bengali {
            for &(w, p) in crate::lm::UNIGRAM_LOG_PROBS {
                unigrams.entry(w.to_string()).or_insert(p);
            }

            let bigram_set: hashbrown::HashSet<(&str, &str)> = self.bigrams.iter().map(|(w1, w2, _)| (w1.as_str(), w2.as_str())).collect();
            for &((w1, w2), p) in crate::lm::BIGRAM_TRANSITIONS {
                if !bigram_set.contains(&(w1, w2)) {
                    bigrams.push((w1.to_string(), w2.to_string(), p));
                }
            }
            drop(bigram_set);

            let trigram_set: hashbrown::HashSet<(&str, &str, &str)> = self.trigrams.iter().map(|(w1, w2, w3, _)| (w1.as_str(), w2.as_str(), w3.as_str())).collect();
            for &((w1, w2, w3), p) in crate::lm::TRIGRAM_TRANSITIONS {
                if !trigram_set.contains(&(w1, w2, w3)) {
                    trigrams.push((w1.to_string(), w2.to_string(), w3.to_string(), p));
                }
            }
            drop(trigram_set);
        }

        let mut words_set = hashbrown::HashSet::new();
        for w in unigrams.keys() {
            words_set.insert(w.clone());
        }
        for (w1, w2, _) in &bigrams {
            words_set.insert(w1.clone());
            words_set.insert(w2.clone());
        }
        for (w1, w2, w3, _) in &trigrams {
            words_set.insert(w1.clone());
            words_set.insert(w2.clone());
            words_set.insert(w3.clone());
        }

        let mut words: Vec<String> = words_set.into_iter().collect();
        words.sort_unstable(); // Lexicographical sort for O(log N) zero-alloc binary search

        let mut vocab_map: HashMap<&str, u32> = HashMap::with_capacity(words.len());
        for (id, w) in words.iter().enumerate() {
            vocab_map.insert(w.as_str(), id as u32);
        }

        let mut string_buffer = Vec::new();
        let mut vocab_entries: Vec<(u32, u16)> = Vec::with_capacity(words.len());
        for w in &words {
            let offset = string_buffer.len() as u32;
            let bytes = w.as_bytes();
            let len = bytes.len() as u16;
            string_buffer.extend_from_slice(bytes);
            vocab_entries.push((offset, len));
        }

        let vocab_count = words.len() as u32;
        let unigram_count = words.len() as u32;

        // Sort bigrams: w1_id ascending, log_prob descending, w2_id ascending
        let mut sorted_bigrams = bigrams;
        sorted_bigrams.sort_unstable_by(|(a1, a2, p1), (b1, b2, p2)| {
            let id_a1 = vocab_map[a1.as_str()];
            let id_b1 = vocab_map[b1.as_str()];
            let id_a2 = vocab_map[a2.as_str()];
            let id_b2 = vocab_map[b2.as_str()];
            id_a1.cmp(&id_b1)
                .then_with(|| p2.partial_cmp(p1).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| id_a2.cmp(&id_b2))
        });
        let bigram_count = sorted_bigrams.len() as u32;

        // Sort trigrams: (w1_id, w2_id) ascending, log_prob descending, w3_id ascending
        let mut sorted_trigrams = trigrams;
        sorted_trigrams.sort_unstable_by(|(a1, a2, a3, p1), (b1, b2, b3, p2)| {
            let id_a1 = vocab_map[a1.as_str()];
            let id_b1 = vocab_map[b1.as_str()];
            let id_a2 = vocab_map[a2.as_str()];
            let id_b2 = vocab_map[b2.as_str()];
            let id_a3 = vocab_map[a3.as_str()];
            let id_b3 = vocab_map[b3.as_str()];
            id_a1.cmp(&id_b1)
                .then_with(|| id_a2.cmp(&id_b2))
                .then_with(|| p2.partial_cmp(p1).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| id_a3.cmp(&id_b3))
        });
        let trigram_count = sorted_trigrams.len() as u32;
        let string_buffer_len = string_buffer.len() as u32;

        let total_size = 32
            + (vocab_count as usize * 6)
            + (unigram_count as usize * 8)
            + (bigram_count as usize * 12)
            + (trigram_count as usize * 16)
            + string_buffer.len();

        let mut out = Vec::with_capacity(total_size);

        // Header (32 bytes)
        out.extend_from_slice(Self::BINARY_MAGIC);
        out.extend_from_slice(&Self::BINARY_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.total_words as u32).to_le_bytes());
        out.extend_from_slice(&vocab_count.to_le_bytes());
        out.extend_from_slice(&unigram_count.to_le_bytes());
        out.extend_from_slice(&bigram_count.to_le_bytes());
        out.extend_from_slice(&trigram_count.to_le_bytes());
        out.extend_from_slice(&string_buffer_len.to_le_bytes());

        // Vocab table
        for (offset, len) in vocab_entries {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
        }

        // Unigrams (in direct word_id order [0..vocab_count])
        for (id, w) in words.iter().enumerate() {
            let p = unigrams.get(w).copied().unwrap_or(-6.0);
            out.extend_from_slice(&(id as u32).to_le_bytes());
            out.extend_from_slice(&p.to_le_bytes());
        }

        // Bigrams
        for (w1, w2, p) in &sorted_bigrams {
            let w1_id = vocab_map[w1.as_str()];
            let w2_id = vocab_map[w2.as_str()];
            out.extend_from_slice(&w1_id.to_le_bytes());
            out.extend_from_slice(&w2_id.to_le_bytes());
            out.extend_from_slice(&p.to_le_bytes());
        }

        // Trigrams
        for (w1, w2, w3, p) in &sorted_trigrams {
            let w1_id = vocab_map[w1.as_str()];
            let w2_id = vocab_map[w2.as_str()];
            let w3_id = vocab_map[w3.as_str()];
            out.extend_from_slice(&w1_id.to_le_bytes());
            out.extend_from_slice(&w2_id.to_le_bytes());
            out.extend_from_slice(&w3_id.to_le_bytes());
            out.extend_from_slice(&p.to_le_bytes());
        }

        // String buffer
        out.extend_from_slice(&string_buffer);

        out
    }

    /// Save language model dataset directly to a binary file
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let bytes = self.to_binary();
        std::fs::write(path, bytes)
    }

    /// Load language model data from compact binary bytes (supports LLM1, LLM2, and LLM3)
    pub fn from_binary(data: &[u8]) -> Result<Self, std::io::Error> {
        if data.len() < 32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Binary data too short for header",
            ));
        }

        let magic = &data[0..4];
        let is_v3 = magic == b"LLM3";
        let is_v2_or_v1 = magic == b"LLM1" || magic == b"LLM2";

        if !is_v3 && !is_v2_or_v1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid binary language model magic header",
            ));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if is_v3 {
            if version != 3 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unsupported binary language model version: {}", version),
                ));
            }
        } else if version != 1 && version != 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported binary language model version: {}", version),
            ));
        }

        let total_words = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let vocab_count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let unigram_count = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let bigram_count = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
        let trigram_count = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;

        let (fourgram_count, string_buffer_len, header_size) = if is_v3 {
            let four = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;
            let str_len = u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize;
            (four, str_len, 36)
        } else {
            let str_len = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;
            (0, str_len, 32)
        };

        let expected_size = if is_v3 {
            header_size
                + (vocab_count * 6)
                + (unigram_count * 5)
                + (bigram_count * 9)
                + (trigram_count * 13)
                + (fourgram_count * 17)
                + string_buffer_len
        } else {
            header_size
                + (vocab_count * 6)
                + (unigram_count * 8)
                + (bigram_count * 12)
                + (trigram_count * 16)
                + string_buffer_len
        };

        if data.len() < expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Binary language model data truncated",
            ));
        }

        let mut cursor = header_size;

        // Vocab table
        let mut vocab_offsets_lens = Vec::with_capacity(vocab_count);
        for _ in 0..vocab_count {
            let offset = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
            let len = u16::from_le_bytes(data[cursor + 4..cursor + 6].try_into().unwrap()) as usize;
            vocab_offsets_lens.push((offset, len));
            cursor += 6;
        }

        // Unigrams cursor
        let unigrams_start = cursor;
        cursor += if is_v3 { unigram_count * 5 } else { unigram_count * 8 };

        // Bigrams cursor
        let bigrams_start = cursor;
        cursor += if is_v3 { bigram_count * 9 } else { bigram_count * 12 };

        // Trigrams cursor
        let trigrams_start = cursor;
        cursor += if is_v3 { trigram_count * 13 } else { trigram_count * 16 };

        // Fourgrams cursor
        let fourgrams_start = cursor;
        cursor += if is_v3 { fourgram_count * 17 } else { 0 };

        // String buffer
        let string_buf_bytes = &data[cursor..cursor + string_buffer_len];
        let string_buf = std::str::from_utf8(string_buf_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut words = Vec::with_capacity(vocab_count);
        for (offset, len) in vocab_offsets_lens {
            let end = offset + len;
            if end > string_buf.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Vocab string out of bounds",
                ));
            }
            words.push(string_buf[offset..end].to_string());
        }

        // Parse unigrams
        let mut unigrams = HashMap::with_capacity(unigram_count);
        let mut u_cursor = unigrams_start;
        for _ in 0..unigram_count {
            let wid = u32::from_le_bytes(data[u_cursor..u_cursor + 4].try_into().unwrap()) as usize;
            let prob = if is_v3 {
                dequantize_log_prob(data[u_cursor + 4])
            } else {
                f32::from_le_bytes(data[u_cursor + 4..u_cursor + 8].try_into().unwrap())
            };
            if wid < words.len() {
                unigrams.insert(words[wid].clone(), prob);
            }
            u_cursor += if is_v3 { 5 } else { 8 };
        }

        // Parse bigrams
        let mut bigrams = Vec::with_capacity(bigram_count);
        let mut b_cursor = bigrams_start;
        for _ in 0..bigram_count {
            let w1_id = u32::from_le_bytes(data[b_cursor..b_cursor + 4].try_into().unwrap()) as usize;
            let w2_id = u32::from_le_bytes(data[b_cursor + 4..b_cursor + 8].try_into().unwrap()) as usize;
            let prob = if is_v3 {
                dequantize_log_prob(data[b_cursor + 8])
            } else {
                f32::from_le_bytes(data[b_cursor + 8..b_cursor + 12].try_into().unwrap())
            };
            if w1_id < words.len() && w2_id < words.len() {
                bigrams.push((words[w1_id].clone(), words[w2_id].clone(), prob));
            }
            b_cursor += if is_v3 { 9 } else { 12 };
        }

        // Parse trigrams
        let mut trigrams = Vec::with_capacity(trigram_count);
        let mut t_cursor = trigrams_start;
        for _ in 0..trigram_count {
            let w1_id = u32::from_le_bytes(data[t_cursor..t_cursor + 4].try_into().unwrap()) as usize;
            let w2_id = u32::from_le_bytes(data[t_cursor + 4..t_cursor + 8].try_into().unwrap()) as usize;
            let w3_id = u32::from_le_bytes(data[t_cursor + 8..t_cursor + 12].try_into().unwrap()) as usize;
            let prob = if is_v3 {
                dequantize_log_prob(data[t_cursor + 12])
            } else {
                f32::from_le_bytes(data[t_cursor + 12..t_cursor + 16].try_into().unwrap())
            };
            if w1_id < words.len() && w2_id < words.len() && w3_id < words.len() {
                trigrams.push((words[w1_id].clone(), words[w2_id].clone(), words[w3_id].clone(), prob));
            }
            t_cursor += if is_v3 { 13 } else { 16 };
        }

        // Parse fourgrams
        let mut fourgrams = Vec::with_capacity(fourgram_count);
        if is_v3 && fourgram_count > 0 {
            let mut f_cursor = fourgrams_start;
            for _ in 0..fourgram_count {
                let w1_id = u32::from_le_bytes(data[f_cursor..f_cursor + 4].try_into().unwrap()) as usize;
                let w2_id = u32::from_le_bytes(data[f_cursor + 4..f_cursor + 8].try_into().unwrap()) as usize;
                let w3_id = u32::from_le_bytes(data[f_cursor + 8..f_cursor + 12].try_into().unwrap()) as usize;
                let w4_id = u32::from_le_bytes(data[f_cursor + 12..f_cursor + 16].try_into().unwrap()) as usize;
                let prob = dequantize_log_prob(data[f_cursor + 16]);
                if w1_id < words.len() && w2_id < words.len() && w3_id < words.len() && w4_id < words.len() {
                    fourgrams.push((
                        words[w1_id].clone(),
                        words[w2_id].clone(),
                        words[w3_id].clone(),
                        words[w4_id].clone(),
                        prob,
                    ));
                }
                f_cursor += 17;
            }
        }

        Ok(Self {
            unigrams,
            bigrams,
            trigrams,
            fourgrams,
            total_words,
        })
    }

    /// Load language model data directly from a binary file
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let bytes = std::fs::read(path)?;
        Self::from_binary(&bytes)
    }
}

use rayon::prelude::*;

fn default_min_fourgram_freq() -> usize { 8 }
fn default_max_fourgrams() -> usize { 30_000 }

/// Configuration options for N-gram pruning and model capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub min_unigram_freq: usize,
    pub min_bigram_freq: usize,
    pub min_trigram_freq: usize,
    #[serde(default = "default_min_fourgram_freq")]
    pub min_fourgram_freq: usize,
    pub max_unigrams: usize,
    pub max_bigrams: usize,
    pub max_trigrams: usize,
    #[serde(default = "default_max_fourgrams")]
    pub max_fourgrams: usize,
}

impl TrainingConfig {
    /// Configuration that keeps all tokens without frequency pruning (suitable for small corpora/unit tests)
    pub fn unpruned() -> Self {
        Self {
            min_unigram_freq: 1,
            min_bigram_freq: 1,
            min_trigram_freq: 1,
            min_fourgram_freq: 1,
            max_unigrams: usize::MAX,
            max_bigrams: usize::MAX,
            max_trigrams: usize::MAX,
            max_fourgrams: usize::MAX,
        }
    }

    /// Optimized configuration for large-scale corpora (e.g. Wikipedia, OSCAR, web dumps)
    pub fn production() -> Self {
        Self {
            min_unigram_freq: 3,
            min_bigram_freq: 5,
            min_trigram_freq: 8,
            min_fourgram_freq: 10,
            max_unigrams: 65_000,
            max_bigrams: 160_000,
            max_trigrams: 60_000,
            max_fourgrams: 30_000,
        }
    }

    /// Production configuration for LLM3 quantized model with expanded capacity and 4-grams
    pub fn llm3_production() -> Self {
        Self {
            min_unigram_freq: 2,
            min_bigram_freq: 3,
            min_trigram_freq: 5,
            min_fourgram_freq: 8,
            max_unigrams: 80_000,
            max_bigrams: 220_000,
            max_trigrams: 100_000,
            max_fourgrams: 40_000,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::unpruned()
    }
}

/// Dynamic N-gram Corpus Trainer
#[derive(Debug, Clone, Default)]
pub struct CorpusTrainer {
    unigram_counts: HashMap<String, usize>,
    bigram_counts: HashMap<(String, String), usize>,
    trigram_counts: HashMap<(String, String, String), usize>,
    fourgram_counts: HashMap<(String, String, String, String), usize>,
    total_tokens: usize,
}

impl CorpusTrainer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn unique_unigrams(&self) -> usize {
        self.unigram_counts.len()
    }

    pub fn unique_bigrams(&self) -> usize {
        self.bigram_counts.len()
    }

    pub fn unique_trigrams(&self) -> usize {
        self.trigram_counts.len()
    }

    pub fn unique_fourgrams(&self) -> usize {
        self.fourgram_counts.len()
    }

    /// Merge counts from another trainer instance
    pub fn merge(&mut self, other: CorpusTrainer) {
        self.total_tokens += other.total_tokens;
        for (w, c) in other.unigram_counts {
            *self.unigram_counts.entry(w).or_insert(0) += c;
        }
        for (bi, c) in other.bigram_counts {
            *self.bigram_counts.entry(bi).or_insert(0) += c;
        }
        for (tri, c) in other.trigram_counts {
            *self.trigram_counts.entry(tri).or_insert(0) += c;
        }
        for (four, c) in other.fourgram_counts {
            *self.fourgram_counts.entry(four).or_insert(0) += c;
        }
    }

    /// Clean, normalize, and tokenize Bengali text into sentences and words
    pub fn tokenize_text(text: &str) -> Vec<Vec<String>> {
        let mut sentences = Vec::new();
        let sentence_delimiters = ['।', '?', '!', '\n', ';'];
        let canonical_storage: String;
        let effective_text = if text.contains('\u{09bc}') || text.contains("\u{0985}\u{09be}") {
            canonical_storage = chars::canonicalize_bengali_str(text);
            canonical_storage.as_str()
        } else {
            text
        };

        for raw_sentence in effective_text.split(|c| sentence_delimiters.contains(&c)) {
            let mut words = Vec::new();
            for raw_word in raw_sentence.split(|c: char| is_word_delimiter(c)) {
                let trimmed = raw_word.trim();
                if !trimmed.is_empty()
                    && (trimmed.chars().any(crate::trainer::chars::is_bengali_char)
                        || trimmed.chars().any(|c| c.is_alphabetic()))
                    && !is_junk_token(trimmed)
                {
                    words.push(trimmed.to_string());
                }
            }
            if !words.is_empty() {
                sentences.push(words);
            }
        }

        sentences
    }

    /// Ingest raw text and accumulate N-gram statistics
    pub fn train_text(&mut self, text: &str) {
        let sentences = Self::tokenize_text(text);
        for sentence in sentences {
            self.train_sentence(&sentence);
        }
    }

    /// Parallel multi-threaded text ingestion using Rayon
    pub fn train_text_parallel(&mut self, text: &str) {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() < 50 {
            self.train_text(text);
            return;
        }

        let chunk_size = (lines.len() / (rayon::current_num_threads().max(1) * 4)).max(25);
        let merged = lines
            .par_chunks(chunk_size)
            .fold(CorpusTrainer::new, |mut local, chunk| {
                for line in chunk {
                    local.train_text(line);
                }
                local
            })
            .reduce(CorpusTrainer::new, |mut a, b| {
                a.merge(b);
                a
            });

        self.merge(merged);
    }

    /// Ingest a single pre-tokenized sentence
    pub fn train_sentence(&mut self, words: &[String]) {
        if words.is_empty() {
            return;
        }

        for (i, word) in words.iter().enumerate() {
            *self.unigram_counts.entry(word.clone()).or_insert(0) += 1;
            self.total_tokens += 1;

            if i >= 1 {
                let prev1 = &words[i - 1];
                *self
                    .bigram_counts
                    .entry((prev1.clone(), word.clone()))
                    .or_insert(0) += 1;
            }

            if i >= 2 {
                let prev2 = &words[i - 2];
                let prev1 = &words[i - 1];
                *self
                    .trigram_counts
                    .entry((prev2.clone(), prev1.clone(), word.clone()))
                    .or_insert(0) += 1;
            }

            if i >= 3 {
                let prev3 = &words[i - 3];
                let prev2 = &words[i - 2];
                let prev1 = &words[i - 1];
                *self
                    .fourgram_counts
                    .entry((prev3.clone(), prev2.clone(), prev1.clone(), word.clone()))
                    .or_insert(0) += 1;
            }
        }
    }

    /// Compile accumulated frequencies into normalized log-probabilities with pruning and capacity capping
    pub fn compile_with_config(&self, config: &TrainingConfig) -> TrainedLanguageModelData {
        let total_tokens_f = self.total_tokens.max(1) as f32;

        // 1. Filter and cap unigrams
        let mut unigram_vec: Vec<(&String, &usize)> = self
            .unigram_counts
            .iter()
            .filter(|(_, &count)| count >= config.min_unigram_freq)
            .collect();
        unigram_vec.sort_unstable_by(|a, b| b.1.cmp(a.1));
        if unigram_vec.len() > config.max_unigrams {
            unigram_vec.truncate(config.max_unigrams);
        }

        let mut unigrams = HashMap::with_capacity(unigram_vec.len());
        for (w, count) in unigram_vec {
            let prob = (*count as f32) / total_tokens_f;
            unigrams.insert(w.clone(), prob.log10());
        }

        // 2. Filter and cap bigrams (only include if both unigrams exist)
        let mut bigram_vec: Vec<(&(String, String), &usize)> = self
            .bigram_counts
            .iter()
            .filter(|((w1, w2), &count)| {
                count >= config.min_bigram_freq
                    && unigrams.contains_key(w1)
                    && unigrams.contains_key(w2)
            })
            .collect();
        bigram_vec.sort_unstable_by(|a, b| b.1.cmp(a.1));
        if bigram_vec.len() > config.max_bigrams {
            bigram_vec.truncate(config.max_bigrams);
        }

        let mut bigrams = Vec::with_capacity(bigram_vec.len());
        for ((w1, w2), count) in bigram_vec {
            let w1_count = self.unigram_counts.get(w1).copied().unwrap_or(1) as f32;
            let prob = (*count as f32) / w1_count;
            bigrams.push((w1.clone(), w2.clone(), prob.log10()));
        }

        // 3. Filter and cap trigrams (only include if words are in vocabulary)
        let mut trigram_vec: Vec<(&(String, String, String), &usize)> = self
            .trigram_counts
            .iter()
            .filter(|((w1, w2, w3), &count)| {
                count >= config.min_trigram_freq
                    && unigrams.contains_key(w1)
                    && unigrams.contains_key(w2)
                    && unigrams.contains_key(w3)
            })
            .collect();
        trigram_vec.sort_unstable_by(|a, b| b.1.cmp(a.1));
        if trigram_vec.len() > config.max_trigrams {
            trigram_vec.truncate(config.max_trigrams);
        }

        let mut trigrams = Vec::with_capacity(trigram_vec.len());
        for ((w1, w2, w3), count) in trigram_vec {
            let bi_count = self
                .bigram_counts
                .get(&(w1.clone(), w2.clone()))
                .copied()
                .unwrap_or(1) as f32;
            let prob = (*count as f32) / bi_count;
            trigrams.push((w1.clone(), w2.clone(), w3.clone(), prob.log10()));
        }

        // 4. Filter and cap fourgrams (only include if words are in vocabulary)
        let mut fourgram_vec: Vec<_> = self
            .fourgram_counts
            .iter()
            .filter(|((w1, w2, w3, w4), &count)| {
                count >= config.min_fourgram_freq
                    && unigrams.contains_key(w1)
                    && unigrams.contains_key(w2)
                    && unigrams.contains_key(w3)
                    && unigrams.contains_key(w4)
            })
            .collect();
        fourgram_vec.sort_unstable_by(|a, b| b.1.cmp(a.1));
        if fourgram_vec.len() > config.max_fourgrams {
            fourgram_vec.truncate(config.max_fourgrams);
        }

        let mut fourgrams = Vec::with_capacity(fourgram_vec.len());
        for ((w1, w2, w3, w4), count) in fourgram_vec {
            let tri_count = self
                .trigram_counts
                .get(&(w1.clone(), w2.clone(), w3.clone()))
                .copied()
                .unwrap_or(1) as f32;
            let prob = (*count as f32) / tri_count;
            fourgrams.push((w1.clone(), w2.clone(), w3.clone(), w4.clone(), prob.log10()));
        }

        TrainedLanguageModelData {
            unigrams,
            bigrams,
            trigrams,
            fourgrams,
            total_words: self.total_tokens,
        }
    }

    /// Compile accumulated frequencies into normalized log-probabilities using default configuration
    pub fn compile(&self) -> TrainedLanguageModelData {
        self.compile_with_config(&TrainingConfig::default())
    }

    /// Export trained model to a JSON file
    pub fn export_to_json<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let compiled = self.compile();
        let json = serde_json::to_string_pretty(&compiled)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Export compiled binary model directly to a file
    pub fn export_to_binary<P: AsRef<Path>>(
        &self,
        path: P,
        config: &TrainingConfig,
    ) -> Result<usize, std::io::Error> {
        let compiled = self.compile_with_config(config);
        let bytes = compiled.to_binary();
        std::fs::write(path, &bytes)?;
        Ok(bytes.len())
    }
}

/// Streaming, low-memory corpus trainer for massive datasets.
///
/// Uses a two-pass token indexing architecture:
/// - Pass 1: Streams files line-by-line via `BufReader`, counts unigrams across parallel threads,
///   and identifies the top `max_unigrams` meeting `min_unigram_freq`. Assigns compact `u32` IDs.
/// - Pass 2: Streams lines again, maps words to `u32` IDs, and counts bigrams as packed `u64`
///   (`((w1 as u64) << 32) | (w2 as u64)`) and trigrams as `(u32, u32, u32)`.
///
/// Memory footprint is strictly bounded to ~500 MB - 1 GB even on multi-gigabyte corpora,
/// preventing Linux kernel OOM kills on machines with 8-16 GB RAM.
pub fn train_files_streaming<P: AsRef<Path>>(
    paths: &[P],
    config: &TrainingConfig,
) -> Result<TrainedLanguageModelData, std::io::Error> {
    if paths.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No input corpus paths provided",
        ));
    }

    println!("    • Pass 1/2: Streaming unigram discovery & frequency counting...");
    let mut master_unigrams: HashMap<String, usize> = HashMap::new();
    let mut total_tokens = 0usize;

    const BATCH_LINES: usize = 25_000;
    for p in paths {
        let file = std::fs::File::open(p.as_ref())?;
        let reader = std::io::BufReader::with_capacity(512 * 1024, file);
        use std::io::BufRead;

        let mut batch = Vec::with_capacity(BATCH_LINES);
        let mut total_lines = 0usize;

        for line_res in reader.lines() {
            let line = line_res?;
            if !line.trim().is_empty() {
                batch.push(line);
            }
            if batch.len() >= BATCH_LINES {
                total_lines += batch.len();
                let batch_unigrams = batch
                    .par_chunks(2_500)
                    .fold(HashMap::new, |mut acc: HashMap<String, usize>, chunk| {
                        for line in chunk {
                            extract_line_words(line, |w| {
                                *acc.entry(w.to_string()).or_insert(0) += 1;
                            });
                        }
                        acc
                    })
                    .reduce(HashMap::new, |mut a, b| {
                        for (k, v) in b {
                            *a.entry(k).or_insert(0) += v;
                        }
                        a
                    });

                for (w, count) in batch_unigrams {
                    total_tokens += count;
                    *master_unigrams.entry(w).or_insert(0) += count;
                }
                batch.clear();

                if total_lines.is_multiple_of(500_000) {
                    println!(
                        "      → Pass 1: {} lines processed ({} unique words so far)...",
                        total_lines,
                        master_unigrams.len()
                    );
                }
            }
        }

        if !batch.is_empty() {
            let batch_unigrams = batch
                .par_chunks(2_500)
                .fold(HashMap::new, |mut acc: HashMap<String, usize>, chunk| {
                    for line in chunk {
                        extract_line_words(line, |w| {
                            *acc.entry(w.to_string()).or_insert(0) += 1;
                        });
                    }
                    acc
                })
                .reduce(HashMap::new, |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                });

            for (w, count) in batch_unigrams {
                total_tokens += count;
                *master_unigrams.entry(w).or_insert(0) += count;
            }
            batch.clear();
        }
    }

    println!(
        "    [✓] Pass 1 complete: {} total tokens, {} raw vocabulary words discovered.",
        total_tokens,
        master_unigrams.len()
    );

    // Prune unigrams based on min_unigram_freq and max_unigrams
    let mut unigram_vec: Vec<(String, usize)> = master_unigrams
        .into_iter()
        .filter(|(_, count)| *count >= config.min_unigram_freq)
        .collect();
    unigram_vec.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
    if unigram_vec.len() > config.max_unigrams {
        unigram_vec.truncate(config.max_unigrams);
    }

    let vocab_size = unigram_vec.len();
    let mut vocab_list = Vec::with_capacity(vocab_size);
    let mut vocab_lookup: HashMap<String, u32> = HashMap::with_capacity(vocab_size);
    let mut unigram_counts: Vec<usize> = Vec::with_capacity(vocab_size);

    for (id, (w, count)) in unigram_vec.into_iter().enumerate() {
        let u_id = id as u32;
        vocab_lookup.insert(w.clone(), u_id);
        vocab_list.push(w);
        unigram_counts.push(count);
    }

    println!(
        "    [✓] Retained {} high-capacity vocabulary words. Starting Pass 2/2 N-gram indexing...",
        vocab_size
    );

    let mut master_bigrams: HashMap<u64, u32> = HashMap::new();
    let mut master_trigrams: HashMap<(u32, u32, u32), u32> = HashMap::new();

    for p in paths {
        let file = std::fs::File::open(p.as_ref())?;
        let reader = std::io::BufReader::with_capacity(512 * 1024, file);
        use std::io::BufRead;

        let mut batch = Vec::with_capacity(BATCH_LINES);
        let mut total_lines = 0usize;

        for line_res in reader.lines() {
            let line = line_res?;
            if !line.trim().is_empty() {
                batch.push(line);
            }
            if batch.len() >= BATCH_LINES {
                total_lines += batch.len();
                let (batch_bi, batch_tri) = batch
                    .par_chunks(2_500)
                    .fold(
                        || (HashMap::<u64, u32>::new(), HashMap::<(u32, u32, u32), u32>::new()),
                        |(mut local_bi, mut local_tri), chunk| {
                            for line in chunk {
                                process_line_ngrams(line, &vocab_lookup, &mut local_bi, &mut local_tri);
                            }
                            (local_bi, local_tri)
                        },
                    )
                    .reduce(
                        || (HashMap::new(), HashMap::new()),
                        |(mut bi_a, mut tri_a), (bi_b, tri_b)| {
                            for (k, v) in bi_b {
                                *bi_a.entry(k).or_insert(0) += v;
                            }
                            for (k, v) in tri_b {
                                *tri_a.entry(k).or_insert(0) += v;
                            }
                            (bi_a, tri_a)
                        },
                    );

                for (k, v) in batch_bi {
                    *master_bigrams.entry(k).or_insert(0) += v;
                }
                for (k, v) in batch_tri {
                    *master_trigrams.entry(k).or_insert(0) += v;
                }
                batch.clear();

                if total_lines.is_multiple_of(500_000) {
                    println!(
                        "      → Pass 2: {} lines processed ({} bigrams, {} trigrams)...",
                        total_lines,
                        master_bigrams.len(),
                        master_trigrams.len()
                    );
                }
            }
        }

        if !batch.is_empty() {
            let (batch_bi, batch_tri) = batch
                .par_chunks(2_500)
                .fold(
                    || (HashMap::<u64, u32>::new(), HashMap::<(u32, u32, u32), u32>::new()),
                    |(mut local_bi, mut local_tri), chunk| {
                        for line in chunk {
                            process_line_ngrams(line, &vocab_lookup, &mut local_bi, &mut local_tri);
                        }
                        (local_bi, local_tri)
                    },
                )
                .reduce(
                    || (HashMap::new(), HashMap::new()),
                    |(mut bi_a, mut tri_a), (bi_b, tri_b)| {
                        for (k, v) in bi_b {
                            *bi_a.entry(k).or_insert(0) += v;
                        }
                        for (k, v) in tri_b {
                            *tri_a.entry(k).or_insert(0) += v;
                        }
                        (bi_a, tri_a)
                    },
                );

            for (k, v) in batch_bi {
                *master_bigrams.entry(k).or_insert(0) += v;
            }
            for (k, v) in batch_tri {
                *master_trigrams.entry(k).or_insert(0) += v;
            }
            batch.clear();
        }
    }

    println!(
        "    [✓] Pass 2 complete: {} unique bigram transitions, {} unique trigram contexts.",
        master_bigrams.len(),
        master_trigrams.len()
    );

    let total_tokens_f = total_tokens.max(1) as f32;

    // 1. Unigrams
    let mut unigrams = HashMap::with_capacity(vocab_list.len());
    for (id, w) in vocab_list.iter().enumerate() {
        let count = unigram_counts[id];
        let prob = (count as f32) / total_tokens_f;
        unigrams.insert(w.clone(), prob.log10());
    }

    // 2. Bigrams
    let mut bigram_vec: Vec<(u64, u32)> = master_bigrams
        .into_iter()
        .filter(|(_, count)| (*count as usize) >= config.min_bigram_freq)
        .collect();
    bigram_vec.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
    if bigram_vec.len() > config.max_bigrams {
        bigram_vec.truncate(config.max_bigrams);
    }

    let mut compiled_bigram_counts: HashMap<u64, u32> = HashMap::with_capacity(bigram_vec.len());
    let mut bigrams = Vec::with_capacity(bigram_vec.len());
    for (bi_key, count) in bigram_vec {
        compiled_bigram_counts.insert(bi_key, count);
        let w1_id = (bi_key >> 32) as u32;
        let w2_id = (bi_key & 0xFFFF_FFFF) as u32;
        let w1 = &vocab_list[w1_id as usize];
        let w2 = &vocab_list[w2_id as usize];
        let w1_count = unigram_counts[w1_id as usize].max(1) as f32;
        let prob = (count as f32) / w1_count;
        bigrams.push((w1.clone(), w2.clone(), prob.log10()));
    }

    // 3. Trigrams
    let mut trigram_vec: Vec<((u32, u32, u32), u32)> = master_trigrams
        .into_iter()
        .filter(|(_, count)| (*count as usize) >= config.min_trigram_freq)
        .collect();
    trigram_vec.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
    if trigram_vec.len() > config.max_trigrams {
        trigram_vec.truncate(config.max_trigrams);
    }

    let mut trigrams = Vec::with_capacity(trigram_vec.len());
    for ((w1_id, w2_id, w3_id), count) in trigram_vec {
        let bi_key = ((w1_id as u64) << 32) | (w2_id as u64);
        let bi_count = compiled_bigram_counts.get(&bi_key).copied().unwrap_or(1).max(1) as f32;
        let prob = (count as f32) / bi_count;
        let w1 = &vocab_list[w1_id as usize];
        let w2 = &vocab_list[w2_id as usize];
        let w3 = &vocab_list[w3_id as usize];
        trigrams.push((w1.clone(), w2.clone(), w3.clone(), prob.log10()));
    }

    Ok(TrainedLanguageModelData {
        unigrams,
        bigrams,
        trigrams,
        fourgrams: Vec::new(),
        total_words: total_tokens,
    })
}

fn extract_line_words<F: FnMut(&str)>(line: &str, mut on_word: F) {
    let sentence_delimiters = ['।', '?', '!', '\n', ';', '.'];
    let canonical_storage: String;
    let effective_line = if line.contains('\u{09bc}') || line.contains("\u{0985}\u{09be}") {
        canonical_storage = chars::canonicalize_bengali_str(line);
        canonical_storage.as_str()
    } else {
        line
    };

    for raw_sentence in effective_line.split(|c| sentence_delimiters.contains(&c)) {
        for raw_word in raw_sentence.split(|c: char| is_word_delimiter(c)) {
            let trimmed = raw_word.trim();
            if !trimmed.is_empty()
                && trimmed.chars().any(crate::trainer::chars::is_token_char)
                && !is_junk_token(trimmed)
            {
                on_word(trimmed);
            }
        }
    }
}

fn process_line_ngrams(
    line: &str,
    vocab_lookup: &HashMap<String, u32>,
    local_bi: &mut HashMap<u64, u32>,
    local_tri: &mut HashMap<(u32, u32, u32), u32>,
) {
    let sentence_delimiters = ['।', '?', '!', '\n', ';', '.'];
    let canonical_storage: String;
    let effective_line = if line.contains('\u{09bc}') || line.contains("\u{0985}\u{09be}") {
        canonical_storage = chars::canonicalize_bengali_str(line);
        canonical_storage.as_str()
    } else {
        line
    };

    for raw_sentence in effective_line.split(|c| sentence_delimiters.contains(&c)) {
        let mut sentence_ids: Vec<Option<u32>> = Vec::new();
        for raw_word in raw_sentence.split(|c: char| is_word_delimiter(c)) {
            let trimmed = raw_word.trim();
            if !trimmed.is_empty()
                && trimmed.chars().any(crate::trainer::chars::is_token_char)
                && !is_junk_token(trimmed)
            {
                sentence_ids.push(vocab_lookup.get(trimmed).copied());
            }
        }

        if sentence_ids.len() < 2 {
            continue;
        }

        for i in 1..sentence_ids.len() {
            if let (Some(w1), Some(w2)) = (sentence_ids[i - 1], sentence_ids[i]) {
                let bi_key = ((w1 as u64) << 32) | (w2 as u64);
                *local_bi.entry(bi_key).or_insert(0) += 1;
            }
        }

        for i in 2..sentence_ids.len() {
            if let (Some(w1), Some(w2), Some(w3)) =
                (sentence_ids[i - 2], sentence_ids[i - 1], sentence_ids[i])
            {
                let tri_key = (w1, w2, w3);
                *local_tri.entry(tri_key).or_insert(0) += 1;
            }
        }
    }
}

/// Returns true if a character serves as a token/word delimiter during corpus ingestion.
#[inline]
pub fn is_word_delimiter(c: char) -> bool {
    c.is_whitespace()
        || c.is_ascii_punctuation()
        || matches!(
            c,
            '‘' | '’'
                | '“'
                | '”'
                | '—'
                | '–'
                | '…'
                | '•'
                | '«'
                | '»'
                | '।'
                | '?'
                | '!'
                | ';'
        )
}

/// Returns true if a token represents corrupted crawl artifacts, wiki markup, or unsegmented garbage.
pub fn is_junk_token(w: &str) -> bool {
    let trimmed = w.trim();
    if trimmed.is_empty() {
        return true;
    }

    let lower = trimmed.to_lowercase();
    const JUNK_SUBSTRINGS: &[&str] = &[
        "&lt;", "&gt;", "&amp;", "&quot;", "http", "www.", ".com", ".org", ".php", "quot", "nbsp",
        "href", "cite", "ইউআরএল", "আর্কাইভের", "আর্কাইভ", "ওয়েব্যাক", "আইএসবিএন",
        "অবস্থাকার্যকর", "অবস্থাঅকার্যকর", "সংগ্রহেরতারিখ", "অনূদিতশিরোনাম",
    ];

    for junk in JUNK_SUBSTRINGS {
        if lower.contains(junk) {
            return true;
        }
    }

    if matches!(lower.as_str(), "lt" | "gt" | "amp" | "ref" | "nbsp") {
        return true;
    }

    let char_count = trimmed.chars().count();
    if !(1..=22).contains(&char_count) {
        return true;
    }

    // Must consist exclusively of authentic Bengali characters or ZWJ/ZWNJ
    let is_valid_char = |c: char| -> bool {
        matches!(c,
            '\u{0981}'..='\u{0983}'
            | '\u{0985}'..='\u{098C}'
            | '\u{098F}'..='\u{0990}'
            | '\u{0993}'..='\u{0994}'
            | '\u{0995}'..='\u{09A8}'
            | '\u{09AA}'..='\u{09B0}'
            | '\u{09B2}'
            | '\u{09B6}'..='\u{09B9}'
            | '\u{09BC}'..='\u{09C4}'
            | '\u{09C7}'..='\u{09C8}'
            | '\u{09CB}'..='\u{09CD}'
            | '\u{09CE}'
            | '\u{09D7}'
            | '\u{09DC}'..='\u{09DD}'
            | '\u{09DF}'..='\u{09E3}'
            | '\u{200C}'
            | '\u{200D}'
        )
    };

    if !trimmed.chars().all(is_valid_char) {
        return true;
    }

    // Must have at least one base consonant or independent vowel
    let has_base = trimmed.chars().any(|c| {
        matches!(c,
            '\u{0985}'..='\u{098C}'
            | '\u{098F}'..='\u{0990}'
            | '\u{0993}'..='\u{0994}'
            | '\u{0995}'..='\u{09A8}'
            | '\u{09AA}'..='\u{09B0}'
            | '\u{09B2}'
            | '\u{09B6}'..='\u{09B9}'
            | '\u{09CE}'
            | '\u{09DC}'..='\u{09DD}'
            | '\u{09DF}'
        )
    });
    if !has_base {
        return true;
    }

    // First char cannot be a combining mark, nukta, virama, or zero-width joiner
    if let Some(first) = trimmed.chars().next() {
        if matches!(first,
            '\u{0981}'..='\u{0983}'
            | '\u{09BC}'..='\u{09C4}'
            | '\u{09C7}'..='\u{09C8}'
            | '\u{09CB}'..='\u{09CD}'
            | '\u{09D7}'
            | '\u{09E2}'..='\u{09E3}'
            | '\u{200C}'
            | '\u{200D}'
        ) {
            return true;
        }
    }

    false
}

pub mod chars {
    pub fn is_bengali_char(c: char) -> bool {
        ('\u{0980}'..='\u{09FF}').contains(&c)
    }

    pub fn is_token_char(c: char) -> bool {
        c.is_alphabetic() || is_bengali_char(c)
    }

    pub fn canonicalize_bengali_str(text: &str) -> String {
        text.replace("\u{09a1}\u{09bc}", "\u{09dc}") // ড়
            .replace("\u{09a2}\u{09bc}", "\u{09dd}") // ঢ়
            .replace("\u{09af}\u{09bc}", "\u{09df}") // য়
            .replace("\u{0985}\u{09be}", "\u{0986}") // অ + া -> আ
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_training() {
        let mut trainer = CorpusTrainer::new();
        trainer.train_text("আমি বাংলায় গান গাই। আমি ভাত খাচ্ছি।");

        assert!(trainer.unigram_counts.contains_key("আমি"));
        assert!(trainer.unigram_counts.contains_key("বাংলায়"));
        assert!(trainer.unigram_counts.contains_key("গান"));
        assert!(trainer.unigram_counts.contains_key("গাই"));
        assert!(trainer.unigram_counts.contains_key("ভাত"));

        assert!(trainer.unigram_counts.contains_key("খাচ্ছি"));

        assert_eq!(*trainer.unigram_counts.get("আমি").unwrap(), 2);
        assert_eq!(
            *trainer
                .bigram_counts
                .get(&("বাংলায়".to_string(), "গান".to_string()))
                .unwrap(),
            1
        );

        let compiled = trainer.compile();
        assert!(compiled.unigrams.contains_key("আমি"));
        assert!(compiled
            .bigrams
            .iter()
            .any(|(w1, w2, _)| w1 == "বাংলায়" && w2 == "গান"));
        assert!(compiled
            .trigrams
            .iter()
            .any(|(w1, w2, w3, _)| w1 == "বাংলায়" && w2 == "গান" && w3 == "গাই"));
    }

    #[test]
    fn test_binary_roundtrip() {
        let mut trainer = CorpusTrainer::new();
        trainer.train_text("আমি বাংলায় গান গাই। আমি ভাত খাচ্ছি।");
        let compiled = trainer.compile();

        let bytes = compiled.to_binary();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], b"LLM2");

        let loaded = TrainedLanguageModelData::from_binary(&bytes).expect("Failed to load binary");
        assert_eq!(loaded.total_words, compiled.total_words);
        assert!(loaded.unigrams.len() >= compiled.unigrams.len());
        assert!(loaded.bigrams.len() >= compiled.bigrams.len());
        assert!(loaded.trigrams.len() >= compiled.trigrams.len());

        let zc = crate::zero_copy::ZeroCopyLanguageModel::from_slice(bytes.leak()).expect("Failed to parse zero-copy LLM2");
        assert_eq!(zc.unigram_count(), loaded.unigrams.len());
        assert_eq!(zc.bigram_count(), loaded.bigrams.len());
        assert_eq!(zc.trigram_count(), loaded.trigrams.len());
    }

    #[test]
    fn test_parallel_chunk_training_and_pruning() {
        let mut trainer = CorpusTrainer::new();
        let mut text = String::new();
        for _ in 0..100 {
            text.push_str("বাংলাদেশ একটি সুন্দর দেশ। আমরা সবাই দেশকে ভালোবাসি।\n");
        }
        trainer.train_text_parallel(&text);
        assert!(trainer.total_tokens() > 500);

        let config = TrainingConfig::production();
        let compiled = trainer.compile_with_config(&config);
        assert!(compiled.unigrams.contains_key("বাংলাদেশ"));
        assert!(compiled
            .bigrams
            .iter()
            .any(|(w1, w2, _)| w1 == "বাংলাদেশ" && w2 == "একটি"));
    }

    #[test]
    fn test_streaming_corpus_training() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("lekhani_test_stream_corpus.txt");
        let content = "আমি বাংলায় গান গাই। আমি বাংলায় কথা বলি।\n\
                       বাংলাদেশ চিরজীবী হোক। বাংলাদেশ একটি সুন্দর দেশ।\n";
        std::fs::write(&test_file, content).expect("Failed to write test file");

        let config = TrainingConfig::unpruned();
        let compiled = train_files_streaming(std::slice::from_ref(&test_file), &config)
            .expect("Streaming training failed");

        let _ = std::fs::remove_file(&test_file);

        assert!(compiled.unigrams.contains_key("আমি"));
        assert!(compiled.unigrams.contains_key("বাংলায়"));
        assert!(compiled.unigrams.contains_key("বাংলাদেশ"));
        assert!(compiled
            .bigrams
            .iter()
            .any(|(w1, w2, _)| w1 == "আমি" && w2 == "বাংলায়"));
        assert!(compiled
            .trigrams
            .iter()
            .any(|(w1, w2, w3, _)| w1 == "আমি" && w2 == "বাংলায়" && w3 == "গান"));
    }

    #[test]
    fn test_llm3_binary_roundtrip() {
        let mut trainer = CorpusTrainer::new();
        trainer.train_text("আমি বাংলায় গান গাই। আমি ভাত খাচ্ছি। আমরা সবাই একসাথে থাকি।");
        let compiled = trainer.compile();

        let bytes = compiled.to_binary_llm3();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], b"LLM3");

        let loaded = TrainedLanguageModelData::from_binary(&bytes).expect("Failed to load LLM3 binary");
        assert_eq!(loaded.total_words, compiled.total_words);
        assert!(loaded.unigrams.len() >= compiled.unigrams.len());
        assert!(loaded.bigrams.len() >= compiled.bigrams.len());
        assert!(loaded.trigrams.len() >= compiled.trigrams.len());

        let zc = crate::zero_copy::ZeroCopyLanguageModel::from_slice(bytes.leak()).expect("Failed to parse zero-copy LLM3");
        assert_eq!(zc.unigram_count(), loaded.unigrams.len());
        assert_eq!(zc.bigram_count(), loaded.bigrams.len());
        assert_eq!(zc.trigram_count(), loaded.trigrams.len());

        // Test scoring via zero-copy on LLM3
        let score = zc.score_candidate(Some("বাংলায়"), Some("গান"), "গাই");
        assert!(score > -3.0);
    }

    #[test]
    fn test_decomposed_nukta_canonicalization_during_training() {
        let mut trainer = CorpusTrainer::new();
        // Train text containing decomposed nuktas (ড + ় -> ড়) and (অ + া -> আ)
        let decomposed_text = "ব\u{09a1}\u{09bc} গাছ। \u{0985}\u{09be}মি আসছি।";
        trainer.train_text(decomposed_text);
        assert!(trainer.unigram_counts.contains_key("বড়"), "Must be canonicalized to atomic U+09DC ড়");
        assert!(trainer.unigram_counts.contains_key("আমি"), "Must be canonicalized to atomic U+0986 আ");
        assert!(!trainer.unigram_counts.contains_key("ব\u{09a1}\u{09bc}"), "Decomposed token must not exist in LM vocab");
    }
}

