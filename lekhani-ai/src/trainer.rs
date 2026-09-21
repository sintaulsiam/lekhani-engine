//! Bengali Corpus Training Engine
//!
//! Tokenizes, segments, and computes N-gram probability distributions
//! from raw Bengali text corpora for on-device language modeling.

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Trained statistical N-gram dataset exported by the trainer
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainedLanguageModelData {
    pub unigrams: HashMap<String, f32>,
    pub bigrams: Vec<(String, String, f32)>,
    pub trigrams: Vec<(String, String, String, f32)>,
    pub total_words: usize,
}

impl TrainedLanguageModelData {
    pub const BINARY_MAGIC: &'static [u8; 4] = b"LLM1";
    pub const BINARY_VERSION: u32 = 1;

    /// Load trained model data from a JSON file
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Serialize language model data into compact binary format
    pub fn to_binary(&self) -> Vec<u8> {
        let mut vocab_map: HashMap<&str, u32> = HashMap::new();
        let mut words: Vec<&str> = Vec::new();

        for w in self.unigrams.keys() {
            if !vocab_map.contains_key(w.as_str()) {
                let id = words.len() as u32;
                vocab_map.insert(w.as_str(), id);
                words.push(w.as_str());
            }
        }
        for (w1, w2, _) in &self.bigrams {
            if !vocab_map.contains_key(w1.as_str()) {
                let id = words.len() as u32;
                vocab_map.insert(w1.as_str(), id);
                words.push(w1.as_str());
            }
            if !vocab_map.contains_key(w2.as_str()) {
                let id = words.len() as u32;
                vocab_map.insert(w2.as_str(), id);
                words.push(w2.as_str());
            }
        }
        for (w1, w2, w3, _) in &self.trigrams {
            if !vocab_map.contains_key(w1.as_str()) {
                let id = words.len() as u32;
                vocab_map.insert(w1.as_str(), id);
                words.push(w1.as_str());
            }
            if !vocab_map.contains_key(w2.as_str()) {
                let id = words.len() as u32;
                vocab_map.insert(w2.as_str(), id);
                words.push(w2.as_str());
            }
            if !vocab_map.contains_key(w3.as_str()) {
                let id = words.len() as u32;
                vocab_map.insert(w3.as_str(), id);
                words.push(w3.as_str());
            }
        }

        let mut string_buffer = Vec::new();
        let mut vocab_entries: Vec<(u32, u16)> = Vec::with_capacity(words.len());
        for &w in &words {
            let offset = string_buffer.len() as u32;
            let bytes = w.as_bytes();
            let len = bytes.len() as u16;
            string_buffer.extend_from_slice(bytes);
            vocab_entries.push((offset, len));
        }

        let vocab_count = words.len() as u32;
        let unigram_count = self.unigrams.len() as u32;
        let bigram_count = self.bigrams.len() as u32;
        let trigram_count = self.trigrams.len() as u32;
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

        // Unigrams
        for (w, p) in &self.unigrams {
            let wid = vocab_map[w.as_str()];
            out.extend_from_slice(&wid.to_le_bytes());
            out.extend_from_slice(&p.to_le_bytes());
        }

        // Bigrams
        for (w1, w2, p) in &self.bigrams {
            let w1_id = vocab_map[w1.as_str()];
            let w2_id = vocab_map[w2.as_str()];
            out.extend_from_slice(&w1_id.to_le_bytes());
            out.extend_from_slice(&w2_id.to_le_bytes());
            out.extend_from_slice(&p.to_le_bytes());
        }

        // Trigrams
        for (w1, w2, w3, p) in &self.trigrams {
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

    /// Load language model data from compact binary bytes
    pub fn from_binary(data: &[u8]) -> Result<Self, std::io::Error> {
        if data.len() < 32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Binary data too short for header",
            ));
        }

        if &data[0..4] != Self::BINARY_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid binary language model magic header",
            ));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != Self::BINARY_VERSION {
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
        let string_buffer_len = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;

        let expected_size = 32
            + (vocab_count * 6)
            + (unigram_count * 8)
            + (bigram_count * 12)
            + (trigram_count * 16)
            + string_buffer_len;

        if data.len() < expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Binary language model data truncated",
            ));
        }

        let mut cursor = 32;

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
        cursor += unigram_count * 8;

        // Bigrams cursor
        let bigrams_start = cursor;
        cursor += bigram_count * 12;

        // Trigrams cursor
        let trigrams_start = cursor;
        cursor += trigram_count * 16;

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
            let prob = f32::from_le_bytes(data[u_cursor + 4..u_cursor + 8].try_into().unwrap());
            if wid < words.len() {
                unigrams.insert(words[wid].clone(), prob);
            }
            u_cursor += 8;
        }

        // Parse bigrams
        let mut bigrams = Vec::with_capacity(bigram_count);
        let mut b_cursor = bigrams_start;
        for _ in 0..bigram_count {
            let w1_id = u32::from_le_bytes(data[b_cursor..b_cursor + 4].try_into().unwrap()) as usize;
            let w2_id = u32::from_le_bytes(data[b_cursor + 4..b_cursor + 8].try_into().unwrap()) as usize;
            let prob = f32::from_le_bytes(data[b_cursor + 8..b_cursor + 12].try_into().unwrap());
            if w1_id < words.len() && w2_id < words.len() {
                bigrams.push((words[w1_id].clone(), words[w2_id].clone(), prob));
            }
            b_cursor += 12;
        }

        // Parse trigrams
        let mut trigrams = Vec::with_capacity(trigram_count);
        let mut t_cursor = trigrams_start;
        for _ in 0..trigram_count {
            let w1_id = u32::from_le_bytes(data[t_cursor..t_cursor + 4].try_into().unwrap()) as usize;
            let w2_id = u32::from_le_bytes(data[t_cursor + 4..t_cursor + 8].try_into().unwrap()) as usize;
            let w3_id = u32::from_le_bytes(data[t_cursor + 8..t_cursor + 12].try_into().unwrap()) as usize;
            let prob = f32::from_le_bytes(data[t_cursor + 12..t_cursor + 16].try_into().unwrap());
            if w1_id < words.len() && w2_id < words.len() && w3_id < words.len() {
                trigrams.push((words[w1_id].clone(), words[w2_id].clone(), words[w3_id].clone(), prob));
            }
            t_cursor += 16;
        }

        Ok(Self {
            unigrams,
            bigrams,
            trigrams,
            total_words,
        })
    }

    /// Load language model data directly from a binary file
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let bytes = std::fs::read(path)?;
        Self::from_binary(&bytes)
    }
}

/// Dynamic N-gram Corpus Trainer
#[derive(Debug, Clone, Default)]
pub struct CorpusTrainer {
    unigram_counts: HashMap<String, usize>,
    bigram_counts: HashMap<(String, String), usize>,
    trigram_counts: HashMap<(String, String, String), usize>,
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

    /// Clean, normalize, and tokenize Bengali text into sentences and words
    pub fn tokenize_text(text: &str) -> Vec<Vec<String>> {
        let mut sentences = Vec::new();
        let sentence_delimiters = ['।', '?', '!', '\n', ';'];

        for raw_sentence in text.split(|c| sentence_delimiters.contains(&c)) {
            let mut words = Vec::new();
            for raw_word in raw_sentence.split_whitespace() {
                let clean_word: String = raw_word
                    .chars()
                    .filter(|c| {
                        !c.is_ascii_punctuation()
                            && *c != '‘'
                            && *c != '’'
                            && *c != '“'
                            && *c != '”'
                            && *c != '\''
                            && *c != '"'
                            && *c != ','
                            && *c != ':'
                            && *c != '—'
                            && *c != '-'
                            && *c != '('
                            && *c != ')'
                            && *c != '['
                            && *c != ']'
                            && *c != '{'
                            && *c != '}'
                    })
                    .collect();

                let trimmed = clean_word.trim();
                if !trimmed.is_empty()
                    && trimmed.chars().any(crate::trainer::chars::is_bengali_char)
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
        }
    }

    /// Compile accumulated frequencies into normalized log-probabilities with Laplace smoothing
    pub fn compile(&self) -> TrainedLanguageModelData {
        let total_tokens_f = self.total_tokens.max(1) as f32;
        let mut unigrams = HashMap::with_capacity(self.unigram_counts.len());
        for (w, count) in &self.unigram_counts {
            let prob = (*count as f32) / total_tokens_f;
            unigrams.insert(w.clone(), prob.log10());
        }

        let mut bigrams = Vec::with_capacity(self.bigram_counts.len());
        for ((w1, w2), count) in &self.bigram_counts {
            let w1_count = self.unigram_counts.get(w1).copied().unwrap_or(1) as f32;
            let prob = (*count as f32) / w1_count;
            bigrams.push((w1.clone(), w2.clone(), prob.log10()));
        }

        let mut trigrams = Vec::with_capacity(self.trigram_counts.len());
        for ((w1, w2, w3), count) in &self.trigram_counts {
            let bi_count = self
                .bigram_counts
                .get(&(w1.clone(), w2.clone()))
                .copied()
                .unwrap_or(1) as f32;
            let prob = (*count as f32) / bi_count;
            trigrams.push((w1.clone(), w2.clone(), w3.clone(), prob.log10()));
        }

        TrainedLanguageModelData {
            unigrams,
            bigrams,
            trigrams,
            total_words: self.total_tokens,
        }
    }

    /// Export trained model to a JSON file
    pub fn export_to_json<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let compiled = self.compile();
        let json = serde_json::to_string_pretty(&compiled)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }
}

pub(crate) mod chars {
    pub fn is_bengali_char(c: char) -> bool {
        ('\u{0980}'..='\u{09FF}').contains(&c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_training() {
        let mut trainer = CorpusTrainer::new();
        trainer.train_text("আমি বাংলায় গান গাই। আমি ভাত খাচ্ছি।");

        assert!(trainer.unigram_counts.contains_key("আমি"));
        assert!(trainer.unigram_counts.contains_key("বাংলায়"));
        assert!(trainer.unigram_counts.contains_key("গান"));
        assert!(trainer.unigram_counts.contains_key("গাই"));
        assert!(trainer.unigram_counts.contains_key("ভাত"));
        assert!(trainer.unigram_counts.contains_key("খাচ্ছি"));

        assert_eq!(*trainer.unigram_counts.get("আমি").unwrap(), 2);
        assert_eq!(
            *trainer
                .bigram_counts
                .get(&("বাংলায়".to_string(), "গান".to_string()))
                .unwrap(),
            1
        );

        let compiled = trainer.compile();
        assert!(compiled.unigrams.contains_key("আমি"));
        assert!(compiled
            .bigrams
            .iter()
            .any(|(w1, w2, _)| w1 == "বাংলায়" && w2 == "গান"));
        assert!(compiled
            .trigrams
            .iter()
            .any(|(w1, w2, w3, _)| w1 == "বাংলায়" && w2 == "গান" && w3 == "গাই"));
    }

    #[test]
    fn test_binary_roundtrip() {
        let mut trainer = CorpusTrainer::new();
        trainer.train_text("আমি বাংলায় গান গাই। আমি ভাত খাচ্ছি।");
        let compiled = trainer.compile();

        let bytes = compiled.to_binary();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], b"LLM1");

        let loaded = TrainedLanguageModelData::from_binary(&bytes).expect("Failed to load binary");
        assert_eq!(loaded.total_words, compiled.total_words);
        assert_eq!(loaded.unigrams.len(), compiled.unigrams.len());
        assert_eq!(loaded.bigrams.len(), compiled.bigrams.len());
        assert_eq!(loaded.trigrams.len(), compiled.trigrams.len());
    }
}
