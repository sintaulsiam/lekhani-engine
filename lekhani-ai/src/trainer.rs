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
    /// Load trained model data from a JSON file
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
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
}
