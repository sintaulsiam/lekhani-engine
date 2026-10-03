//! Bilingual BPE Vocabulary & Subword Tokenizer
//!
//! Maps conversational Bengali and English tokens to compact integer IDs.
//! Supports subword segmentation, code-mixed text, and special control tokens.

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

pub const PAD_TOKEN_ID: u32 = 0;
pub const UNK_TOKEN_ID: u32 = 1;
pub const BOS_TOKEN_ID: u32 = 2;
pub const EOS_TOKEN_ID: u32 = 3;

pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const BOS_TOKEN: &str = "<s>";
pub const EOS_TOKEN: &str = "</s>";

/// Bilingual Subword Vocabulary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeVocabulary {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
}

impl Default for BpeVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

impl BpeVocabulary {
    /// Initialize with base special tokens
    pub fn new() -> Self {
        let mut vocab = Self {
            tokens: Vec::with_capacity(512),
            token_to_id: HashMap::with_capacity(512),
        };

        vocab.add_token(PAD_TOKEN);
        vocab.add_token(UNK_TOKEN);
        vocab.add_token(BOS_TOKEN);
        vocab.add_token(EOS_TOKEN);

        vocab
    }

    /// Add a new token if not already present
    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.tokens.len() as u32;
        self.tokens.push(token.to_string());
        self.token_to_id.insert(token.to_string(), id);
        id
    }

    /// Build vocabulary from a list of tokens
    pub fn from_tokens(tokens: Vec<String>) -> Self {
        let mut vocab = Self::new();
        for token in tokens {
            vocab.add_token(&token);
        }
        vocab
    }

    /// Total number of tokens in the vocabulary
    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Lookup ID for token, returning UNK_TOKEN_ID if missing
    #[inline]
    pub fn get_id(&self, token: &str) -> u32 {
        self.token_to_id.get(token).copied().unwrap_or(UNK_TOKEN_ID)
    }

    /// Lookup token string for given ID
    #[inline]
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Tokenize text into sequence of token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            // Check direct word match first
            if let Some(&id) = self.token_to_id.get(word) {
                ids.push(id);
                continue;
            }

            // Grapheme-cluster-aware subword matching using byte slices (zero allocations)
            let boundaries: Vec<usize> = word
                .grapheme_indices(true)
                .map(|(i, _)| i)
                .chain(std::iter::once(word.len()))
                .collect();

            let mut start = 0;
            while start < boundaries.len() - 1 {
                let mut matched = false;
                for end in (start + 1..boundaries.len()).rev() {
                    let sub = &word[boundaries[start]..boundaries[end]];
                    if let Some(&id) = self.token_to_id.get(sub) {
                        ids.push(id);
                        start = end;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    ids.push(UNK_TOKEN_ID);
                    start += 1;
                }
            }
        }
        ids
    }

    /// Decode sequence of token IDs back into string
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut words = Vec::new();
        for &id in ids {
            if id == PAD_TOKEN_ID || id == BOS_TOKEN_ID || id == EOS_TOKEN_ID {
                continue;
            }
            if let Some(token) = self.get_token(id) {
                words.push(token);
            }
        }
        words.join(" ")
    }

    /// Load vocabulary from a JSON file
    pub fn load_json<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save vocabulary to a bincode binary file
    pub fn save_binary<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, self)
            .map_err(std::io::Error::other)
    }

    /// Load vocabulary from a bincode binary file
    pub fn load_binary<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        bincode::deserialize_from(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_encoding_decoding() {
        let mut vocab = BpeVocabulary::new();
        vocab.add_token("আমি");
        vocab.add_token("ভাত");
        vocab.add_token("খাব");
        vocab.add_token("office");
        vocab.add_token("call");

        let encoded = vocab.encode("আমি office এ ভাত খাব");
        assert_eq!(encoded[0], vocab.get_id("আমি"));
        assert_eq!(encoded[1], vocab.get_id("office"));

        let decoded = vocab.decode(&[vocab.get_id("আমি"), vocab.get_id("ভাত")]);
        assert_eq!(decoded, "আমি ভাত");
    }

    #[test]
    fn test_unknown_token_handling() {
        let vocab = BpeVocabulary::new();
        let encoded = vocab.encode("অজানা_শব্দ");
        assert!(encoded.iter().all(|&id| id == UNK_TOKEN_ID));
    }
}
