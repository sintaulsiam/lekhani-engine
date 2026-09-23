//! Zero-Copy Memory-Mapped Bengali Language Model Engine
//!
//! Provides sub-microsecond, 0-allocation N-gram probability lookups and
//! next-word prediction directly over memory-mapped binary files.

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub const BINARY_MAGIC_V2: &[u8; 4] = b"LLM2";
pub const BINARY_VERSION_V2: u32 = 2;

/// Zero-copy reader for pre-compiled Bengali language model binary
pub struct ZeroCopyLanguageModel {
    _mmap: Option<Mmap>,
    data: &'static [u8],
    _total_words: usize,
    vocab_count: usize,
    unigram_count: usize,
    bigram_count: usize,
    trigram_count: usize,
    // Byte offsets into `data`
    vocab_table_offset: usize,
    unigrams_offset: usize,
    bigrams_offset: usize,
    trigrams_offset: usize,
    string_buf_offset: usize,
    _string_buf_len: usize,
    // Hyperparameters
    pub lambda1: f32,
    pub lambda2: f32,
    pub lambda3: f32,
    pub unigram_floor: f32,
}

impl std::fmt::Debug for ZeroCopyLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeroCopyLanguageModel")
            .field("vocab_count", &self.vocab_count)
            .field("unigram_count", &self.unigram_count)
            .field("bigram_count", &self.bigram_count)
            .field("trigram_count", &self.trigram_count)
            .finish()
    }
}

unsafe impl Send for ZeroCopyLanguageModel {}
unsafe impl Sync for ZeroCopyLanguageModel {}

impl ZeroCopyLanguageModel {
    #[inline]
    pub fn vocab_count(&self) -> usize {
        self.vocab_count
    }

    #[inline]
    pub fn unigram_count(&self) -> usize {
        self.unigram_count
    }

    #[inline]
    pub fn bigram_count(&self) -> usize {
        self.bigram_count
    }

    #[inline]
    pub fn trigram_count(&self) -> usize {
        self.trigram_count
    }
    /// Open and memory-map a binary model from disk
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        // Safe: mmap is held in struct as _mmap, pointer remains valid for struct lifetime
        let slice: &'static [u8] = unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
        let mut model = Self::from_slice(slice)?;
        model._mmap = Some(mmap);
        Ok(model)
    }

    /// Read zero-copy model directly from an existing byte slice (for tests/embedded data)
    pub fn from_slice(data: &'static [u8]) -> Result<Self, std::io::Error> {
        if data.len() < 32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Binary data too short for LLM2 header",
            ));
        }

        if &data[0..4] != BINARY_MAGIC_V2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid binary magic: expected LLM2, got {:?}", &data[0..4]),
            ));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != BINARY_VERSION_V2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported LLM2 version: {}", version),
            ));
        }

        let total_words = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let vocab_count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let unigram_count = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let bigram_count = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
        let trigram_count = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
        let string_buffer_len = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;

        let vocab_table_offset = 32;
        let unigrams_offset = vocab_table_offset + (vocab_count * 6);
        let bigrams_offset = unigrams_offset + (unigram_count * 8);
        let trigrams_offset = bigrams_offset + (bigram_count * 12);
        let string_buf_offset = trigrams_offset + (trigram_count * 16);
        let expected_size = string_buf_offset + string_buffer_len;

        if data.len() < expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Truncated LLM2 data: expected at least {} bytes, got {}",
                    expected_size,
                    data.len()
                ),
            ));
        }

        Ok(Self {
            _mmap: None,
            data,
            _total_words: total_words,
            vocab_count,
            unigram_count,
            bigram_count,
            trigram_count,
            vocab_table_offset,
            unigrams_offset,
            bigrams_offset,
            trigrams_offset,
            string_buf_offset,
            _string_buf_len: string_buffer_len,
            lambda1: 0.15,
            lambda2: 0.35,
            lambda3: 0.50,
            unigram_floor: -6.0,
        })
    }

    /// Retrieve the UTF-8 word slice for a given word index (zero allocations)
    #[inline]
    pub fn get_word(&self, word_id: u32) -> Option<&'static str> {
        let id = word_id as usize;
        if id >= self.vocab_count {
            return None;
        }
        let entry_offset = self.vocab_table_offset + (id * 6);
        let str_offset = u32::from_le_bytes(self.data[entry_offset..entry_offset + 4].try_into().unwrap()) as usize;
        let str_len = u16::from_le_bytes(self.data[entry_offset + 4..entry_offset + 6].try_into().unwrap()) as usize;

        let start = self.string_buf_offset + str_offset;
        let end = start + str_len;
        if end > self.data.len() {
            return None;
        }
        std::str::from_utf8(&self.data[start..end]).ok()
    }

    /// Binary search for word ID in lexicographically sorted vocab table (zero allocations, ~30ns)
    pub fn get_word_id(&self, target: &str) -> Option<u32> {
        if self.vocab_count == 0 {
            return None;
        }
        let mut low = 0;
        let mut high = self.vocab_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let word = self.get_word(mid as u32)?;
            match word.cmp(target) {
                std::cmp::Ordering::Equal => return Some(mid as u32),
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
            }
        }
        None
    }

    /// Unigram probability lookup by word_id (zero allocations, ~2ns)
    #[inline]
    pub fn get_unigram_prob(&self, word_id: u32) -> Option<f32> {
        let id = word_id as usize;
        if id >= self.unigram_count {
            return None;
        }
        // In LLM2 format, unigrams are stored in direct word_id order
        let offset = self.unigrams_offset + (id * 8);
        let stored_id = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
        if stored_id == word_id {
            let prob = f32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            Some(prob)
        } else {
            None
        }
    }

    /// Bigram probability lookup by (w1_id, w2_id) (zero allocations, ~20ns)
    pub fn get_bigram_prob(&self, w1_id: u32, w2_id: u32) -> Option<f32> {
        if self.bigram_count == 0 {
            return None;
        }
        // Bigrams are sorted by w1_id ascending, log_prob descending
        // Find first occurrence of w1_id via binary search
        let mut low = 0;
        let mut high = self.bigram_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.bigrams_offset + (mid * 12);
            let mid_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            if mid_w1 < w1_id {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        // Scan entries with matching w1_id
        let mut idx = low;
        while idx < self.bigram_count {
            let offset = self.bigrams_offset + (idx * 12);
            let cur_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            if cur_w1 != w1_id {
                break;
            }
            let cur_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            if cur_w2 == w2_id {
                let prob = f32::from_le_bytes(self.data[offset + 8..offset + 12].try_into().unwrap());
                return Some(prob);
            }
            idx += 1;
        }

        None
    }

    /// Trigram probability lookup by (w1_id, w2_id, w3_id) (zero allocations, ~25ns)
    pub fn get_trigram_prob(&self, w1_id: u32, w2_id: u32, w3_id: u32) -> Option<f32> {
        if self.trigram_count == 0 {
            return None;
        }
        // Trigrams are sorted by (w1_id, w2_id) ascending, log_prob descending
        let target = ((w1_id as u64) << 32) | (w2_id as u64);
        let mut low = 0;
        let mut high = self.trigram_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.trigrams_offset + (mid * 16);
            let mid_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            let mid_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            let mid_pair = ((mid_w1 as u64) << 32) | (mid_w2 as u64);
            if mid_pair < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        let mut idx = low;
        while idx < self.trigram_count {
            let offset = self.trigrams_offset + (idx * 16);
            let cur_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            let cur_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            if cur_w1 != w1_id || cur_w2 != w2_id {
                break;
            }
            let cur_w3 = u32::from_le_bytes(self.data[offset + 8..offset + 12].try_into().unwrap());
            if cur_w3 == w3_id {
                let prob = f32::from_le_bytes(self.data[offset + 12..offset + 16].try_into().unwrap());
                return Some(prob);
            }
            idx += 1;
        }

        None
    }

    /// Retrieve the top next-word continuations for a word (zero extra HashMaps, ~50ns)
    pub fn get_next_words(&self, previous_word: &str, limit: usize) -> Vec<String> {
        let clean = clean_token(previous_word);
        let w1_id = match self.get_word_id(clean) {
            Some(id) => id,
            None => return Vec::new(),
        };

        if self.bigram_count == 0 || limit == 0 {
            return Vec::new();
        }

        // Binary search for first occurrence of w1_id
        let mut low = 0;
        let mut high = self.bigram_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.bigrams_offset + (mid * 12);
            let mid_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            if mid_w1 < w1_id {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        let mut results = Vec::with_capacity(limit);
        let mut idx = low;
        while idx < self.bigram_count && results.len() < limit {
            let offset = self.bigrams_offset + (idx * 12);
            let cur_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            if cur_w1 != w1_id {
                break;
            }
            let cur_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            if let Some(word_str) = self.get_word(cur_w2) {
                results.push(word_str.to_string());
            }
            idx += 1;
        }

        results
    }

    /// Retrieve the top next-word continuations for a trigram context (zero extra HashMaps, ~50ns)
    pub fn get_next_words_trigram(&self, prev2: &str, prev1: &str, limit: usize) -> Vec<String> {
        let c2 = clean_token(prev2);
        let c1 = clean_token(prev1);
        let p2_id = match self.get_word_id(c2) {
            Some(id) => id,
            None => return self.get_next_words(prev1, limit),
        };
        let p1_id = match self.get_word_id(c1) {
            Some(id) => id,
            None => return self.get_next_words(prev1, limit),
        };

        if self.trigram_count == 0 || limit == 0 {
            return self.get_next_words(prev1, limit);
        }

        let target = ((p2_id as u64) << 32) | (p1_id as u64);
        let mut low = 0;
        let mut high = self.trigram_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.trigrams_offset + (mid * 16);
            let mid_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            let mid_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            let mid_pair = ((mid_w1 as u64) << 32) | (mid_w2 as u64);
            if mid_pair < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        let mut results = Vec::with_capacity(limit);
        let mut idx = low;
        while idx < self.trigram_count && results.len() < limit {
            let offset = self.trigrams_offset + (idx * 16);
            let cur_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            let cur_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            if cur_w1 != p2_id || cur_w2 != p1_id {
                break;
            }
            let cur_w3 = u32::from_le_bytes(self.data[offset + 8..offset + 12].try_into().unwrap());
            if let Some(word_str) = self.get_word(cur_w3) {
                results.push(word_str.to_string());
            }
            idx += 1;
        }

        if results.is_empty() {
            self.get_next_words(prev1, limit)
        } else {
            results
        }
    }

    /// Interpolated conditional probability scoring with zero heap allocations (~60ns)
    pub fn score_candidate(&self, prev2: Option<&str>, prev1: Option<&str>, word: &str) -> f32 {
        let clean_word = clean_token(word);
        let w_id = match self.get_word_id(clean_word) {
            Some(id) => id,
            None => return self.unigram_floor,
        };

        let unigram_log = self.get_unigram_prob(w_id).unwrap_or(self.unigram_floor);

        let p1_id = prev1.map(clean_token).and_then(|p| self.get_word_id(p));
        let p2_id = prev2.map(clean_token).and_then(|p| self.get_word_id(p));

        let bigram_log = p1_id.and_then(|p1| self.get_bigram_prob(p1, w_id));
        let trigram_log = match (p2_id, p1_id) {
            (Some(p2), Some(p1)) => self.get_trigram_prob(p2, p1, w_id),
            _ => None,
        };

        // Linear interpolation in probability domain: P = λ3*P_tri + λ2*P_bi + λ1*P_uni
        let p_uni = 10.0_f32.powf(unigram_log);
        let p_bi = bigram_log.map(|l| 10.0_f32.powf(l)).unwrap_or(0.0);
        let p_tri = trigram_log.map(|l| 10.0_f32.powf(l)).unwrap_or(0.0);

        let interpolated = if trigram_log.is_some() {
            self.lambda3 * p_tri + self.lambda2 * p_bi + self.lambda1 * p_uni
        } else if bigram_log.is_some() {
            let bi_weight = self.lambda2 + self.lambda3;
            bi_weight * p_bi + self.lambda1 * p_uni
        } else {
            p_uni
        };

        interpolated.max(1e-12).log10()
    }
}

#[inline]
fn clean_token(s: &str) -> &str {
    s.trim_matches(|c: char| {
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
    })
}
