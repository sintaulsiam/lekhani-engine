//! Zero-Copy Memory-Mapped Bengali Language Model Engine
//!
//! Provides sub-microsecond, 0-allocation N-gram probability lookups and
//! next-word prediction directly over memory-mapped binary files.

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub const BINARY_MAGIC_V2: &[u8; 4] = b"LLM2";
pub const BINARY_VERSION_V2: u32 = 2;
pub const BINARY_MAGIC_V3: &[u8; 4] = b"LLM3";
pub const BINARY_VERSION_V3: u32 = 3;

pub const QUANT_FLOOR: f32 = -12.0;
pub const QUANT_CEIL: f32 = 0.0;

#[inline]
pub fn quantize_log_prob(p: f32) -> u8 {
    let clamped = p.clamp(QUANT_FLOOR, QUANT_CEIL);
    let normalized = (clamped - QUANT_FLOOR) / (QUANT_CEIL - QUANT_FLOOR);
    (normalized * 255.0).round() as u8
}

#[inline]
pub fn dequantize_log_prob(q: u8) -> f32 {
    QUANT_FLOOR + (q as f32 / 255.0) * (QUANT_CEIL - QUANT_FLOOR)
}

/// Zero-copy reader for pre-compiled Bengali language model binary (LLM2 and LLM3)
pub struct ZeroCopyLanguageModel {
    _mmap: Option<Mmap>,
    data: &'static [u8],
    _total_words: usize,
    pub version: u32,
    vocab_count: usize,
    unigram_count: usize,
    bigram_count: usize,
    trigram_count: usize,
    fourgram_count: usize,
    // Byte offsets into `data`
    vocab_table_offset: usize,
    unigrams_offset: usize,
    bigrams_offset: usize,
    trigrams_offset: usize,
    fourgrams_offset: usize,
    string_buf_offset: usize,
    _string_buf_len: usize,
    // Hyperparameters
    pub lambda1: f32,
    pub lambda2: f32,
    pub lambda3: f32,
    pub lambda4: f32,
    pub unigram_floor: f32,
}

impl std::fmt::Debug for ZeroCopyLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeroCopyLanguageModel")
            .field("version", &self.version)
            .field("vocab_count", &self.vocab_count)
            .field("unigram_count", &self.unigram_count)
            .field("bigram_count", &self.bigram_count)
            .field("trigram_count", &self.trigram_count)
            .field("fourgram_count", &self.fourgram_count)
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

    #[inline]
    pub fn fourgram_count(&self) -> usize {
        self.fourgram_count
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
                "Binary data too short for language model header",
            ));
        }

        let magic = &data[0..4];
        let is_v2 = magic == BINARY_MAGIC_V2;
        let is_v3 = magic == BINARY_MAGIC_V3;

        if !is_v2 && !is_v3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid binary magic: expected LLM2 or LLM3, got {:?}", magic),
            ));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if is_v2 && version != BINARY_VERSION_V2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported LLM2 version: {}", version),
            ));
        }
        if is_v3 && version != BINARY_VERSION_V3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported LLM3 version: {}", version),
            ));
        }

        if is_v3 && data.len() < 36 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Binary data too short for LLM3 header",
            ));
        }

        let total_words = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let vocab_count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let unigram_count = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let bigram_count = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
        let trigram_count = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;

        let (fourgram_count, string_buffer_len, header_size) = if is_v3 {
            let four_count = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;
            let str_len = u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize;
            (four_count, str_len, 36)
        } else {
            let str_len = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;
            (0, str_len, 32)
        };

        let vocab_bytes = vocab_count.checked_mul(6).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "vocab_count overflow")
        })?;

        let (unigram_bytes, bigram_bytes, trigram_bytes, fourgram_bytes) = if is_v3 {
            (
                unigram_count.checked_mul(5).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "unigram_count overflow")
                })?,
                bigram_count.checked_mul(9).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "bigram_count overflow")
                })?,
                trigram_count.checked_mul(13).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "trigram_count overflow")
                })?,
                fourgram_count.checked_mul(17).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "fourgram_count overflow")
                })?,
            )
        } else {
            (
                unigram_count.checked_mul(8).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "unigram_count overflow")
                })?,
                bigram_count.checked_mul(12).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "bigram_count overflow")
                })?,
                trigram_count.checked_mul(16).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "trigram_count overflow")
                })?,
                0,
            )
        };

        let vocab_table_offset = header_size;
        let unigrams_offset = vocab_table_offset + vocab_bytes;
        let bigrams_offset = unigrams_offset + unigram_bytes;
        let trigrams_offset = bigrams_offset + bigram_bytes;
        let fourgrams_offset = trigrams_offset + trigram_bytes;
        let string_buf_offset = fourgrams_offset + fourgram_bytes;
        let expected_size = string_buf_offset.checked_add(string_buffer_len).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "LM file size overflow")
        })?;

        if data.len() < expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "Truncated LM data: expected at least {} bytes, got {}",
                    expected_size,
                    data.len()
                ),
            ));
        }

        // Validate that string buffer contains valid UTF-8
        let string_buf = &data[string_buf_offset..expected_size];
        let string_buf_str = std::str::from_utf8(string_buf).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid UTF-8 in LM string buffer: {}", e),
            )
        })?;

        // Validate vocabulary table offsets and lengths
        for id in 0..vocab_count {
            let entry_offset = vocab_table_offset + (id * 6);
            let str_offset = u32::from_le_bytes(data[entry_offset..entry_offset + 4].try_into().unwrap()) as usize;
            let str_len = u16::from_le_bytes(data[entry_offset + 4..entry_offset + 6].try_into().unwrap()) as usize;
            let end = str_offset.checked_add(str_len).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Word entry {} offset overflow", id))
            })?;
            if end > string_buffer_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Word entry {} extends past string buffer bounds", id),
                ));
            }
            if !string_buf_str.is_char_boundary(str_offset) || !string_buf_str.is_char_boundary(end) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Word entry {} splits a multi-byte UTF-8 character", id),
                ));
            }
        }

        Ok(Self {
            _mmap: None,
            data,
            _total_words: total_words,
            version,
            vocab_count,
            unigram_count,
            bigram_count,
            trigram_count,
            fourgram_count,
            vocab_table_offset,
            unigrams_offset,
            bigrams_offset,
            trigrams_offset,
            fourgrams_offset,
            string_buf_offset,
            _string_buf_len: string_buffer_len,
            lambda1: 0.10,
            lambda2: 0.25,
            lambda3: 0.35,
            lambda4: 0.30,
            unigram_floor: -6.0,
        })
    }

    /// Retrieve raw bytes for word at index (zero allocations, no UTF-8 check)
    #[inline]
    pub fn get_word_bytes(&self, word_id: u32) -> Option<&'static [u8]> {
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
        Some(&self.data[start..end])
    }

    /// Retrieve the UTF-8 word slice for a given word index (zero allocations)
    #[inline]
    pub fn get_word(&self, word_id: u32) -> Option<&'static str> {
        let bytes = self.get_word_bytes(word_id)?;
        std::str::from_utf8(bytes).ok()
    }

    /// Binary search for word ID in lexicographically sorted vocab table (zero allocations, ~25ns)
    #[inline]
    pub fn get_word_id(&self, target: &str) -> Option<u32> {
        if self.vocab_count == 0 {
            return None;
        }
        let target_bytes = target.as_bytes();
        let mut low = 0;
        let mut high = self.vocab_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let word_bytes = self.get_word_bytes(mid as u32)?;
            match word_bytes.cmp(target_bytes) {
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
        if self.version == 3 {
            let offset = self.unigrams_offset + (id * 5);
            let stored_id = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            if stored_id == word_id {
                Some(dequantize_log_prob(self.data[offset + 4]))
            } else {
                None
            }
        } else {
            let offset = self.unigrams_offset + (id * 8);
            let stored_id = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            if stored_id == word_id {
                let prob = f32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
                Some(prob)
            } else {
                None
            }
        }
    }

    /// Bigram probability lookup by (w1_id, w2_id) (zero allocations, ~20ns)
    pub fn get_bigram_prob(&self, w1_id: u32, w2_id: u32) -> Option<f32> {
        if self.bigram_count == 0 {
            return None;
        }
        let entry_size = if self.version == 3 { 9 } else { 12 };
        let mut low = 0;
        let mut high = self.bigram_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.bigrams_offset + (mid * entry_size);
            let mid_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            if mid_w1 < w1_id {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        let mut idx = low;
        while idx < self.bigram_count {
            let offset = self.bigrams_offset + (idx * entry_size);
            let cur_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            if cur_w1 != w1_id {
                break;
            }
            let cur_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            if cur_w2 == w2_id {
                let prob = if self.version == 3 {
                    dequantize_log_prob(self.data[offset + 8])
                } else {
                    f32::from_le_bytes(self.data[offset + 8..offset + 12].try_into().unwrap())
                };
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
        let entry_size = if self.version == 3 { 13 } else { 16 };
        let target = ((w1_id as u64) << 32) | (w2_id as u64);
        let mut low = 0;
        let mut high = self.trigram_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.trigrams_offset + (mid * entry_size);
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
            let offset = self.trigrams_offset + (idx * entry_size);
            let cur_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            let cur_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            if cur_w1 != w1_id || cur_w2 != w2_id {
                break;
            }
            let cur_w3 = u32::from_le_bytes(self.data[offset + 8..offset + 12].try_into().unwrap());
            if cur_w3 == w3_id {
                let prob = if self.version == 3 {
                    dequantize_log_prob(self.data[offset + 12])
                } else {
                    f32::from_le_bytes(self.data[offset + 12..offset + 16].try_into().unwrap())
                };
                return Some(prob);
            }
            idx += 1;
        }

        None
    }

    /// Fourgram probability lookup by (w1_id, w2_id, w3_id, w4_id) (zero allocations, ~30ns)
    pub fn get_fourgram_prob(&self, w1_id: u32, w2_id: u32, w3_id: u32, w4_id: u32) -> Option<f32> {
        if self.fourgram_count == 0 {
            return None;
        }
        let entry_size = 17;
        let mut low = 0;
        let mut high = self.fourgram_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.fourgrams_offset + (mid * entry_size);
            let mid_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            let mid_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            let mid_w3 = u32::from_le_bytes(self.data[offset + 8..offset + 12].try_into().unwrap());

            let cmp = mid_w1.cmp(&w1_id)
                .then_with(|| mid_w2.cmp(&w2_id))
                .then_with(|| mid_w3.cmp(&w3_id));

            if cmp == std::cmp::Ordering::Less {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        let mut idx = low;
        while idx < self.fourgram_count {
            let offset = self.fourgrams_offset + (idx * entry_size);
            let cur_w1 = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
            let cur_w2 = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap());
            let cur_w3 = u32::from_le_bytes(self.data[offset + 8..offset + 12].try_into().unwrap());
            if cur_w1 != w1_id || cur_w2 != w2_id || cur_w3 != w3_id {
                break;
            }
            let cur_w4 = u32::from_le_bytes(self.data[offset + 12..offset + 16].try_into().unwrap());
            if cur_w4 == w4_id {
                let prob = dequantize_log_prob(self.data[offset + 16]);
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

        let entry_size = if self.version == 3 { 9 } else { 12 };
        let mut low = 0;
        let mut high = self.bigram_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.bigrams_offset + (mid * entry_size);
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
            let offset = self.bigrams_offset + (idx * entry_size);
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

        let entry_size = if self.version == 3 { 13 } else { 16 };
        let target = ((p2_id as u64) << 32) | (p1_id as u64);
        let mut low = 0;
        let mut high = self.trigram_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = self.trigrams_offset + (mid * entry_size);
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
            let offset = self.trigrams_offset + (idx * entry_size);
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

    /// Interpolated conditional probability scoring with optional 4-gram context (~60ns)
    pub fn score_candidate_fourgram(
        &self,
        prev3: Option<&str>,
        prev2: Option<&str>,
        prev1: Option<&str>,
        word: &str,
    ) -> f32 {
        let clean_word = clean_token(word);
        let w_id = match self.get_word_id(clean_word) {
            Some(id) => id,
            None => return self.unigram_floor,
        };

        let unigram_log = self.get_unigram_prob(w_id).unwrap_or(self.unigram_floor);

        let p1_id = prev1.map(clean_token).and_then(|p| self.get_word_id(p));
        let p2_id = prev2.map(clean_token).and_then(|p| self.get_word_id(p));
        let p3_id = prev3.map(clean_token).and_then(|p| self.get_word_id(p));

        let bigram_log = p1_id.and_then(|p1| self.get_bigram_prob(p1, w_id));
        let trigram_log = match (p2_id, p1_id) {
            (Some(p2), Some(p1)) => self.get_trigram_prob(p2, p1, w_id),
            _ => None,
        };
        let fourgram_log = match (p3_id, p2_id, p1_id) {
            (Some(p3), Some(p2), Some(p1)) => self.get_fourgram_prob(p3, p2, p1, w_id),
            _ => None,
        };

        let p_uni = 10.0_f32.powf(unigram_log);
        let p_bi = bigram_log.map(|l| 10.0_f32.powf(l)).unwrap_or(0.0);
        let p_tri = trigram_log.map(|l| 10.0_f32.powf(l)).unwrap_or(0.0);
        let p_four = fourgram_log.map(|l| 10.0_f32.powf(l)).unwrap_or(0.0);

        let interpolated = if fourgram_log.is_some() {
            self.lambda4 * p_four + self.lambda3 * p_tri + self.lambda2 * p_bi + self.lambda1 * p_uni
        } else if trigram_log.is_some() {
            let tri_weight = self.lambda3 + self.lambda4;
            tri_weight * p_tri + self.lambda2 * p_bi + self.lambda1 * p_uni
        } else if bigram_log.is_some() {
            let bi_weight = self.lambda2 + self.lambda3 + self.lambda4;
            bi_weight * p_bi + self.lambda1 * p_uni
        } else {
            p_uni
        };

        interpolated.max(1e-12).log10()
    }

    /// Interpolated conditional probability scoring with zero heap allocations (~60ns)
    pub fn score_candidate(&self, prev2: Option<&str>, prev1: Option<&str>, word: &str) -> f32 {
        self.score_candidate_fourgram(None, prev2, prev1, word)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_rejects_invalid_utf8() {
        // Construct minimum valid header with 1 vocab word
        let mut buf = Vec::new();
        buf.extend_from_slice(BINARY_MAGIC_V2); // 0..4
        buf.extend_from_slice(&BINARY_VERSION_V2.to_le_bytes()); // 4..8
        buf.extend_from_slice(&1u32.to_le_bytes()); // total_words: 8..12
        buf.extend_from_slice(&1u32.to_le_bytes()); // vocab_count: 12..16
        buf.extend_from_slice(&0u32.to_le_bytes()); // unigram_count: 16..20
        buf.extend_from_slice(&0u32.to_le_bytes()); // bigram_count: 20..24
        buf.extend_from_slice(&0u32.to_le_bytes()); // trigram_count: 24..28
        buf.extend_from_slice(&2u32.to_le_bytes()); // string_buf_len: 28..32

        // Vocab table entry: offset = 0, len = 2
        buf.extend_from_slice(&0u32.to_le_bytes()); // offset
        buf.extend_from_slice(&2u16.to_le_bytes()); // len

        // Invalid UTF-8 string buffer: [0xFF, 0xFE]
        buf.push(0xFF);
        buf.push(0xFE);

        // Leak buffer to get &'static [u8] for test
        let slice: &'static [u8] = Box::leak(buf.into_boxed_slice());
        let res = ZeroCopyLanguageModel::from_slice(slice);
        assert!(res.is_err(), "ZeroCopyLanguageModel must reject invalid UTF-8 string buffer");
    }

    #[test]
    fn test_zero_copy_rejects_out_of_bounds_offset() {
        let mut buf = Vec::new();
        buf.extend_from_slice(BINARY_MAGIC_V2);
        buf.extend_from_slice(&BINARY_VERSION_V2.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // string_buf_len = 4

        // Vocab table entry: offset = 2, len = 5 (exceeds string_buf_len 4!)
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&5u16.to_le_bytes());

        // Valid UTF-8 string buffer of len 4
        buf.extend_from_slice(b"test");

        let slice: &'static [u8] = Box::leak(buf.into_boxed_slice());
        let res = ZeroCopyLanguageModel::from_slice(slice);
        assert!(res.is_err(), "ZeroCopyLanguageModel must reject out-of-bounds vocab entry");
    }
}

