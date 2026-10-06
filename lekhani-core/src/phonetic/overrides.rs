//! Zero-Copy, High-Performance Supervised Phonetic Overrides Engine
//!
//! Stores compiled phonetic overrides (e.g., colloquialisms, texting slang, proper nouns)
//! in a contiguous, cache-aligned binary layout (POVR v2). Lookups execute via binary search
//! directly against memory slices in ~100 nanoseconds with ZERO runtime heap allocations.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Index;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverrideCandidate<'a> {
    pub word: &'a str,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct OverrideEntry<'a> {
    items: [(&'a str, f32); 4],
    len: usize,
}

impl<'a> OverrideEntry<'a> {
    #[inline]
    pub fn new() -> Self {
        Self {
            items: [("", 0.0); 4],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, word: &'a str, conf: f32) {
        if self.len < 4 {
            self.items[self.len] = (word, conf);
            self.len += 1;
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[(&'a str, f32)] {
        &self.items[..self.len]
    }

    #[inline]
    pub fn first(&self) -> Option<(&'a str, f32)> {
        if self.len > 0 {
            Some(self.items[0])
        } else {
            None
        }
    }
}

impl<'a> Index<usize> for OverrideEntry<'a> {
    type Output = (&'a str, f32);

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        if index < self.len {
            &self.items[index]
        } else {
            panic!("OverrideEntry index out of bounds: {} >= {}", index, self.len);
        }
    }
}

impl<'a> IntoIterator for OverrideEntry<'a> {
    type Item = (&'a str, f32);
    type IntoIter = OverrideEntryIter<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        OverrideEntryIter {
            entry: self,
            index: 0,
        }
    }
}

pub struct OverrideEntryIter<'a> {
    entry: OverrideEntry<'a>,
    index: usize,
}

impl<'a> Iterator for OverrideEntryIter<'a> {
    type Item = (&'a str, f32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.entry.len {
            let item = self.entry.items[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.entry.len - self.index;
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for OverrideEntryIter<'a> {}

#[derive(Debug, Clone)]
pub enum OverrideBuffer {
    Static(&'static [u8]),
    Shared(Arc<[u8]>),
}

impl AsRef<[u8]> for OverrideBuffer {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Static(s) => s,
            Self::Shared(s) => s.as_ref(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZeroCopyOverrides {
    buffer: OverrideBuffer,
    key_count: usize,
    cand_count: usize,
    keys_offset: usize,
    cands_offset: usize,
    strings_offset: usize,
    strings_len: usize,
}

impl Default for ZeroCopyOverrides {
    fn default() -> Self {
        Self {
            buffer: OverrideBuffer::Static(&[]),
            key_count: 0,
            cand_count: 0,
            keys_offset: 0,
            cands_offset: 0,
            strings_offset: 0,
            strings_len: 0,
        }
    }
}

impl ZeroCopyOverrides {
    pub const MAGIC: &'static [u8; 4] = b"POVR";
    pub const VERSION: u32 = 2;
    pub const HEADER_LEN: usize = 64;
    pub const KEY_RECORD_LEN: usize = 12;
    pub const CAND_RECORD_LEN: usize = 12;

    pub fn from_static(bytes: &'static [u8]) -> Result<Self, &'static str> {
        if bytes.len() >= 8 && &bytes[0..4] == Self::MAGIC && u32::from_le_bytes(bytes[4..8].try_into().unwrap_or([0; 4])) == 1 {
            // Backward compatibility with legacy POVR v1
            let map = parse_legacy_v1(bytes).ok_or("Failed to parse legacy v1 POVR")?;
            return Ok(Self::from_map(&map));
        }
        Self::parse_buffer(OverrideBuffer::Static(bytes))
    }

    pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self, &'static str> {
        if bytes.len() >= 8 && &bytes[0..4] == Self::MAGIC && u32::from_le_bytes(bytes[4..8].try_into().unwrap_or([0; 4])) == 1 {
            let map = parse_legacy_v1(&bytes).ok_or("Failed to parse legacy v1 POVR")?;
            return Ok(Self::from_map(&map));
        }
        Self::parse_buffer(OverrideBuffer::Shared(bytes))
    }

    pub fn from_map(map: &std::collections::HashMap<String, Vec<(String, f32)>>) -> Self {
        let compiled = Self::compile_to_bytes(map.iter().map(|(k, v)| (k.as_str(), v.as_slice())));
        Self::parse_buffer(OverrideBuffer::Shared(Arc::from(compiled.into_boxed_slice())))
            .unwrap_or_default()
    }

    pub fn from_hashbrown_map(map: &hashbrown::HashMap<String, Vec<(String, f32)>>) -> Self {
        let compiled = Self::compile_to_bytes(map.iter().map(|(k, v)| (k.as_str(), v.as_slice())));
        Self::parse_buffer(OverrideBuffer::Shared(Arc::from(compiled.into_boxed_slice())))
            .unwrap_or_default()
    }

    fn parse_buffer(buffer: OverrideBuffer) -> Result<Self, &'static str> {
        let data = buffer.as_ref();
        if data.is_empty() {
            return Ok(Self::default());
        }
        if data.len() < Self::HEADER_LEN {
            return Err("Binary data too short for header");
        }
        if &data[0..4] != Self::MAGIC {
            return Err("Invalid POVR magic header");
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != Self::VERSION {
            return Err("Unsupported POVR version");
        }
        let key_count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let cand_count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let keys_offset = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let cands_offset = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
        let strings_offset = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
        let strings_len = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;

        let keys_total = key_count.checked_mul(Self::KEY_RECORD_LEN).ok_or("Overflow keys len")?;
        let cands_total = cand_count.checked_mul(Self::CAND_RECORD_LEN).ok_or("Overflow cands len")?;

        if keys_offset + keys_total > data.len() {
            return Err("Keys table out of bounds");
        }
        if cands_offset + cands_total > data.len() {
            return Err("Candidates table out of bounds");
        }
        if strings_offset + strings_len > data.len() {
            return Err("Strings buffer out of bounds");
        }

        std::str::from_utf8(&data[strings_offset..strings_offset + strings_len])
            .map_err(|_| "Invalid UTF-8 in strings buffer")?;

        Ok(Self {
            buffer,
            key_count,
            cand_count,
            keys_offset,
            cands_offset,
            strings_offset,
            strings_len,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.key_count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.key_count == 0
    }

    #[inline]
    pub fn lookup(&self, query: &str) -> Option<OverrideEntry<'_>> {
        if self.key_count == 0 {
            return None;
        }
        let data = self.buffer.as_ref();
        let strings = &data[self.strings_offset..self.strings_offset + self.strings_len];

        let mut low = 0;
        let mut high = self.key_count;

        while low < high {
            let mid = (low + high) / 2;
            let pos = self.keys_offset + mid * Self::KEY_RECORD_LEN;
            let k_off = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            let k_len = u16::from_le_bytes(data[pos + 4..pos + 6].try_into().unwrap()) as usize;

            let key_str = unsafe {
                std::str::from_utf8_unchecked(&strings[k_off..k_off + k_len])
            };

            match key_str.cmp(query) {
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
                Ordering::Equal => {
                    let cand_start = u32::from_le_bytes(data[pos + 6..pos + 10].try_into().unwrap()) as usize;
                    let cand_count = u16::from_le_bytes(data[pos + 10..pos + 12].try_into().unwrap()) as usize;

                    let mut entry = OverrideEntry::new();
                    for i in 0..cand_count.min(4) {
                        let c_idx = cand_start + i;
                        if c_idx >= self.cand_count { break; }
                        let c_pos = self.cands_offset + c_idx * Self::CAND_RECORD_LEN;
                        let b_off = u32::from_le_bytes(data[c_pos..c_pos + 4].try_into().unwrap()) as usize;
                        let b_len = u16::from_le_bytes(data[c_pos + 4..c_pos + 6].try_into().unwrap()) as usize;
                        let conf = f32::from_le_bytes(data[c_pos + 6..c_pos + 10].try_into().unwrap());

                        let bengali_str = unsafe {
                            std::str::from_utf8_unchecked(&strings[b_off..b_off + b_len])
                        };
                        entry.push(bengali_str, conf);
                    }
                    return Some(entry);
                }
            }
        }
        None
    }

    pub fn compile(map: &std::collections::HashMap<String, Vec<(String, f32)>>) -> Vec<u8> {
        Self::compile_to_bytes(map.iter().map(|(k, v)| (k.as_str(), v.as_slice())))
    }

    pub fn compile_hashbrown(map: &HashMap<String, Vec<(String, f32)>>) -> Vec<u8> {
        Self::compile_to_bytes(map.iter().map(|(k, v)| (k.as_str(), v.as_slice())))
    }

    pub fn compile_to_bytes<'a, I>(entries: I) -> Vec<u8>
    where
        I: IntoIterator<Item = (&'a str, &'a [(String, f32)])>,
    {
        let mut sorted_entries: Vec<(&'a str, &'a [(String, f32)])> = entries.into_iter().collect();
        sorted_entries.sort_by(|a, b| a.0.cmp(b.0));

        let key_count = sorted_entries.len() as u32;
        let mut cand_count = 0u32;
        for (_, cands) in &sorted_entries {
            cand_count += cands.len() as u32;
        }

        let keys_offset = Self::HEADER_LEN as u32;
        let cands_offset = keys_offset + key_count * Self::KEY_RECORD_LEN as u32;
        let strings_offset = cands_offset + cand_count * Self::CAND_RECORD_LEN as u32;

        let mut keys_bytes = Vec::with_capacity((key_count as usize) * Self::KEY_RECORD_LEN);
        let mut cands_bytes = Vec::with_capacity((cand_count as usize) * Self::CAND_RECORD_LEN);
        let mut strings_bytes = Vec::new();

        let mut current_cand_start = 0u32;

        for (latin, cands) in sorted_entries {
            let k_off = strings_bytes.len() as u32;
            let k_len = latin.len() as u16;
            strings_bytes.extend_from_slice(latin.as_bytes());

            let c_count = cands.len() as u16;

            keys_bytes.extend_from_slice(&k_off.to_le_bytes());
            keys_bytes.extend_from_slice(&k_len.to_le_bytes());
            keys_bytes.extend_from_slice(&current_cand_start.to_le_bytes());
            keys_bytes.extend_from_slice(&c_count.to_le_bytes());

            for (bn, conf) in cands {
                let b_off = strings_bytes.len() as u32;
                let b_len = bn.len() as u16;
                strings_bytes.extend_from_slice(bn.as_bytes());

                cands_bytes.extend_from_slice(&b_off.to_le_bytes());
                cands_bytes.extend_from_slice(&b_len.to_le_bytes());
                cands_bytes.extend_from_slice(&conf.to_le_bytes());
                cands_bytes.extend_from_slice(&0u16.to_le_bytes()); // padding to 12 bytes
            }

            current_cand_start += c_count as u32;
        }

        let strings_len = strings_bytes.len() as u32;

        let mut out = Vec::with_capacity(strings_offset as usize + strings_bytes.len());
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&Self::VERSION.to_le_bytes());
        out.extend_from_slice(&key_count.to_le_bytes());
        out.extend_from_slice(&cand_count.to_le_bytes());
        out.extend_from_slice(&keys_offset.to_le_bytes());
        out.extend_from_slice(&cands_offset.to_le_bytes());
        out.extend_from_slice(&strings_offset.to_le_bytes());
        out.extend_from_slice(&strings_len.to_le_bytes());
        out.resize(Self::HEADER_LEN, 0);

        out.extend_from_slice(&keys_bytes);
        out.extend_from_slice(&cands_bytes);
        out.extend_from_slice(&strings_bytes);

        out
    }
}

fn parse_legacy_v1(bytes: &[u8]) -> Option<HashMap<String, Vec<(String, f32)>>> {
    if bytes.len() < 12 || &bytes[0..4] != b"POVR" {
        return None;
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let mut cursor = 12;
    let mut map = HashMap::with_capacity(count);

    for _ in 0..count {
        if cursor + 2 > bytes.len() { break; }
        let latin_len = u16::from_le_bytes(bytes[cursor..cursor+2].try_into().ok()?) as usize;
        cursor += 2;
        if cursor + latin_len > bytes.len() { break; }
        let latin = std::str::from_utf8(&bytes[cursor..cursor+latin_len]).ok()?.to_string();
        cursor += latin_len;

        if cursor + 2 > bytes.len() { break; }
        let cand_count = u16::from_le_bytes(bytes[cursor..cursor+2].try_into().ok()?) as usize;
        cursor += 2;

        let mut cands = Vec::with_capacity(cand_count);
        for _ in 0..cand_count {
            if cursor + 2 > bytes.len() { break; }
            let bengali_len = u16::from_le_bytes(bytes[cursor..cursor+2].try_into().ok()?) as usize;
            cursor += 2;
            if cursor + bengali_len > bytes.len() { break; }
            let bengali = std::str::from_utf8(&bytes[cursor..cursor+bengali_len]).ok()?.to_string();
            cursor += bengali_len;

            if cursor + 4 > bytes.len() { break; }
            let conf = f32::from_le_bytes(bytes[cursor..cursor+4].try_into().ok()?);
            cursor += 4;
            cands.push((bengali, conf));
        }
        map.insert(latin, cands);
    }
    Some(map)
}
