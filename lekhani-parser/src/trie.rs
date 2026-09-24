//! Compact L1 Prefix Trie for Fast Phonetic Pattern Matching

pub const NO_PATTERN: u16 = u16::MAX;
pub const NO_NODE: u16 = u16::MAX;

#[derive(Debug, Clone)]
pub struct TrieNode {
    pub pattern_idx: u16,
    pub children: [u16; 128],
}

impl Default for TrieNode {
    fn default() -> Self {
        Self {
            pattern_idx: NO_PATTERN,
            children: [NO_NODE; 128],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternTrie {
    pub nodes: Vec<TrieNode>,
    pub max_depth: usize,
}

impl Default for PatternTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternTrie {
    pub fn new() -> Self {
        let mut trie = Self {
            nodes: Vec::with_capacity(384),
            max_depth: 0,
        };
        // Allocate root node (node 0)
        trie.nodes.push(TrieNode::default());
        trie
    }

    /// Insert an ASCII pattern into the Trie
    pub fn insert(&mut self, pattern_bytes: &[u8], pattern_idx: u16) {
        let mut current_idx: usize = 0;
        let mut depth = 0;

        for &b in pattern_bytes {
            let ascii_idx = (b & 0x7F) as usize;
            let next_idx = self.nodes[current_idx].children[ascii_idx];

            if next_idx == NO_NODE {
                let new_idx = self.nodes.len() as u16;
                self.nodes.push(TrieNode::default());
                self.nodes[current_idx].children[ascii_idx] = new_idx;
                current_idx = new_idx as usize;
            } else {
                current_idx = next_idx as usize;
            }
            depth += 1;
        }

        self.nodes[current_idx].pattern_idx = pattern_idx;
        if depth > self.max_depth {
            self.max_depth = depth;
        }
    }

    /// Find the longest matching pattern prefix starting at `start` in `input`.
    /// Returns `Some((pattern_index, match_len))` or `None`.
    #[inline(always)]
    pub fn longest_match(&self, input: &[u8], start: usize) -> Option<(u16, usize)> {
        let len = input.len();
        if start >= len {
            return None;
        }

        let mut current_node = 0usize;
        let mut best_match: Option<(u16, usize)> = None;
        let limit = (start + self.max_depth).min(len);

        let mut offset = start;
        while offset < limit {
            let b = input[offset];
            if b >= 128 {
                break;
            }
            let next_node = self.nodes[current_node].children[b as usize];
            if next_node == NO_NODE {
                break;
            }
            current_node = next_node as usize;
            let pat_idx = self.nodes[current_node].pattern_idx;
            if pat_idx != NO_PATTERN {
                best_match = Some((pat_idx, offset - start + 1));
            }
            offset += 1;
        }

        best_match
    }
}
