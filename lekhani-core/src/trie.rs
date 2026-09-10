//! High-Performance Prefix Trie for Dictionary Search

use hashbrown::HashMap;

#[derive(Debug, Clone, Default)]
pub struct PrefixTrie {
    children: HashMap<char, PrefixTrie>,
    values: Vec<String>,
    is_terminal: bool,
}

impl PrefixTrie {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key and its corresponding candidate representation
    pub fn insert(&mut self, key: &str, candidate: String) {
        let mut current = self;
        for ch in key.chars() {
            current = current.children.entry(ch).or_default();
        }
        current.is_terminal = true;
        if !current.values.iter().any(|v| v == &candidate) {
            current.values.push(candidate);
        }
    }

    /// Exact lookup for a key
    pub fn get_exact(&self, key: &str) -> Option<&[String]> {
        let mut current = self;
        for ch in key.chars() {
            match current.children.get(&ch) {
                Some(next) => current = next,
                None => return None,
            }
        }
        if current.is_terminal {
            Some(&current.values)
        } else {
            None
        }
    }

    /// Find all candidates whose key starts with the given prefix
    pub fn find_prefix_matches(&self, prefix: &str, limit: usize) -> Vec<String> {
        let mut current = self;
        for ch in prefix.chars() {
            match current.children.get(&ch) {
                Some(next) => current = next,
                None => return Vec::new(),
            }
        }

        let mut results = Vec::new();
        Self::collect_values(current, &mut results, limit);
        results
    }

    fn collect_values(node: &PrefixTrie, results: &mut Vec<String>, limit: usize) {
        if results.len() >= limit {
            return;
        }

        if node.is_terminal {
            for val in &node.values {
                if results.len() >= limit {
                    return;
                }
                if !results.iter().any(|r| r == val) {
                    results.push(val.clone());
                }
            }
        }

        for child in node.children.values() {
            Self::collect_values(child, results, limit);
            if results.len() >= limit {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_trie() {
        let mut trie = PrefixTrie::new();
        trie.insert("bangla", "বাংলা".to_string());
        trie.insert("bangladesh", "বাংলাদেশ".to_string());
        trie.insert("bangladeshi", "বাংলাদেশী".to_string());
        trie.insert("boi", "বই".to_string());

        assert_eq!(trie.get_exact("bangla"), Some(&["বাংলা".to_string()][..]));
        assert_eq!(trie.get_exact("unknown"), None);

        let matches = trie.find_prefix_matches("bang", 10);
        assert_eq!(matches.len(), 3);
        assert!(matches.contains(&"বাংলা".to_string()));
        assert!(matches.contains(&"বাংলাদেশ".to_string()));
        assert!(matches.contains(&"বাংলাদেশী".to_string()));
    }
}
