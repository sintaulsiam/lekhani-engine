//! Snippets & Macro Expansion Engine

use hashbrown::HashMap;

#[derive(Debug, Clone, Default)]
pub struct SnippetManager {
    user_snippets: HashMap<String, String>,
}

impl SnippetManager {
    pub fn new() -> Self {
        let mut user_snippets = HashMap::new();
        user_snippets.insert("!shubhechha".to_string(), "আন্তরিক শুভেচ্ছা ও অভিনন্দন".to_string());
        user_snippets.insert("!dhonnobad".to_string(), "আপনাকে অনেক অনেক ধন্যবাদ".to_string());
        user_snippets.insert("!shagotom".to_string(), "স্বাগতম".to_string());
        Self { user_snippets }
    }

    pub fn insert_snippet(&mut self, trigger: String, expansion: String) {
        self.user_snippets.insert(trigger, expansion);
    }

    pub fn remove_snippet(&mut self, trigger: &str) -> Option<String> {
        self.user_snippets.remove(trigger)
    }

    pub fn get_snippets(&self) -> &HashMap<String, String> {
        &self.user_snippets
    }

    /// Check if word matches a dynamic macro or snippet
    pub fn expand(&self, word: &str) -> Option<String> {
        if let Some(val) = self.user_snippets.get(word) {
            return Some(val.clone());
        }

        // Built-in Dynamic Macros
        match word {
            "#tarikh" | "#date" => {
                Some("১১ সেপ্টেম্বর ২০২৬".to_string())
            }
            "#shomoy" | "#time" => {
                Some("০৩:১৫".to_string())
            }
            _ => None,
        }
    }
}
