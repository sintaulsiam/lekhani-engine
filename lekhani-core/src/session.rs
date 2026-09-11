//! Unified Input Session Engine

use serde_json::Value;

use crate::fixed::FixedMethod;
use crate::phonetic::PhoneticMethod;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveLayoutType {
    Phonetic,
    Fixed,
}

#[derive(Debug, Clone)]
pub struct InputSession {
    pub active_layout_type: ActiveLayoutType,
    pub phonetic: PhoneticMethod,
    pub fixed: FixedMethod,
}

impl Default for InputSession {
    fn default() -> Self {
        Self::new()
    }
}

impl InputSession {
    pub fn new() -> Self {
        Self {
            active_layout_type: ActiveLayoutType::Phonetic,
            phonetic: PhoneticMethod::new(),
            fixed: FixedMethod::new(),
        }
    }

    pub fn load_database<P: AsRef<std::path::Path>>(&mut self, dir: P) {
        let _ = self.phonetic.suggestion_engine.database.load_from_dir(dir);
    }

    pub fn load_user_autocorrect<P: AsRef<std::path::Path>>(&mut self, path: P) {
        self.phonetic.suggestion_engine.database.load_user_autocorrect(path);
    }

    pub fn load_user_learned<P: AsRef<std::path::Path>>(&mut self, path: P) {
        self.phonetic.suggestion_engine.database.load_user_learned(path);
    }

    pub fn save_user_learned<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), std::io::Error> {
        self.phonetic.suggestion_engine.database.save_user_learned(path)
    }

    pub fn load_stats<P: AsRef<std::path::Path>>(&mut self, path: P) {
        self.phonetic.stats = crate::ngram::UserStats::load_from_path(path);
    }

    pub fn save_stats<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), std::io::Error> {
        self.phonetic.stats.save_to_path(path)
    }

    pub fn get_stats(&self) -> &crate::ngram::UserStats {
        &self.phonetic.stats
    }

    pub fn get_stats_mut(&mut self) -> &mut crate::ngram::UserStats {
        &mut self.phonetic.stats
    }

    pub fn set_layout(&mut self, layout_type: ActiveLayoutType, layout_json: &Value) {
        self.active_layout_type = layout_type;
        match layout_type {
            ActiveLayoutType::Phonetic => {
                self.phonetic.suggestion_engine.set_layout(layout_json);
            }
            ActiveLayoutType::Fixed => {
                self.fixed.set_layout(layout_json);
            }
        }
    }

    pub fn process_key(&mut self, keycode: u16, modifier_mask: u8) -> bool {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.process_key(keycode, modifier_mask),
            ActiveLayoutType::Fixed => self.fixed.process_key(keycode, modifier_mask).is_some(),
        }
    }

    pub fn process_backspace(&mut self) -> bool {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.process_backspace(),
            ActiveLayoutType::Fixed => self.fixed.process_backspace(),
        }
    }

    pub fn get_preedit_text(&self) -> String {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => {
                self.phonetic.get_current_candidate().unwrap_or("").to_string()
            }
            ActiveLayoutType::Fixed => {
                self.fixed.get_buffer().to_string()
            }
        }
    }

    pub fn get_auxiliary_text(&self) -> String {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.get_buffer().to_string(),
            ActiveLayoutType::Fixed => String::new(),
        }
    }

    pub fn get_candidates(&self) -> &[String] {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.get_candidates(),
            ActiveLayoutType::Fixed => &[],
        }
    }

    pub fn get_selected_index(&self) -> usize {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.get_selected_index(),
            ActiveLayoutType::Fixed => 0,
        }
    }

    pub fn select_next(&mut self) {
        if self.active_layout_type == ActiveLayoutType::Phonetic {
            self.phonetic.select_next();
        }
    }

    pub fn select_prev(&mut self) {
        if self.active_layout_type == ActiveLayoutType::Phonetic {
            self.phonetic.select_prev();
        }
    }

    pub fn commit(&mut self, index: usize) -> Option<String> {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.commit(index),
            ActiveLayoutType::Fixed => {
                let committed = self.fixed.commit();
                if !committed.is_empty() {
                    self.phonetic.stats.record_commit(committed.len(), &committed);
                    Some(committed)
                } else {
                    None
                }
            }
        }
    }

    pub fn get_next_word_predictions(&self) -> Vec<String> {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => {
                if let Some(ref last) = self.phonetic.last_committed_word {
                    self.phonetic.suggestion_engine.suggest_next_words(last)
                } else {
                    Vec::new()
                }
            }
            ActiveLayoutType::Fixed => Vec::new(),
        }
    }

    pub fn clear_context(&mut self) {
        self.phonetic.clear_context();
        self.fixed.reset();
    }

    pub fn is_prediction_mode(&self) -> bool {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.is_prediction_mode,
            ActiveLayoutType::Fixed => false,
        }
    }

    pub fn populate_predictions(&mut self) -> bool {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.populate_predictions(),
            ActiveLayoutType::Fixed => false,
        }
    }

    pub fn reset(&mut self) {
        self.phonetic.reset();
        self.fixed.reset();
    }

    pub fn is_active(&self) -> bool {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.is_active(),
            ActiveLayoutType::Fixed => self.fixed.is_active(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keycodes::*;

    #[test]
    fn test_zero_preedit_next_word_predictions() {
        let mut session = InputSession::new();
        // Type "ami" (VC_A=30, VC_M=50, VC_I=23)
        session.process_key(VC_A, 0);
        session.process_key(VC_M, 0);
        session.process_key(VC_I, 0);
        assert!(session.is_active());

        // Commit "আমি"
        let committed = session.commit(0);
        assert_eq!(committed, Some("আমি".to_string()));

        // Populate next-word predictions in zero-preedit state
        let has_preds = session.populate_predictions();
        assert!(has_preds);
        assert!(session.is_prediction_mode());
        assert!(session.is_active());
        let cands = session.get_candidates();
        assert!(!cands.is_empty());

        // Selecting candidate index 1 (e.g. "তোমাকে") commits it
        let next_committed = session.commit(1);
        assert!(next_committed.is_some());
        assert!(!session.is_prediction_mode());
    }
}
