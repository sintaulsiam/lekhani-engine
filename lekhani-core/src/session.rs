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
        self.phonetic
            .suggestion_engine
            .database
            .load_user_autocorrect(path);
    }

    pub fn load_user_learned<P: AsRef<std::path::Path>>(&mut self, path: P) {
        self.phonetic
            .suggestion_engine
            .database
            .load_user_learned(path);
        self.phonetic.candidate_memory = self
            .phonetic
            .suggestion_engine
            .database
            .learner
            .candidate_memory
            .clone();
    }

    pub fn save_user_learned<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), std::io::Error> {
        self.phonetic
            .suggestion_engine
            .database
            .save_user_learned(path)
    }

    pub fn clear_user_learned<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> Result<(), std::io::Error> {
        self.phonetic
            .suggestion_engine
            .database
            .learner
            .clear_user_data();
        self.phonetic.candidate_memory.clear();
        self.phonetic
            .suggestion_engine
            .database
            .save_user_learned(path)
    }

    pub fn get_learned_counts(&self) -> (usize, usize) {
        let l = &self.phonetic.suggestion_engine.database.learner;
        (l.learned_words.len(), l.user_bigrams.len())
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

    pub fn update_suggestion_config(&mut self, config: crate::PhoneticSuggestionConfig) {
        self.phonetic.suggestion_engine.update_config(config);
    }

    pub fn update_fixed_config(
        &mut self,
        auto_vowel: bool,
        auto_chandra: bool,
        traditional_kar: bool,
        old_reph: bool,
        numberpad: bool,
    ) {
        self.fixed.update_config(
            auto_vowel,
            auto_chandra,
            traditional_kar,
            old_reph,
            numberpad,
        );
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
            ActiveLayoutType::Phonetic => self
                .phonetic
                .get_current_candidate()
                .unwrap_or("")
                .to_string(),
            ActiveLayoutType::Fixed => self.fixed.get_buffer().to_string(),
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
                    self.phonetic
                        .stats
                        .record_commit(committed.len(), &committed);
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

    pub fn is_prediction_navigated(&self) -> bool {
        match self.active_layout_type {
            ActiveLayoutType::Phonetic => self.phonetic.is_prediction_navigated,
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
        let layout_candidates = [
            std::path::Path::new("../../data/layouts/avrophonetic.json"),
            std::path::Path::new("data/layouts/avrophonetic.json"),
            std::path::Path::new("../data/layouts/avrophonetic.json"),
        ];
        for p in layout_candidates {
            if p.exists() {
                if let Ok(content) = std::fs::read_to_string(p) {
                    if let Ok(json) = serde_json::from_str(&content) {
                        session.set_layout(ActiveLayoutType::Phonetic, &json);
                        break;
                    }
                }
            }
        }
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

    #[test]
    fn test_session_fixed_layout_typing() {
        let mut session = InputSession::new();
        let unijoy_raw = include_str!("../../../data/layouts/Unijoy.json");
        let val: serde_json::Value =
            serde_json::from_str(unijoy_raw).expect("Unijoy JSON parse failed");
        session.set_layout(ActiveLayoutType::Fixed, &val);

        // 'h' = ব, 'f' = া, 'j' = ক -> "বাক"
        session.process_key(VC_H, 0);
        assert_eq!(session.get_preedit_text(), "ব");
        session.process_key(VC_F, 0);
        assert_eq!(session.get_preedit_text(), "বা");
        session.process_key(VC_J, 0);
        assert_eq!(session.get_preedit_text(), "বাক");

        // Backspace removes 'ক'
        session.process_backspace();
        assert_eq!(session.get_preedit_text(), "বা");
    }

    #[test]
    fn test_typing_tomar() {
        let mut session = InputSession::new();
        let avro_raw = include_str!("../../../data/layouts/avrophonetic.json");
        let val: serde_json::Value = serde_json::from_str(avro_raw).unwrap();
        session.set_layout(ActiveLayoutType::Phonetic, &val);

        let parser = rupantor::parser::PhoneticParser::new(val.get("layout").unwrap_or(&val));
        println!("parser to: {:?}", parser.convert("to"));
        println!("parser tom: {:?}", parser.convert("tom"));
        println!("parser toma: {:?}", parser.convert("toma"));
        println!("parser tomar: {:?}", parser.convert("tomar"));
        println!("parser towmar: {:?}", parser.convert("towmar"));
        println!("parser twmar: {:?}", parser.convert("twmar"));
        println!("parser tO: {:?}", parser.convert("tO"));

        session.process_key(VC_T, 0);
        println!("after t: preedit={:?}, cands={:?}", session.get_preedit_text(), session.get_candidates());
        session.process_key(VC_O, 0);
        println!("after to: preedit={:?}, cands={:?}", session.get_preedit_text(), session.get_candidates());
        session.process_key(VC_M, 0);
        println!("after tom: preedit={:?}, cands={:?}", session.get_preedit_text(), session.get_candidates());
        session.process_key(VC_A, 0);
        println!("after toma: preedit={:?}, cands={:?}", session.get_preedit_text(), session.get_candidates());
        session.process_key(VC_R, 0);
        println!("after tomar: preedit={:?}, cands={:?}", session.get_preedit_text(), session.get_candidates());

        let dict_candidates = [
            std::path::Path::new("../../data/dictionaries"),
            std::path::Path::new("data/dictionaries"),
            std::path::Path::new("../data/dictionaries"),
        ];
        for p in dict_candidates {
            if p.exists() {
                session.load_database(p);
                break;
            }
        }
        let mut session2 = InputSession::new();
        session2.set_layout(ActiveLayoutType::Phonetic, &val);
        for p in dict_candidates {
            if p.exists() {
                session2.load_database(p);
                break;
            }
        }
        session2.process_key(VC_T, 0);
        println!("with dict after t: preedit={:?}, cands={:?}", session2.get_preedit_text(), session2.get_candidates());
        session2.process_key(VC_O, 0);
        println!("with dict after to: preedit={:?}, cands={:?}", session2.get_preedit_text(), session2.get_candidates());
        session2.process_key(VC_M, 0);
        println!("with dict after tom: preedit={:?}, cands={:?}", session2.get_preedit_text(), session2.get_candidates());
        session2.process_key(VC_A, 0);
        println!("with dict after toma: preedit={:?}, cands={:?}", session2.get_preedit_text(), session2.get_candidates());
        session2.process_key(VC_R, 0);
        println!("with dict after tomar: preedit={:?}, cands={:?}", session2.get_preedit_text(), session2.get_candidates());

        let mut sugg = crate::phonetic::PhoneticSuggestion::new();
        sugg.set_layout(&val);
        for p in dict_candidates {
            if p.exists() {
                let _ = sugg.database.load_from_dir(p);
                break;
            }
        }
        let empty_memory = hashbrown::HashMap::new();
        let (cands_towmar, _) = sugg.suggest("towmar", true, true, &empty_memory);
        println!("suggest('towmar') = {:?}", cands_towmar);
        let (cands_to, _) = sugg.suggest("to", true, true, &empty_memory);
        println!("suggest('to') = {:?}", cands_to);
    }

    #[test]
    fn test_session_word_pair_and_candidate_learning() {
        let mut session = InputSession::new();
        session.active_layout_type = ActiveLayoutType::Phonetic;

        // 1. Consecutive word commit
        session.phonetic.current_candidates = vec!["আমি".to_string()];
        session.commit(0);
        assert_eq!(session.phonetic.last_committed_word, Some("আমি".to_string()));

        session.phonetic.current_candidates = vec!["খাব".to_string()];
        session.commit(0);
        assert_eq!(session.phonetic.last_committed_word, Some("খাব".to_string()));

        // Check user bigram was recorded
        assert_eq!(
            session
                .phonetic
                .suggestion_engine
                .database
                .learner
                .user_bigrams
                .get("আমি\tখাব"),
            Some(&1)
        );

        // Check next-word suggestion promotion
        let next_words = session.phonetic.suggestion_engine.suggest_next_words_with_context(&["আমি"]);
        assert!(next_words.contains(&"খাব".to_string()));
        assert_eq!(next_words[0], "খাব");

        // 2. Candidate memory preference override and custom word learning
        session.phonetic.buffer = "siam".to_string();
        session.phonetic.selected_index = 0;
        session.phonetic.current_candidates = vec!["সিয়াম".to_string(), "সায়াম".to_string()];
        session.commit(1); // pick index 1 ("সায়াম")

        assert_eq!(session.phonetic.candidate_memory.get("siam"), Some(&"সায়াম".to_string()));
        assert_eq!(
            session
                .phonetic
                .suggestion_engine
                .database
                .learner
                .candidate_memory
                .get("siam"),
            Some(&"সায়াম".to_string())
        );

        // 3. Serialization and Deserialization round-trip
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("lekhani_test_learning_roundtrip.json");
        session.save_user_learned(&temp_file).unwrap();

        let mut session2 = InputSession::new();
        session2.load_user_learned(&temp_file);
        assert_eq!(session2.phonetic.candidate_memory.get("siam"), Some(&"সায়াম".to_string()));
        assert_eq!(
            session2
                .phonetic
                .suggestion_engine
                .database
                .learner
                .user_bigrams
                .get("আমি\tখাব"),
            Some(&1)
        );

        let (words_cnt, bigrams_cnt) = session2.get_learned_counts();
        assert!(words_cnt > 0);
        assert!(bigrams_cnt > 0);

        // 4. Clear learned data
        session2.clear_user_learned(&temp_file).unwrap();
        assert_eq!(session2.phonetic.candidate_memory.get("siam"), None);
        let _ = std::fs::remove_file(&temp_file);
    }
}
