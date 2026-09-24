//! Phonetic Method Session Handler

use hashbrown::HashMap;
use serde_json::Value;

use super::suggestion::PhoneticSuggestion;
use crate::keycodes::*;

use std::sync::{Arc, Mutex};

use crate::ngram::UserStats;

#[derive(Debug, Clone, Default)]
pub struct PhoneticMethod {
    pub buffer: String,
    pub suggestion_engine: PhoneticSuggestion,
    pub candidate_memory: HashMap<String, String>,
    pub current_candidates: Vec<String>,
    pub selected_index: usize,
    pub default_selected_index: usize,
    pub use_dictionary: bool,
    pub include_english: bool,
    pub last_committed_word: Option<String>,
    pub recent_context: Vec<String>,
    pub stats: Arc<Mutex<UserStats>>,
    pub is_prediction_mode: bool,
    pub is_prediction_navigated: bool,
}

impl PhoneticMethod {
    pub fn new() -> Self {
        Self {
            buffer: String::with_capacity(32),
            suggestion_engine: PhoneticSuggestion::new(),
            candidate_memory: HashMap::new(),
            current_candidates: Vec::new(),
            selected_index: 0,
            default_selected_index: 0,
            use_dictionary: true,
            include_english: true,
            last_committed_word: None,
            recent_context: Vec::with_capacity(8),
            stats: Arc::new(Mutex::new(UserStats::new())),
            is_prediction_mode: false,
            is_prediction_navigated: false,
        }
    }

    pub fn with_layout(layout: &Value) -> Self {
        let mut m = Self::new();
        m.suggestion_engine.set_layout(layout);
        m
    }

    pub fn populate_predictions(&mut self) -> bool {
        if !self.buffer.is_empty() {
            return false;
        }
        let ctx_refs: Vec<&str> = self.recent_context.iter().map(|s| s.as_str()).collect();
        if !ctx_refs.is_empty() {
            let preds = self
                .suggestion_engine
                .suggest_next_words_with_context(&ctx_refs);
            if !preds.is_empty() {
                self.current_candidates = preds;
                self.selected_index = 0;
                self.is_prediction_mode = true;
                self.is_prediction_navigated = false;
                return true;
            }
        }
        self.is_prediction_mode = false;
        self.is_prediction_navigated = false;
        self.current_candidates.clear();
        false
    }

    pub fn process_key(&mut self, key: u16, _modifier: u8) -> bool {
        if self.is_prediction_mode {
            self.is_prediction_mode = false;
            self.current_candidates.clear();
            self.selected_index = 0;
        }
        if let Some(ch) = keycode_to_char(key) {
            self.buffer.push(ch);
            self.update_suggestions();
            true
        } else {
            false
        }
    }

    pub fn process_backspace(&mut self) -> bool {
        if self.is_prediction_mode {
            self.is_prediction_mode = false;
            self.current_candidates.clear();
            self.selected_index = 0;
            return false;
        }
        if !self.buffer.is_empty() {
            self.buffer.pop();
            if !self.buffer.is_empty() {
                self.update_suggestions();
            } else {
                self.current_candidates.clear();
                self.selected_index = 0;
            }
            true
        } else {
            false
        }
    }

    pub fn update_suggestions(&mut self) {
        let ctx_refs: Vec<&str> = self.recent_context.iter().map(|s| s.as_str()).collect();
        let (candidates, selected) = self.suggestion_engine.suggest_with_multi_context(
            &self.buffer,
            &ctx_refs,
            self.include_english,
            self.use_dictionary,
            &self.candidate_memory,
        );
        self.current_candidates = candidates;
        self.selected_index = selected;
        self.default_selected_index = selected;
    }

    pub fn get_buffer(&self) -> &str {
        &self.buffer
    }

    pub fn get_candidates(&self) -> &[String] {
        &self.current_candidates
    }

    pub fn get_selected_index(&self) -> usize {
        self.selected_index
    }

    pub fn set_selected_index(&mut self, index: usize) {
        if index < self.current_candidates.len() {
            self.selected_index = index;
        }
    }

    pub fn select_next(&mut self) {
        if self.is_prediction_mode {
            self.is_prediction_navigated = true;
        }
        if !self.current_candidates.is_empty() {
            self.selected_index = (self.selected_index + 1) % self.current_candidates.len();
        }
    }

    pub fn select_prev(&mut self) {
        if self.is_prediction_mode {
            self.is_prediction_navigated = true;
        }
        if !self.current_candidates.is_empty() {
            if self.selected_index == 0 {
                self.selected_index = self.current_candidates.len() - 1;
            } else {
                self.selected_index -= 1;
            }
        }
    }

    pub fn get_current_candidate(&self) -> Option<&str> {
        if self.is_prediction_mode {
            None
        } else {
            self.current_candidates
                .get(self.selected_index)
                .map(|s| s.as_str())
        }
    }

    pub fn commit(&mut self, index: usize) -> Option<String> {
        let text = self.current_candidates.get(index).cloned();
        if let Some(ref committed) = text {
            let typed_len = self.buffer.len();
            if index != self.default_selected_index && !self.buffer.is_empty() && !self.is_prediction_mode {
                self.candidate_memory
                    .insert(self.buffer.clone(), committed.clone());
                if let Ok(mut l) = self.suggestion_engine.database.learner.write() {
                    l.record_candidate_selection(&self.buffer, committed);
                }
            }
            let prev_word = self.last_committed_word.clone();
            self.suggestion_engine
                .observe_committed(prev_word.as_deref(), committed);
            if let Ok(mut stats) = self.stats.lock() {
                stats.record_commit(typed_len, committed);
            }
            self.recent_context.push(committed.clone());
            if self.recent_context.len() > 6 {
                self.recent_context.remove(0);
            }
            self.last_committed_word = Some(committed.clone());
        }
        self.reset();
        text
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.current_candidates.clear();
        self.selected_index = 0;
        self.default_selected_index = 0;
        self.is_prediction_mode = false;
        self.is_prediction_navigated = false;
    }

    pub fn clear_context(&mut self) {
        self.last_committed_word = None;
        self.recent_context.clear();
        self.reset();
    }

    pub fn is_active(&self) -> bool {
        !self.buffer.is_empty() || self.is_prediction_mode
    }
}

fn keycode_to_char(key: u16) -> Option<char> {
    match key {
        VC_0 => Some('0'),
        VC_1 => Some('1'),
        VC_2 => Some('2'),
        VC_3 => Some('3'),
        VC_4 => Some('4'),
        VC_5 => Some('5'),
        VC_6 => Some('6'),
        VC_7 => Some('7'),
        VC_8 => Some('8'),
        VC_9 => Some('9'),

        VC_A => Some('a'),
        VC_B => Some('b'),
        VC_C => Some('c'),
        VC_D => Some('d'),
        VC_E => Some('e'),
        VC_F => Some('f'),
        VC_G => Some('g'),
        VC_H => Some('h'),
        VC_I => Some('i'),
        VC_J => Some('j'),
        VC_K => Some('k'),
        VC_L => Some('l'),
        VC_M => Some('m'),
        VC_N => Some('n'),
        VC_O => Some('o'),
        VC_P => Some('p'),
        VC_Q => Some('q'),
        VC_R => Some('r'),
        VC_S => Some('s'),
        VC_T => Some('t'),
        VC_U => Some('u'),
        VC_V => Some('v'),
        VC_W => Some('w'),
        VC_X => Some('x'),
        VC_Y => Some('y'),
        VC_Z => Some('z'),

        VC_A_SHIFT => Some('A'),
        VC_B_SHIFT => Some('B'),
        VC_C_SHIFT => Some('C'),
        VC_D_SHIFT => Some('D'),
        VC_E_SHIFT => Some('E'),
        VC_F_SHIFT => Some('F'),
        VC_G_SHIFT => Some('G'),
        VC_H_SHIFT => Some('H'),
        VC_I_SHIFT => Some('I'),
        VC_J_SHIFT => Some('J'),
        VC_K_SHIFT => Some('K'),
        VC_L_SHIFT => Some('L'),
        VC_M_SHIFT => Some('M'),
        VC_N_SHIFT => Some('N'),
        VC_O_SHIFT => Some('O'),
        VC_P_SHIFT => Some('P'),
        VC_Q_SHIFT => Some('Q'),
        VC_R_SHIFT => Some('R'),
        VC_S_SHIFT => Some('S'),
        VC_T_SHIFT => Some('T'),
        VC_U_SHIFT => Some('U'),
        VC_V_SHIFT => Some('V'),
        VC_W_SHIFT => Some('W'),
        VC_X_SHIFT => Some('X'),
        VC_Y_SHIFT => Some('Y'),
        VC_Z_SHIFT => Some('Z'),

        VC_GRAVE => Some('`'),
        VC_TILDE => Some('~'),
        VC_EXCLAIM => Some('!'),
        VC_AT => Some('@'),
        VC_HASH => Some('#'),
        VC_DOLLAR => Some('$'),
        VC_PERCENT => Some('%'),
        VC_CIRCUM => Some('^'),
        VC_AMPERSAND => Some('&'),
        VC_ASTERISK => Some('*'),
        VC_PAREN_LEFT => Some('('),
        VC_PAREN_RIGHT => Some(')'),

        VC_MINUS => Some('-'),
        VC_UNDERSCORE => Some('_'),
        VC_EQUALS => Some('='),
        VC_PLUS => Some('+'),

        VC_BRACKET_LEFT => Some('['),
        VC_BRACKET_RIGHT => Some(']'),
        VC_BRACE_LEFT => Some('{'),
        VC_BRACE_RIGHT => Some('}'),
        VC_BACK_SLASH => Some('\\'),
        VC_BAR => Some('|'),

        VC_SEMICOLON => Some(';'),
        VC_COLON => Some(':'),
        VC_APOSTROPHE => Some('\''),
        VC_QUOTE => Some('"'),

        VC_COMMA => Some(','),
        VC_LESS => Some('<'),
        VC_PERIOD => Some('.'),
        VC_GREATER => Some('>'),
        VC_SLASH => Some('/'),
        VC_QUESTION => Some('?'),

        // NumPad
        VC_KP_0 => Some('0'),
        VC_KP_1 => Some('1'),
        VC_KP_2 => Some('2'),
        VC_KP_3 => Some('3'),
        VC_KP_4 => Some('4'),
        VC_KP_5 => Some('5'),
        VC_KP_6 => Some('6'),
        VC_KP_7 => Some('7'),
        VC_KP_8 => Some('8'),
        VC_KP_9 => Some('9'),
        VC_KP_DECIMAL => Some('.'),
        VC_KP_DIVIDE => Some('/'),
        VC_KP_MULTIPLY => Some('*'),
        VC_KP_SUBTRACT => Some('-'),
        VC_KP_ADD => Some('+'),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_navigation_learns_candidate() {
        let mut method = PhoneticMethod::new();
        method.process_key(VC_A, 0);
        method.process_key(VC_M, 0);
        method.process_key(VC_I, 0);

        assert!(!method.current_candidates.is_empty());
        assert_eq!(method.default_selected_index, 0);

        // Arrow down to candidate 1
        method.select_next();
        assert_eq!(method.selected_index, 1);
        let chosen = method.current_candidates[1].clone();

        // Commit candidate 1 (as Space/Enter does with self.selected_index)
        let committed = method.commit(method.selected_index).unwrap();
        assert_eq!(committed, chosen);

        // Verify candidate memory now has the user's explicit preference
        assert_eq!(method.candidate_memory.get("ami"), Some(&chosen));
    }
}

