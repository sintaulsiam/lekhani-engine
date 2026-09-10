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
                    Some(committed)
                } else {
                    None
                }
            }
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
