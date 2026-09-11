//! Fixed Layout Method Session & Automation Engine

use serde_json::Value;

use super::parser::FixedLayoutParser;
use crate::chars::*;
use crate::keycodes::*;

const MARKS: &str = "`~!@#$%^+*-_=+\\|\"/;:,./?><()[]{}";

#[derive(Debug, Clone, Default)]
pub struct FixedMethod {
    buffer: String,
    pub parser: FixedLayoutParser,
    pub auto_vowel: bool,
    pub auto_chandra: bool,
    pub traditional_kar: bool,
    pub old_reph: bool,
    pub numberpad: bool,
}

impl FixedMethod {
    pub fn new() -> Self {
        Self {
            buffer: String::with_capacity(32),
            parser: FixedLayoutParser::new(),
            auto_vowel: true,
            auto_chandra: true,
            traditional_kar: false,
            old_reph: true,
            numberpad: true,
        }
    }

    pub fn with_layout(layout: &Value) -> Self {
        let mut m = Self::new();
        if let Ok(parser) = FixedLayoutParser::from_json(layout) {
            m.parser = parser;
        }
        m
    }

    pub fn set_layout(&mut self, layout: &Value) {
        if let Ok(parser) = FixedLayoutParser::from_json(layout) {
            self.parser = parser;
        }
    }

    pub fn process_key(&mut self, keycode: u16, modifier_mask: u8) -> Option<String> {
        let modifier = if (modifier_mask & MODIFIER_ALT_GR) != 0 {
            KeyModifier::AltGr
        } else {
            KeyModifier::Normal
        };

        let val = self.parser.get_char(keycode, modifier).map(|s| s.to_string())
            .or_else(|| {
                if self.numberpad {
                    match keycode {
                        VC_KP_0 => Some("০".to_string()),
                        VC_KP_1 => Some("১".to_string()),
                        VC_KP_2 => Some("২".to_string()),
                        VC_KP_3 => Some("৩".to_string()),
                        VC_KP_4 => Some("৪".to_string()),
                        VC_KP_5 => Some("৫".to_string()),
                        VC_KP_6 => Some("৬".to_string()),
                        VC_KP_7 => Some("৭".to_string()),
                        VC_KP_8 => Some("৮".to_string()),
                        VC_KP_9 => Some("৯".to_string()),
                        VC_KP_DECIMAL => Some(".".to_string()),
                        _ => None,
                    }
                } else {
                    None
                }
            });

        if let Some(val) = val {
            self.process_value(&val);
            Some(self.buffer.clone())
        } else {
            None
        }
    }

    fn process_value(&mut self, value: &str) {
        let rmc = self.buffer.chars().last().unwrap_or_default();

        // 1. Zo-fola (্য)
        if value == "\u{09CD}\u{09AF}" {
            if rmc == B_R && self.buffer.chars().rev().nth(1).unwrap_or_default() != B_HASANTA {
                self.buffer.push(ZWJ);
            }
            self.buffer.push_str(value);
            return;
        }

        // 2. Old-style Reph (র্)
        if value == "\u{09B0}\u{09CD}" && self.old_reph {
            self.insert_old_style_reph();
            return;
        }

        if let Some(first_char) = value.chars().next() {
            // 3. Kar insertion & Automatic Vowel Forming
            if first_char.is_kar() {
                if self.auto_vowel && (self.buffer.is_empty() || rmc.is_vowel() || MARKS.contains(rmc)) {
                    match first_char {
                        B_AA_KAR => self.buffer.push(B_AA),
                        B_I_KAR => self.buffer.push(B_I),
                        B_II_KAR => self.buffer.push(B_II),
                        B_U_KAR => self.buffer.push(B_U),
                        B_UU_KAR => self.buffer.push(B_UU),
                        B_RRI_KAR => self.buffer.push(B_RRI),
                        B_E_KAR => self.buffer.push(B_E),
                        B_OI_KAR => self.buffer.push(B_OI),
                        B_O_KAR => self.buffer.push(B_O),
                        B_OU_KAR => self.buffer.push(B_OU),
                        _ => self.buffer.push(first_char),
                    }
                    return;
                } else if self.auto_chandra && rmc == B_CHANDRA {
                    self.buffer.pop();
                    self.buffer.push(first_char);
                    self.buffer.push(B_CHANDRA);
                    return;
                } else if rmc == B_HASANTA {
                    match first_char {
                        B_AA_KAR => { self.buffer.pop(); self.buffer.push(B_AA); }
                        B_I_KAR => { self.buffer.pop(); self.buffer.push(B_I); }
                        B_II_KAR => { self.buffer.pop(); self.buffer.push(B_II); }
                        B_U_KAR => { self.buffer.pop(); self.buffer.push(B_U); }
                        B_UU_KAR => { self.buffer.pop(); self.buffer.push(B_UU); }
                        B_RRI_KAR => { self.buffer.pop(); self.buffer.push(B_RRI); }
                        B_E_KAR => { self.buffer.pop(); self.buffer.push(B_E); }
                        B_OI_KAR => { self.buffer.pop(); self.buffer.push(B_OI); }
                        B_O_KAR => { self.buffer.pop(); self.buffer.push(B_O); }
                        B_OU_KAR => { self.buffer.pop(); self.buffer.push(B_OU); }
                        _ => self.buffer.push(first_char),
                    }
                    return;
                } else if self.traditional_kar && rmc.is_pure_consonant() && is_ligature_kar(first_char) {
                    self.buffer.push(ZWNJ);
                    self.buffer.push(first_char);
                    return;
                }
            }

            // Hasanta + Hasanta = ZWNJ
            if first_char == B_HASANTA && rmc == B_HASANTA {
                self.buffer.push(ZWNJ);
                return;
            }
        }

        self.buffer.push_str(value);
    }

    fn insert_old_style_reph(&mut self) {
        if self.buffer.is_empty() {
            self.buffer.push(B_R);
            self.buffer.push(B_HASANTA);
            return;
        }

        let len = self.buffer.chars().count();
        let mut step = 0;
        let mut constant = false;
        let mut hasanta = false;
        let mut vowel = false;
        let mut chandra = false;

        for (index, ch) in self.buffer.chars().rev().enumerate() {
            if ch.is_pure_consonant() {
                if constant && !hasanta {
                    break;
                }
                constant = true;
                hasanta = false;
                step += 1;
            } else if ch == B_HASANTA {
                hasanta = true;
                step += 1;
            } else if ch.is_vowel() {
                if vowel {
                    break;
                }
                if index == 0 || chandra {
                    vowel = true;
                    step += 1;
                    continue;
                }
                break;
            } else if ch == B_CHANDRA {
                if index == 0 {
                    chandra = true;
                    step += 1;
                    continue;
                }
                break;
            } else {
                break;
            }
        }

        let suffix: String = self.buffer.chars().skip(len - step).collect();
        let truncate_len = self.buffer.chars().take(len - step).map(|c| c.len_utf8()).sum();
        self.buffer.truncate(truncate_len);
        self.buffer.push(B_R);
        self.buffer.push(B_HASANTA);
        self.buffer.push_str(&suffix);
    }

    pub fn process_backspace(&mut self) -> bool {
        if !self.buffer.is_empty() {
            self.buffer.pop();
            true
        } else {
            false
        }
    }

    pub fn get_buffer(&self) -> &str {
        &self.buffer
    }

    pub fn commit(&mut self) -> String {
        let res = self.buffer.clone();
        self.buffer.clear();
        res
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    pub fn is_active(&self) -> bool {
        !self.buffer.is_empty()
    }
}

fn is_ligature_kar(c: char) -> bool {
    c == B_U_KAR || c == B_UU_KAR || c == B_RRI_KAR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reph_insertion() {
        let mut method = FixedMethod::new();
        method.buffer = "ক".to_string();
        method.insert_old_style_reph();
        assert_eq!(method.buffer, "র্ক");

        method.buffer = "কত".to_string();
        method.insert_old_style_reph();
        assert_eq!(method.buffer, "কর্ত");
    }

    #[test]
    fn test_unijoy_layout() {
        let unijoy_raw = include_str!("../../../../data/layouts/Unijoy.json");
        let val: serde_json::Value = serde_json::from_str(unijoy_raw).expect("Unijoy JSON parse failed");
        let mut method = FixedMethod::with_layout(&val);

        // 'h' = ব, 'f' = া, 'v' = র, 'f' = া (বাংলা / বারা)
        assert_eq!(method.process_key(VC_H, 0), Some("ব".to_string()));
        assert_eq!(method.process_key(VC_F, 0), Some("বা".to_string()));
        assert_eq!(method.process_key(VC_J, 0), Some("বাক".to_string()));
        assert_eq!(method.commit(), "বাক");
    }
}
