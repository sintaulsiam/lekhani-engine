//! Fixed Layout Parser with Fast O(1) Key Table

use hashbrown::HashMap;
use serde_json::Value;

use crate::keycodes::*;

#[derive(Debug, Clone, Default)]
pub struct FixedLayoutParser {
    table: HashMap<(u16, KeyModifier), String>,
    pub name: String,
    pub version: String,
}

impl FixedLayoutParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_json(json: &Value) -> Self {
        let mut parser = Self::new();

        if let Some(info) = json.get("info") {
            if let Some(layout_info) = info.get("layout") {
                parser.name = layout_info
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string();
                parser.version = layout_info
                    .get("version")
                    .and_then(|v| v.as_str())
                    .unwrap_or("1.0")
                    .to_string();
            }
        }

        if let Some(layout) = json.get("layout").and_then(|v| v.as_object()) {
            for (key_str, val) in layout {
                if let Some(val_str) = val.as_str() {
                    if val_str.is_empty() {
                        continue;
                    }
                    if let Some((keycode, modifier)) = parse_key_str(key_str) {
                        parser
                            .table
                            .insert((keycode, modifier), val_str.to_string());
                    }
                }
            }
        }

        parser
    }

    pub fn get_char(&self, keycode: u16, modifier: KeyModifier) -> Option<&str> {
        self.table.get(&(keycode, modifier)).map(|s| s.as_str())
    }
}

fn parse_key_str(key_str: &str) -> Option<(u16, KeyModifier)> {
    let modifier = if key_str.ends_with("_AltGr") {
        KeyModifier::AltGr
    } else {
        KeyModifier::Normal
    };

    let base = key_str
        .strip_prefix("Key_")
        .unwrap_or(key_str)
        .strip_suffix("_Normal")
        .or_else(|| key_str.strip_prefix("Key_")?.strip_suffix("_AltGr"))
        .unwrap_or(key_str);

    let keycode = match base {
        "0" => VC_0,
        "1" => VC_1,
        "2" => VC_2,
        "3" => VC_3,
        "4" => VC_4,
        "5" => VC_5,
        "6" => VC_6,
        "7" => VC_7,
        "8" => VC_8,
        "9" => VC_9,

        "a" => VC_A,
        "b" => VC_B,
        "c" => VC_C,
        "d" => VC_D,
        "e" => VC_E,
        "f" => VC_F,
        "g" => VC_G,
        "h" => VC_H,
        "i" => VC_I,
        "j" => VC_J,
        "k" => VC_K,
        "l" => VC_L,
        "m" => VC_M,
        "n" => VC_N,
        "o" => VC_O,
        "p" => VC_P,
        "q" => VC_Q,
        "r" => VC_R,
        "s" => VC_S,
        "t" => VC_T,
        "u" => VC_U,
        "v" => VC_V,
        "w" => VC_W,
        "x" => VC_X,
        "y" => VC_Y,
        "z" => VC_Z,

        "A" => VC_A_SHIFT,
        "B" => VC_B_SHIFT,
        "C" => VC_C_SHIFT,
        "D" => VC_D_SHIFT,
        "E" => VC_E_SHIFT,
        "F" => VC_F_SHIFT,
        "G" => VC_G_SHIFT,
        "H" => VC_H_SHIFT,
        "I" => VC_I_SHIFT,
        "J" => VC_J_SHIFT,
        "K" => VC_K_SHIFT,
        "L" => VC_L_SHIFT,
        "M" => VC_M_SHIFT,
        "N" => VC_N_SHIFT,
        "O" => VC_O_SHIFT,
        "P" => VC_P_SHIFT,
        "Q" => VC_Q_SHIFT,
        "R" => VC_R_SHIFT,
        "S" => VC_S_SHIFT,
        "T" => VC_T_SHIFT,
        "U" => VC_U_SHIFT,
        "V" => VC_V_SHIFT,
        "W" => VC_W_SHIFT,
        "X" => VC_X_SHIFT,
        "Y" => VC_Y_SHIFT,
        "Z" => VC_Z_SHIFT,

        "Grave" | "BackQuote" | "`" => VC_GRAVE,
        "Tilde" | "~" => VC_TILDE,
        "Minus" | "-" => VC_MINUS,
        "UnderScore" | "_" => VC_UNDERSCORE,
        "Equals" | "=" => VC_EQUALS,
        "Plus" | "+" => VC_PLUS,
        "BracketLeft" | "OpenBracket" | "[" => VC_BRACKET_LEFT,
        "BraceLeft" | "OpenBrace" | "{" => VC_BRACE_LEFT,
        "BracketRight" | "CloseBracket" | "]" => VC_BRACKET_RIGHT,
        "BraceRight" | "CloseBrace" | "}" => VC_BRACE_RIGHT,
        "BackSlash" | "\\" => VC_BACK_SLASH,
        "Bar" | "|" => VC_BAR,
        "Semicolon" | ";" => VC_SEMICOLON,
        "Colon" | ":" => VC_COLON,
        "Apostrophe" | "'" => VC_APOSTROPHE,
        "Quote" | "\"" => VC_QUOTE,
        "Comma" | "," => VC_COMMA,
        "Less" | "<" => VC_LESS,
        "Period" | "." => VC_PERIOD,
        "Greater" | ">" => VC_GREATER,
        "Slash" | "/" => VC_SLASH,
        "Question" | "?" => VC_QUESTION,
        "Exclaim" | "!" => VC_EXCLAIM,
        "At" | "@" => VC_AT,
        "Hash" | "#" => VC_HASH,
        "Dollar" | "$" => VC_DOLLAR,
        "Percent" | "%" => VC_PERCENT,
        "Circum" | "^" => VC_CIRCUM,
        "Ampersand" | "&" => VC_AMPERSAND,
        "Asterisk" | "*" => VC_ASTERISK,
        "ParenLeft" | "(" => VC_PAREN_LEFT,
        "ParenRight" | ")" => VC_PAREN_RIGHT,

        // NumPad
        "Num0" => VC_KP_0,
        "Num1" => VC_KP_1,
        "Num2" => VC_KP_2,
        "Num3" => VC_KP_3,
        "Num4" => VC_KP_4,
        "Num5" => VC_KP_5,
        "Num6" => VC_KP_6,
        "Num7" => VC_KP_7,
        "Num8" => VC_KP_8,
        "Num9" => VC_KP_9,
        "NumDivide" => VC_KP_DIVIDE,
        "NumMultiply" => VC_KP_MULTIPLY,
        "NumSubtract" => VC_KP_SUBTRACT,
        "NumAdd" => VC_KP_ADD,
        "NumDecimal" => VC_KP_DECIMAL,

        _ => return None,
    };

    Some((keycode, modifier))
}
