//! Virtual Key Codes Definition and Modifiers

pub const VC_0: u16 = 11;
pub const VC_1: u16 = 2;
pub const VC_2: u16 = 3;
pub const VC_3: u16 = 4;
pub const VC_4: u16 = 5;
pub const VC_5: u16 = 6;
pub const VC_6: u16 = 7;
pub const VC_7: u16 = 8;
pub const VC_8: u16 = 9;
pub const VC_9: u16 = 10;

pub const VC_A: u16 = 41110;
pub const VC_B: u16 = 41111;
pub const VC_C: u16 = 41112;
pub const VC_D: u16 = 41113;
pub const VC_E: u16 = 41114;
pub const VC_F: u16 = 41115;
pub const VC_G: u16 = 41116;
pub const VC_H: u16 = 41117;
pub const VC_I: u16 = 41118;
pub const VC_J: u16 = 41119;
pub const VC_K: u16 = 41120;
pub const VC_L: u16 = 41121;
pub const VC_M: u16 = 41122;
pub const VC_N: u16 = 41123;
pub const VC_O: u16 = 41124;
pub const VC_P: u16 = 41125;
pub const VC_Q: u16 = 41126;
pub const VC_R: u16 = 41127;
pub const VC_S: u16 = 41128;
pub const VC_T: u16 = 41129;
pub const VC_U: u16 = 41130;
pub const VC_V: u16 = 41131;
pub const VC_W: u16 = 41132;
pub const VC_X: u16 = 41133;
pub const VC_Y: u16 = 41134;
pub const VC_Z: u16 = 41135;

pub const VC_A_SHIFT: u16 = 41140;
pub const VC_B_SHIFT: u16 = 41141;
pub const VC_C_SHIFT: u16 = 41142;
pub const VC_D_SHIFT: u16 = 41143;
pub const VC_E_SHIFT: u16 = 41144;
pub const VC_F_SHIFT: u16 = 41145;
pub const VC_G_SHIFT: u16 = 41146;
pub const VC_H_SHIFT: u16 = 41147;
pub const VC_I_SHIFT: u16 = 41148;
pub const VC_J_SHIFT: u16 = 41149;
pub const VC_K_SHIFT: u16 = 41150;
pub const VC_L_SHIFT: u16 = 41151;
pub const VC_M_SHIFT: u16 = 41152;
pub const VC_N_SHIFT: u16 = 41153;
pub const VC_O_SHIFT: u16 = 41154;
pub const VC_P_SHIFT: u16 = 41155;
pub const VC_Q_SHIFT: u16 = 41156;
pub const VC_R_SHIFT: u16 = 41157;
pub const VC_S_SHIFT: u16 = 41158;
pub const VC_T_SHIFT: u16 = 41159;
pub const VC_U_SHIFT: u16 = 41160;
pub const VC_V_SHIFT: u16 = 41161;
pub const VC_W_SHIFT: u16 = 41162;
pub const VC_X_SHIFT: u16 = 41163;
pub const VC_Y_SHIFT: u16 = 41164;
pub const VC_Z_SHIFT: u16 = 41165;

pub const VC_GRAVE: u16 = 41;
pub const VC_TILDE: u16 = 1;
pub const VC_MINUS: u16 = 12;
pub const VC_UNDERSCORE: u16 = 87;
pub const VC_EQUALS: u16 = 13;
pub const VC_PLUS: u16 = 88;

pub const VC_BRACKET_LEFT: u16 = 26;
pub const VC_BRACKET_RIGHT: u16 = 27;
pub const VC_BRACE_LEFT: u16 = 91;
pub const VC_BRACE_RIGHT: u16 = 92;

pub const VC_BACK_SLASH: u16 = 43;
pub const VC_BAR: u16 = 93;
pub const VC_SEMICOLON: u16 = 39;
pub const VC_COLON: u16 = 99;
pub const VC_APOSTROPHE: u16 = 40;
pub const VC_QUOTE: u16 = 100;

pub const VC_COMMA: u16 = 51;
pub const VC_LESS: u16 = 101;
pub const VC_PERIOD: u16 = 52;
pub const VC_GREATER: u16 = 102;
pub const VC_SLASH: u16 = 53;
pub const VC_QUESTION: u16 = 103;

pub const VC_PAREN_LEFT: u16 = 67;
pub const VC_PAREN_RIGHT: u16 = 68;
pub const VC_EXCLAIM: u16 = 59;
pub const VC_AT: u16 = 60;
pub const VC_HASH: u16 = 61;
pub const VC_DOLLAR: u16 = 62;
pub const VC_PERCENT: u16 = 63;
pub const VC_CIRCUM: u16 = 64;
pub const VC_AMPERSAND: u16 = 65;
pub const VC_ASTERISK: u16 = 66;

// Keypad
pub const VC_KP_0: u16 = 82;
pub const VC_KP_1: u16 = 79;
pub const VC_KP_2: u16 = 80;
pub const VC_KP_3: u16 = 81;
pub const VC_KP_4: u16 = 75;
pub const VC_KP_5: u16 = 76;
pub const VC_KP_6: u16 = 77;
pub const VC_KP_7: u16 = 71;
pub const VC_KP_8: u16 = 72;
pub const VC_KP_9: u16 = 73;

pub const VC_KP_DIVIDE: u16 = 3637;
pub const VC_KP_MULTIPLY: u16 = 55;
pub const VC_KP_SUBTRACT: u16 = 74;
pub const VC_KP_ADD: u16 = 78;
pub const VC_KP_DECIMAL: u16 = 83;
pub const VC_KP_ENTER: u16 = 3612;
pub const VC_KP_EQUALS: u16 = 3597;

pub const VC_UNKNOWN: u16 = 0x0046;

pub const MODIFIER_SHIFT: u8 = 1 << 0;
pub const MODIFIER_ALT_GR: u8 = 1 << 1;
pub const MODIFIER_CTRL: u8 = 1 << 2;
pub const MODIFIER_ALT: u8 = 1 << 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyModifier {
    Normal,
    AltGr,
}

use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct KeycodeMapper {
    map: HashMap<u32, u16>,
}

impl Default for KeycodeMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl KeycodeMapper {
    pub fn new() -> Self {
        let mut map = HashMap::new();

        // Alphanumeric Keys
        map.insert(0x0060, VC_GRAVE);
        map.insert(0xfe50, VC_GRAVE);
        map.insert(0x007e, VC_TILDE);

        map.insert(0x0030, VC_0);
        map.insert(0x0031, VC_1);
        map.insert(0x0032, VC_2);
        map.insert(0x0033, VC_3);
        map.insert(0x0034, VC_4);
        map.insert(0x0035, VC_5);
        map.insert(0x0036, VC_6);
        map.insert(0x0037, VC_7);
        map.insert(0x0038, VC_8);
        map.insert(0x0039, VC_9);

        map.insert(0x0029, VC_PAREN_RIGHT);
        map.insert(0x0021, VC_EXCLAIM);
        map.insert(0x0040, VC_AT);
        map.insert(0x0023, VC_HASH);
        map.insert(0x0024, VC_DOLLAR);
        map.insert(0x0025, VC_PERCENT);
        map.insert(0x005e, VC_CIRCUM);
        map.insert(0x0026, VC_AMPERSAND);
        map.insert(0x002a, VC_ASTERISK);
        map.insert(0x0028, VC_PAREN_LEFT);

        map.insert(0x002d, VC_MINUS);
        map.insert(0x005f, VC_UNDERSCORE);
        map.insert(0x003d, VC_EQUALS);
        map.insert(0x002b, VC_PLUS);

        // Alphabet Lowercase
        map.insert(0x0061, VC_A);
        map.insert(0x0062, VC_B);
        map.insert(0x0063, VC_C);
        map.insert(0x0064, VC_D);
        map.insert(0x0065, VC_E);
        map.insert(0x0066, VC_F);
        map.insert(0x0067, VC_G);
        map.insert(0x0068, VC_H);
        map.insert(0x0069, VC_I);
        map.insert(0x006a, VC_J);
        map.insert(0x006b, VC_K);
        map.insert(0x006c, VC_L);
        map.insert(0x006d, VC_M);
        map.insert(0x006e, VC_N);
        map.insert(0x006f, VC_O);
        map.insert(0x0070, VC_P);
        map.insert(0x0071, VC_Q);
        map.insert(0x0072, VC_R);
        map.insert(0x0073, VC_S);
        map.insert(0x0074, VC_T);
        map.insert(0x0075, VC_U);
        map.insert(0x0076, VC_V);
        map.insert(0x0077, VC_W);
        map.insert(0x0078, VC_X);
        map.insert(0x0079, VC_Y);
        map.insert(0x007a, VC_Z);

        // Alphabet Uppercase
        map.insert(0x0041, VC_A_SHIFT);
        map.insert(0x0042, VC_B_SHIFT);
        map.insert(0x0043, VC_C_SHIFT);
        map.insert(0x0044, VC_D_SHIFT);
        map.insert(0x0045, VC_E_SHIFT);
        map.insert(0x0046, VC_F_SHIFT);
        map.insert(0x0047, VC_G_SHIFT);
        map.insert(0x0048, VC_H_SHIFT);
        map.insert(0x0049, VC_I_SHIFT);
        map.insert(0x004a, VC_J_SHIFT);
        map.insert(0x004b, VC_K_SHIFT);
        map.insert(0x004c, VC_L_SHIFT);
        map.insert(0x004d, VC_M_SHIFT);
        map.insert(0x004e, VC_N_SHIFT);
        map.insert(0x004f, VC_O_SHIFT);
        map.insert(0x0050, VC_P_SHIFT);
        map.insert(0x0051, VC_Q_SHIFT);
        map.insert(0x0052, VC_R_SHIFT);
        map.insert(0x0053, VC_S_SHIFT);
        map.insert(0x0054, VC_T_SHIFT);
        map.insert(0x0055, VC_U_SHIFT);
        map.insert(0x0056, VC_V_SHIFT);
        map.insert(0x0057, VC_W_SHIFT);
        map.insert(0x0058, VC_X_SHIFT);
        map.insert(0x0059, VC_Y_SHIFT);
        map.insert(0x005a, VC_Z_SHIFT);

        // Punctuation
        map.insert(0x005b, VC_BRACKET_LEFT);
        map.insert(0x007b, VC_BRACE_LEFT);
        map.insert(0x005d, VC_BRACKET_RIGHT);
        map.insert(0x007d, VC_BRACE_RIGHT);
        map.insert(0x005c, VC_BACK_SLASH);
        map.insert(0x007c, VC_BAR);
        map.insert(0x002f, VC_SLASH);
        map.insert(0x003f, VC_QUESTION);
        map.insert(0x003b, VC_SEMICOLON);
        map.insert(0x003a, VC_COLON);
        map.insert(0x002c, VC_COMMA);
        map.insert(0x003c, VC_LESS);
        map.insert(0x002e, VC_PERIOD);
        map.insert(0x003e, VC_GREATER);
        map.insert(0x0027, VC_APOSTROPHE);
        map.insert(0x0022, VC_QUOTE);

        // NumPad
        map.insert(0xffaf, VC_KP_DIVIDE);
        map.insert(0xffaa, VC_KP_MULTIPLY);
        map.insert(0xffad, VC_KP_SUBTRACT);
        map.insert(0xffab, VC_KP_ADD);
        map.insert(0xffae, VC_KP_DECIMAL);
        map.insert(0xff8d, VC_KP_ENTER);
        map.insert(0xffbd, VC_KP_EQUALS);

        map.insert(0xffb0, VC_KP_0);
        map.insert(0xffb1, VC_KP_1);
        map.insert(0xffb2, VC_KP_2);
        map.insert(0xffb3, VC_KP_3);
        map.insert(0xffb4, VC_KP_4);
        map.insert(0xffb5, VC_KP_5);
        map.insert(0xffb6, VC_KP_6);
        map.insert(0xffb7, VC_KP_7);
        map.insert(0xffb8, VC_KP_8);
        map.insert(0xffb9, VC_KP_9);

        Self { map }
    }

    pub fn map_keyval(&self, keyval: u32) -> u16 {
        self.map.get(&keyval).copied().unwrap_or(VC_UNKNOWN)
    }
}

