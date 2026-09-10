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
