//! Unicode Bengali Characters and Helper Functions

// Various signs
pub const B_SIGN_ANJI: char = '\u{0980}';
pub const B_CHANDRA: char = '\u{0981}';
pub const B_ANUSHAR: char = '\u{0982}';
pub const B_BISHARGA: char = '\u{0983}';

// Independent vowels
pub const B_A: char = '\u{0985}';
pub const B_AA: char = '\u{0986}';
pub const B_I: char = '\u{0987}';
pub const B_II: char = '\u{0988}';
pub const B_U: char = '\u{0989}';
pub const B_UU: char = '\u{098A}';
pub const B_RRI: char = '\u{098B}';
pub const B_VOCALIC_L: char = '\u{098C}';
pub const B_E: char = '\u{098F}';
pub const B_OI: char = '\u{0990}';
pub const B_O: char = '\u{0993}';
pub const B_OU: char = '\u{0994}';

// Consonants
pub const B_K: char = '\u{0995}';
pub const B_KH: char = '\u{0996}';
pub const B_G: char = '\u{0997}';
pub const B_GH: char = '\u{0998}';
pub const B_NGA: char = '\u{0999}';
pub const B_C: char = '\u{099A}';
pub const B_CH: char = '\u{099B}';
pub const B_J: char = '\u{099C}';
pub const B_JH: char = '\u{099D}';
pub const B_NYA: char = '\u{099E}';
pub const B_TT: char = '\u{099F}';
pub const B_TTH: char = '\u{09A0}';
pub const B_DD: char = '\u{09A1}';
pub const B_DDH: char = '\u{09A2}';
pub const B_NN: char = '\u{09A3}';
pub const B_T: char = '\u{09A4}';
pub const B_TH: char = '\u{09A5}';
pub const B_D: char = '\u{09A6}';
pub const B_DH: char = '\u{09A7}';
pub const B_N: char = '\u{09A8}';
pub const B_P: char = '\u{09AA}';
pub const B_PH: char = '\u{09AB}';
pub const B_B: char = '\u{09AC}';
pub const B_BH: char = '\u{09AD}';
pub const B_M: char = '\u{09AE}';
pub const B_Z: char = '\u{09AF}';
pub const B_R: char = '\u{09B0}';
pub const B_L: char = '\u{09B2}';
pub const B_SH: char = '\u{09B6}';
pub const B_SS: char = '\u{09B7}';
pub const B_S: char = '\u{09B8}';
pub const B_H: char = '\u{09B9}';

// Signs
pub const B_SIGN_NUKTA: char = '\u{09BC}';
pub const B_SIGN_AVAGRAHA: char = '\u{09BD}';

// Dependent vowel signs (Kars)
pub const B_AA_KAR: char = '\u{09BE}';
pub const B_I_KAR: char = '\u{09BF}';
pub const B_II_KAR: char = '\u{09C0}';
pub const B_U_KAR: char = '\u{09C1}';
pub const B_UU_KAR: char = '\u{09C2}';
pub const B_RRI_KAR: char = '\u{09C3}';
pub const B_VOCALIC_RR: char = '\u{09C4}';
pub const B_E_KAR: char = '\u{09C7}';
pub const B_OI_KAR: char = '\u{09C8}';
pub const B_O_KAR: char = '\u{09CB}';
pub const B_OU_KAR: char = '\u{09CC}';

// Virama / Hasant
pub const B_HASANTA: char = '\u{09CD}';
pub const B_KHANDATTA: char = '\u{09CE}';
pub const B_LENGTH_MARK: char = '\u{09D7}';

// Additional consonants
pub const B_RR: char = '\u{09DC}';
pub const B_RH: char = '\u{09DD}';
pub const B_Y: char = '\u{09DF}';

// Punctuation
pub const B_DARI: char = '\u{0964}';
pub const B_DDARI: char = '\u{0965}';

// Digits
pub const B_0: char = '\u{09E6}';
pub const B_1: char = '\u{09E7}';
pub const B_2: char = '\u{09E8}';
pub const B_3: char = '\u{09E9}';
pub const B_4: char = '\u{09EA}';
pub const B_5: char = '\u{09EB}';
pub const B_6: char = '\u{09EC}';
pub const B_7: char = '\u{09ED}';
pub const B_8: char = '\u{09EE}';
pub const B_9: char = '\u{09EF}';

// Currency
pub const B_TAKA: char = '\u{09F3}';

// Zero Width Characters
pub const ZWJ: char = '\u{200D}';
pub const ZWNJ: char = '\u{200C}';

pub trait BengaliCharExt {
    fn is_bengali(&self) -> bool;
    fn is_vowel(&self) -> bool;
    fn is_kar(&self) -> bool;
    fn is_consonant(&self) -> bool;
    fn is_pure_consonant(&self) -> bool;
    fn is_bengali_digit(&self) -> bool;
}

impl BengaliCharExt for char {
    fn is_bengali(&self) -> bool {
        matches!(*self, '\u{0980}'..='\u{09FF}' | '\u{0964}' | '\u{0965}')
    }

    fn is_vowel(&self) -> bool {
        matches!(
            *self,
            B_A | B_AA
                | B_I
                | B_II
                | B_U
                | B_UU
                | B_RRI
                | B_E
                | B_OI
                | B_O
                | B_OU
                | B_AA_KAR
                | B_I_KAR
                | B_II_KAR
                | B_U_KAR
                | B_UU_KAR
                | B_RRI_KAR
                | B_E_KAR
                | B_OI_KAR
                | B_O_KAR
                | B_OU_KAR
        )
    }

    fn is_kar(&self) -> bool {
        matches!(
            *self,
            B_AA_KAR
                | B_I_KAR
                | B_II_KAR
                | B_U_KAR
                | B_UU_KAR
                | B_RRI_KAR
                | B_VOCALIC_RR
                | B_E_KAR
                | B_OI_KAR
                | B_O_KAR
                | B_OU_KAR
        )
    }

    fn is_consonant(&self) -> bool {
        matches!(
            *self,
            B_K..=B_NGA | B_C..=B_NYA | B_TT..=B_NN | B_T..=B_N |
            B_P..=B_M | B_Z..=B_H | B_RR | B_RH | B_Y | B_KHANDATTA
        )
    }

    fn is_pure_consonant(&self) -> bool {
        matches!(
            *self,
            B_K..=B_NGA | B_C..=B_NYA | B_TT..=B_NN | B_T..=B_N |
            B_P..=B_M | B_Z..=B_H | B_RR | B_RH | B_Y
        )
    }

    fn is_bengali_digit(&self) -> bool {
        matches!(*self, B_0..=B_9)
    }
}
