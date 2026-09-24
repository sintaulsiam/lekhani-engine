//! Phonetic Rule and Condition Data Structures

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    Prefix,
    Suffix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionScope {
    Vowel,
    Consonant,
    Punctuation,
    Number,
    Exact,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleCondition {
    pub match_type: MatchType,
    pub scope: ConditionScope,
    pub is_negative: bool,
    pub exact_value: Box<str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternRule {
    pub conditions: Vec<RuleCondition>,
    pub replace: Box<str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    pub find: Box<str>,
    pub default_replace: Box<str>,
    pub rules: Vec<PatternRule>,
}
