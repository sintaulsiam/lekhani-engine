//! Lekhani Core Typing Engine

pub mod bilingual;
pub mod chars;
pub mod conjuncts;
pub mod converter;
pub mod emojis;
pub mod fixed;
pub mod fs;
pub mod keycodes;
pub mod morpheme;
pub mod ngram;
pub mod personas;
pub mod phonetic;
pub mod session;
pub mod snippets;
pub mod tokenization;
pub mod trie;

pub use bilingual::*;

pub use chars::*;
pub use conjuncts::{ConjunctCatalog, ConjunctInfo};
pub use converter::{bijoy_to_unicode, unicode_to_bijoy};
pub use emojis::EmojiMap;
pub use fixed::{FixedLayoutParser, FixedMethod};
pub use fs::atomic_write_secure;
pub use keycodes::*;
pub use lekhani_parser::{
    default_avro_parser, has_explicit_avro_intent, CompiledLayout, LekhaniParser,
};
pub use morpheme::{
    BengaliMorphAnalyzer, MorphemeDecomposition, RupmulDecomposition, RupmulGraphSynthesizer,
    RupmulTier, SuffixCategory,
};
pub use ngram::UserStats;
pub use phonetic::{
    AutonomousLearner, PhoneticDatabase, PhoneticMethod, PhoneticSuggestion,
    PhoneticSuggestionConfig,
};
pub use session::{ActiveLayoutType, InputSession, ShiftOrigin};
pub use snippets::SnippetManager;
pub use trie::PrefixTrie;
