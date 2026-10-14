//! WebAssembly (WASM) Bindings for Lekhani Parser
//!
//! Enables zero-overhead in-browser and Node.js Bengali phonetic transliteration.

use crate::default_avro_parser;
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper around the compiled Avro phonetic parser.
#[wasm_bindgen]
pub struct WasmAvroParser {
    parser: crate::LekhaniParser,
}

#[wasm_bindgen]
impl WasmAvroParser {
    /// Create a new Avro phonetic parser instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            parser: default_avro_parser(),
        }
    }

    /// Transliterate Latin phonetic text into Unicode Bengali.
    #[wasm_bindgen]
    pub fn convert(&self, input: &str) -> String {
        self.parser.convert(input)
    }
}

impl Default for WasmAvroParser {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot convenience function to transliterate phonetic input into Bengali.
#[wasm_bindgen]
pub fn parse_avro(input: &str) -> String {
    default_avro_parser().convert(input)
}

/// Check if the phonetic input contains explicit, unambiguous Avro grammatical operators or mid-word casing.
#[wasm_bindgen]
pub fn has_explicit_avro_intent(input: &str) -> bool {
    crate::has_explicit_avro_intent(input)
}
