//! Layout JSON Compiler for Phonetic Parser

use serde_json::Value;
use super::rule::{ConditionScope, MatchType, Pattern, PatternRule, RuleCondition};
use super::trie::PatternTrie;

#[derive(Debug, Clone)]
pub struct CompiledLayout {
    pub patterns: Vec<Pattern>,
    pub trie: PatternTrie,
}

impl CompiledLayout {
    pub fn from_json(val: &Value) -> Result<Self, &'static str> {
        let layout_obj = if let Some(l) = val.get("layout") {
            l
        } else {
            val
        };

        let patterns_val = layout_obj
            .get("patterns")
            .and_then(|p| p.as_array())
            .ok_or("Missing patterns array in layout JSON")?;

        let mut patterns = Vec::with_capacity(patterns_val.len());
        let mut trie = PatternTrie::new();

        for p_val in patterns_val {
            let find = p_val
                .get("find")
                .and_then(|f| f.as_str())
                .ok_or("Pattern missing 'find' field")?;
            let default_replace = p_val
                .get("replace")
                .and_then(|r| r.as_str())
                .ok_or("Pattern missing 'replace' field")?;

            let rules_val = p_val.get("rules").and_then(|r| r.as_array());
            let mut pattern_rules = Vec::new();

            if let Some(r_arr) = rules_val {
                for r_item in r_arr {
                    let replace = r_item
                        .get("replace")
                        .and_then(|rep| rep.as_str())
                        .ok_or("Rule missing 'replace' field")?;
                    let matches_val = r_item
                        .get("matches")
                        .and_then(|m| m.as_array())
                        .ok_or("Rule missing 'matches' field")?;

                    let mut conditions = Vec::with_capacity(matches_val.len());
                    for m_item in matches_val {
                        let type_str = m_item
                            .get("type")
                            .and_then(|t| t.as_str())
                            .ok_or("Match missing 'type' field")?;
                        let mut scope_str = m_item
                            .get("scope")
                            .and_then(|s| s.as_str())
                            .ok_or("Match missing 'scope' field")?;

                        let match_type = match type_str {
                            "prefix" => MatchType::Prefix,
                            "suffix" => MatchType::Suffix,
                            _ => return Err("Invalid match type"),
                        };

                        let is_negative = if scope_str.starts_with('!') {
                            scope_str = &scope_str[1..];
                            true
                        } else {
                            false
                        };

                        let scope = match scope_str {
                            "vowel" => ConditionScope::Vowel,
                            "consonant" => ConditionScope::Consonant,
                            "punctuation" => ConditionScope::Punctuation,
                            "number" => ConditionScope::Number,
                            "exact" => ConditionScope::Exact,
                            _ => return Err("Unknown condition scope"),
                        };

                        let exact_val = m_item
                            .get("value")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();

                        conditions.push(RuleCondition {
                            match_type,
                            scope,
                            is_negative,
                            exact_value: exact_val.into(),
                        });
                    }

                    pattern_rules.push(PatternRule {
                        conditions,
                        replace: replace.into(),
                    });
                }
            }

            let may_have_combining = default_replace.contains('\u{09BC}')
                || default_replace.contains('ঁ')
                || pattern_rules
                    .iter()
                    .any(|r| r.replace.contains('\u{09BC}') || r.replace.contains('ঁ'));

            let pat_idx = patterns.len() as u16;
            trie.insert(find.as_bytes(), pat_idx);

            patterns.push(Pattern {
                find: find.into(),
                default_replace: default_replace.into(),
                rules: pattern_rules,
                may_have_combining,
            });
        }

        Ok(Self { patterns, trie })
    }
}
