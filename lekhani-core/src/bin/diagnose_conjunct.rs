//! Diagnose Baseline Performance Stratified by Conjunct and Word Length on DEV
//!
//! Evaluates the uncontaminated baseline (dictionary.json only) across:
//! 1. Conjunct vs Non-Conjunct words
//! 2. Word length buckets (1-3, 4-6, 7+)
//! 3. Cross-stratification: Conjunct x Length

use hashbrown::HashMap;
use lekhani_core::chars::canonicalize;
use lekhani_core::personas::{transform_avro_strict, transform_lazy, transform_natural};
use lekhani_core::phonetic::PhoneticSuggestion;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

pub fn has_bengali_conjunct(s: &str) -> bool {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return false;
    }
    for i in 0..chars.len() - 1 {
        if chars[i] == '্' {
            let next_c = chars[i + 1];
            if ('\u{0995}'..='\u{09B9}').contains(&next_c)
                || matches!(next_c, '\u{09CE}' | '\u{09DC}' | '\u{09DD}' | '\u{09DF}')
            {
                return true;
            }
        }
    }
    false
}

fn setup_engine() -> PhoneticSuggestion {
    let mut sugg = PhoneticSuggestion::new();
    let layout_candidates = [
        PathBuf::from("data/layouts/avrophonetic.json"),
        PathBuf::from("../data/layouts/avrophonetic.json"),
    ];
    for p in &layout_candidates {
        if p.exists() {
            if let Ok(content) = std::fs::read_to_string(p) {
                if let Ok(json) = serde_json::from_str(&content) {
                    sugg.set_layout(&json);
                    break;
                }
            }
        }
    }
    let dict_candidates = [
        PathBuf::from("data/dictionaries"),
        PathBuf::from("../data/dictionaries"),
    ];
    for p in &dict_candidates {
        if p.exists() {
            let _ = sugg.database.load_from_dir_with_dict(p, "dictionary.json");
            break;
        }
    }
    sugg
}

fn load_pairs(path_str: &str) -> Vec<(String, String)> {
    let file = File::open(path_str).expect("Failed to open TSV");
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for l in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = l.split('\t').collect();
        if parts.len() >= 2 {
            let bn = canonicalize(parts[0].trim());
            let lat = parts[1].trim().to_string();
            if !bn.is_empty() && !lat.is_empty() {
                pairs.push((bn, lat));
            }
        }
    }
    pairs
}

#[derive(Default)]
struct AccCounter {
    total: usize,
    top1: usize,
    top3: usize,
    unreachable: usize,
}

impl AccCounter {
    fn record(&mut self, is_top1: bool, is_top3: bool, is_reachable: bool) {
        self.total += 1;
        if is_top1 { self.top1 += 1; }
        if is_top3 { self.top3 += 1; }
        if !is_reachable { self.unreachable += 1; }
    }
    fn top1_pct(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.top1 as f64 / self.total as f64 * 100.0 }
    }
    fn top3_pct(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.top3 as f64 / self.total as f64 * 100.0 }
    }
    fn unreach_pct(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.unreachable as f64 / self.total as f64 * 100.0 }
    }
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════════");
    println!("   🔬 BASELINE ENGINE STRATIFICATION: CONJUNCT VS NON-CONJUNCT & LENGTH ON DEV");
    println!("══════════════════════════════════════════════════════════════════════════════════\n");

    let mut sugg = setup_engine();
    let dev_pairs = load_pairs("data/eval/bn.translit.sampled.dev.tsv");
    println!("Loaded {} pairs from DEV split.\n", dev_pairs.len());

    let empty_mem = HashMap::new();

    let personas = [
        ("natural", "Natural (Human Romanization)"),
        ("lazy", "Lazy Persona"),
        ("avro_strict", "Avro-Strict Persona"),
    ];

    for (p_id, p_name) in personas {
        let mut overall = AccCounter::default();
        let mut conj_acc = AccCounter::default();
        let mut non_conj_acc = AccCounter::default();

        let mut len_1_3 = AccCounter::default();
        let mut len_4_6 = AccCounter::default();
        let mut len_7_plus = AccCounter::default();

        let mut conj_len_1_3 = AccCounter::default();
        let mut conj_len_4_6 = AccCounter::default();
        let mut conj_len_7_plus = AccCounter::default();

        let mut non_conj_len_1_3 = AccCounter::default();
        let mut non_conj_len_4_6 = AccCounter::default();
        let mut non_conj_len_7_plus = AccCounter::default();

        for (target, raw_input) in &dev_pairs {
            let input = match p_id {
                "natural" => transform_natural(raw_input),
                "lazy" => transform_lazy(raw_input),
                "avro_strict" => transform_avro_strict(target),
                _ => raw_input.clone(),
            };

            let (cands, _) = sugg.suggest(&input, true, true, &empty_mem);
            let is_top1 = cands.first().map(|c| c == target).unwrap_or(false);
            let is_top3 = cands.iter().take(3).any(|c| c == target);
            let is_reachable = cands.iter().any(|c| c == target);

            overall.record(is_top1, is_top3, is_reachable);

            let is_conj = has_bengali_conjunct(target);
            if is_conj {
                conj_acc.record(is_top1, is_top3, is_reachable);
            } else {
                non_conj_acc.record(is_top1, is_top3, is_reachable);
            }

            let bn_len = target.chars().count();
            if bn_len <= 3 {
                len_1_3.record(is_top1, is_top3, is_reachable);
                if is_conj { conj_len_1_3.record(is_top1, is_top3, is_reachable); }
                else { non_conj_len_1_3.record(is_top1, is_top3, is_reachable); }
            } else if bn_len <= 6 {
                len_4_6.record(is_top1, is_top3, is_reachable);
                if is_conj { conj_len_4_6.record(is_top1, is_top3, is_reachable); }
                else { non_conj_len_4_6.record(is_top1, is_top3, is_reachable); }
            } else {
                len_7_plus.record(is_top1, is_top3, is_reachable);
                if is_conj { conj_len_7_plus.record(is_top1, is_top3, is_reachable); }
                else { non_conj_len_7_plus.record(is_top1, is_top3, is_reachable); }
            }
        }

        println!("### Persona: {}", p_name);
        println!("| Stratum | Rows | Top-1 (%) | Top-3 (%) | Unreachable (%) |");
        println!("|:---|:---:|:---:|:---:|:---:|");
        println!("| **Overall DEV** | {} | **{:.2}%** | {:.2}% | {:.2}% |", overall.total, overall.top1_pct(), overall.top3_pct(), overall.unreach_pct());
        println!("| ├─ **With Conjunct** | {} | **{:.2}%** | {:.2}% | {:.2}% |", conj_acc.total, conj_acc.top1_pct(), conj_acc.top3_pct(), conj_acc.unreach_pct());
        println!("| └─ **Without Conjunct** | {} | **{:.2}%** | {:.2}% | {:.2}% |", non_conj_acc.total, non_conj_acc.top1_pct(), non_conj_acc.top3_pct(), non_conj_acc.unreach_pct());
        println!("| Length: 1-3 chars | {} | {:.2}% | {:.2}% | {:.2}% |", len_1_3.total, len_1_3.top1_pct(), len_1_3.top3_pct(), len_1_3.unreach_pct());
        println!("| Length: 4-6 chars | {} | {:.2}% | {:.2}% | {:.2}% |", len_4_6.total, len_4_6.top1_pct(), len_4_6.top3_pct(), len_4_6.unreach_pct());
        println!("| Length: 7+ chars | {} | {:.2}% | {:.2}% | {:.2}% |", len_7_plus.total, len_7_plus.top1_pct(), len_7_plus.top3_pct(), len_7_plus.unreach_pct());
        println!("| Conjunct x Length: 1-3 | {} | {:.2}% | {:.2}% | {:.2}% |", conj_len_1_3.total, conj_len_1_3.top1_pct(), conj_len_1_3.top3_pct(), conj_len_1_3.unreach_pct());
        println!("| Conjunct x Length: 4-6 | {} | {:.2}% | {:.2}% | {:.2}% |", conj_len_4_6.total, conj_len_4_6.top1_pct(), conj_len_4_6.top3_pct(), conj_len_4_6.unreach_pct());
        println!("| Conjunct x Length: 7+ | {} | {:.2}% | {:.2}% | {:.2}% |", conj_len_7_plus.total, conj_len_7_plus.top1_pct(), conj_len_7_plus.top3_pct(), conj_len_7_plus.unreach_pct());
        println!("| Non-Conj x Length: 1-3 | {} | {:.2}% | {:.2}% | {:.2}% |", non_conj_len_1_3.total, non_conj_len_1_3.top1_pct(), non_conj_len_1_3.top3_pct(), non_conj_len_1_3.unreach_pct());
        println!("| Non-Conj x Length: 4-6 | {} | {:.2}% | {:.2}% | {:.2}% |", non_conj_len_4_6.total, non_conj_len_4_6.top1_pct(), non_conj_len_4_6.top3_pct(), non_conj_len_4_6.unreach_pct());
        println!("| Non-Conj x Length: 7+ | {} | {:.2}% | {:.2}% | {:.2}% |\n", non_conj_len_7_plus.total, non_conj_len_7_plus.top1_pct(), non_conj_len_7_plus.top3_pct(), non_conj_len_7_plus.unreach_pct());
    }
}
