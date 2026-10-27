use hashbrown::HashMap;
use lekhani_core::chars::canonicalize;
use lekhani_core::phonetic::PhoneticSuggestion;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

fn is_acceptable_match(cand: &str, target: &str) -> bool {
    let cand_norm = canonicalize(cand);
    let target_norm = canonicalize(target);
    if cand_norm == target_norm {
        return true;
    }
    // 1. Dental ন vs Cerebral ণ
    let cand_n = cand_norm.replace('ণ', "ন");
    let target_n = target_norm.replace('ণ', "ন");
    if cand_n == target_n {
        return true;
    }
    // 2. Chalita verb terminal ব vs বো
    if (cand_norm.ends_with("বো") && target_norm.ends_with('ব') && cand_norm.strip_suffix("বো") == target_norm.strip_suffix('ব'))
        || (cand_norm.ends_with('ব') && target_norm.ends_with("বো") && cand_norm.strip_suffix('ব') == target_norm.strip_suffix("বো"))
    {
        return true;
    }
    // 3. Spoken / Sadhu O-kar variations (লো/ল, নো/ন)
    if (cand_norm.ends_with("লো") && target_norm.ends_with('ল') && cand_norm.strip_suffix("লো") == target_norm.strip_suffix('ল'))
        || (cand_norm.ends_with('ল') && target_norm.ends_with("লো") && cand_norm.strip_suffix('ল') == target_norm.strip_suffix("লো"))
        || (cand_norm.ends_with("নো") && target_norm.ends_with('ন') && cand_norm.strip_suffix("নো") == target_norm.strip_suffix('ন'))
        || (cand_norm.ends_with('ন') && target_norm.ends_with("নো") && cand_norm.strip_suffix('ন') == target_norm.strip_suffix("নো"))
    {
        return true;
    }
    // 4. Sanskrit Tatsama velar nasal conjuncts vs Anusvara orthographic variants (ঙ্ক/ংক, ঙ্গ/ংগ, ঙ্ঘ/ংঘ)
    let norm_velar = |s: &str| -> String {
        s.replace("ঙ্ক", "ংক").replace("ঙ্গ", "ংগ").replace("ঙ্ঘ", "ংঘ")
    };
    if norm_velar(&cand_norm) == norm_velar(&target_norm)
        || norm_velar(&cand_n) == norm_velar(&target_n)
    {
        return true;
    }
    false
}

fn main() {
    let dev_path = PathBuf::from("data/eval/bn.translit.sampled.dev.tsv");
    let mut sugg = PhoneticSuggestion::new();
    let dict_dir = PathBuf::from("data/dictionaries");
    sugg.load_from_dir_with_dict(&dict_dir, "dictionary.json")
        .expect("load dict");

    let layout_path = PathBuf::from("data/layouts/avrophonetic.json");
    if let Ok(content) = std::fs::read_to_string(&layout_path) {
        if let Ok(json) = serde_json::from_str(&content) {
            sugg.set_layout(&json);
        }
    }

    let file = File::open(&dev_path).expect("open dev.tsv");
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for l in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = l.split('\t').collect();
        if parts.len() >= 2 {
            pairs.push((parts[0].trim().to_string(), parts[1].trim().to_string()));
        }
    }

    let empty_memory: HashMap<String, String> = HashMap::new();

    let mut exact_top1 = 0;
    let mut acceptable_top1 = 0;
    let mut acceptable_top3 = 0;
    let mut real_misses = Vec::new();

    for (target_raw, input) in &pairs {
        let target = canonicalize(target_raw);
        let (cands, _) = sugg.suggest(input, true, true, &empty_memory);
        let top1 = cands.first().map(|s| canonicalize(s)).unwrap_or_default();

        if top1 == target {
            exact_top1 += 1;
            acceptable_top1 += 1;
            acceptable_top3 += 1;
        } else if is_acceptable_match(&top1, &target) {
            acceptable_top1 += 1;
            acceptable_top3 += 1;
        } else {
            let in_top3 = cands.iter().take(3).any(|c| is_acceptable_match(c, &target));
            if in_top3 {
                acceptable_top3 += 1;
            }
            real_misses.push((input.clone(), target.clone(), top1.clone(), in_top3, cands));
        }
    }

    let n = pairs.len() as f64;
    println!("Total Dev Samples: {}", pairs.len());
    println!("Strict Top-1 Accuracy:       {:.2}% ({}/{})", exact_top1 as f64 / n * 100.0, exact_top1, pairs.len());
    println!("Acceptable Top-1 Accuracy:   {:.2}% ({}/{})", acceptable_top1 as f64 / n * 100.0, acceptable_top1, pairs.len());
    println!("Acceptable Top-3 Accuracy:   {:.2}% ({}/{})", acceptable_top3 as f64 / n * 100.0, acceptable_top3, pairs.len());
    println!("Total Real Misses:           {}", real_misses.len());

    let mut miss_in_dict = 0usize;
    let mut miss_not_in_dict = 0usize;
    let mut miss_target_in_cands = 0usize;
    let mut miss_stem_in_dict = 0usize;

    let suffixes = ["এর", "র", "ে", "তে", "ও", "সহ", "গুলো", "গুলি", "সমূহের", "সমূহ", "দের", "টি", "টা", "কে"];

    for (_input, target, _top1, _in_top3, cands) in &real_misses {
        let in_dict = sugg.database.is_exact_dictionary_word(target);
        if in_dict {
            miss_in_dict += 1;
        } else {
            miss_not_in_dict += 1;
            let mut stem_found = false;
            for suf in &suffixes {
                if target.ends_with(suf) && target.len() > suf.len() {
                    let stem = &target[..target.len() - suf.len()];
                    if sugg.database.is_exact_dictionary_word(stem) {
                        stem_found = true;
                        break;
                    }
                }
            }
            if stem_found {
                miss_stem_in_dict += 1;
            }
        }
        if cands.iter().any(|c| is_acceptable_match(c, target)) {
            miss_target_in_cands += 1;
        }
    }

    println!("\n=== COMPREHENSIVE BREAKDOWN OF THE {} MISSES ===", real_misses.len());
    println!("1. Target IS in Dictionary (Ranking / Model Misses):   {} ({:.1}%)", miss_in_dict, miss_in_dict as f64 / real_misses.len() as f64 * 100.0);
    println!("   - Target was generated in candidates (Rank 2+):    {} ({:.1}%)", miss_target_in_cands, miss_target_in_cands as f64 / real_misses.len() as f64 * 100.0);
    println!("   - Target was NOT generated at all:                 {} ({:.1}%)", miss_in_dict.saturating_sub(miss_target_in_cands), miss_in_dict.saturating_sub(miss_target_in_cands) as f64 / real_misses.len() as f64 * 100.0);
    println!("2. Target is NOT in Dictionary (Lexicon Ceiling):     {} ({:.1}%)", miss_not_in_dict, miss_not_in_dict as f64 / real_misses.len() as f64 * 100.0);
    println!("   - Out-of-dict targets with stem in dictionary:     {} ({:.1}%)", miss_stem_in_dict, miss_stem_in_dict as f64 / real_misses.len() as f64 * 100.0);
    println!("   - Completely unknown vocabulary (names, noise):    {} ({:.1}%)", miss_not_in_dict - miss_stem_in_dict, (miss_not_in_dict - miss_stem_in_dict) as f64 / real_misses.len() as f64 * 100.0);
}
