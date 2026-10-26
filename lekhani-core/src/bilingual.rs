//! Bilingual Clitic, Digit-Unit, and Technical Acronym Engine
//!
//! Provides cross-platform handling for:
//! - Digit prefixes with units/currency/ordinals (e.g. "10ta" -> "১০টা", "500tk" -> "৫০০ টাকা", "1st" -> "১ম")
//! - Hyphenated bilingual enclitics (e.g. "Facebook-e" -> "Facebook-এ", "post-er" -> "post-এর")
//! - Unhyphenated bilingual clitics (e.g. "WhatsAppe" -> "WhatsApp-এ", "filegulo" -> "file-গুলো")
//! - Technical acronym detection ("API", "RAM", "JSON", "UI", "CCA")

use std::sync::atomic::{AtomicBool, Ordering};

static BILINGUAL_CLITIC_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable bilingual clitic resolution globally (default: false)
pub fn set_bilingual_clitic_enabled(enabled: bool) {
    BILINGUAL_CLITIC_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns true if bilingual clitic resolution is currently enabled
pub fn is_bilingual_clitic_enabled() -> bool {
    BILINGUAL_CLITIC_ENABLED.load(Ordering::Relaxed)
}

pub const COMMON_TECH_ACRONYMS: &[&str] = &[
    "RAM", "ROM", "CPU", "GPU", "SSD", "HDD", "USB", "LAN", "LED", "LCD", "URL", "URI", "API",
    "SDK", "SQL", "CSS", "HTML", "JSON", "XML", "PDF", "APK", "JPG", "PNG", "MP3", "MP4", "OTP",
    "NFC", "GPS", "VPN", "DNS", "SSH", "FTP", "MAC", "IP", "OS", "PC", "TV", "AI", "UI", "UX",
    "ID", "SMS", "CCA", "SIM", "PIN", "CVV", "ATM", "POS", "QR", "FAQ", "DDR", "PCI", "SATA",
    "NVME", "WIFI", "RFC", "HTTP", "HTTPS", "REST", "CLI", "GUI", "IDE", "AAB", "JNI", "FFI",
    "ABI", "NDK", "SOC", "DSP", "ISP", "CER", "WER", "LLM", "AGI", "ASR", "TTS", "NLP",
];

pub fn is_common_tech_acronym(token: &str) -> bool {
    COMMON_TECH_ACRONYMS.contains(&token)
}

pub const STATIC_SHORT_ENGLISH_WORDS: &[&str] = &[
    "a", "i", "ok", "hi", "to", "no", "go", "in", "on", "at", "is", "am", "he", "me", "we", "fb",
    "id", "ai", "pc", "tv", "so", "my", "by", "up", "it", "as", "or", "if", "do", "an", "us", "ex",
    "vs", "app", "web", "api", "otp", "dev", "doc", "ui", "ux", "os", "ip",
];

pub fn is_static_short_english_word(token: &str) -> bool {
    STATIC_SHORT_ENGLISH_WORDS.contains(&token.to_lowercase().as_str())
}

/// Converts ASCII digits ('0'..='9') to Bengali digits ('০'..='৯')
pub fn ascii_digits_to_bengali(digits: &str) -> String {
    digits
        .chars()
        .map(|c| match c {
            '0' => '০',
            '1' => '১',
            '2' => '২',
            '3' => '৩',
            '4' => '৪',
            '5' => '৫',
            '6' => '৬',
            '7' => '৭',
            '8' => '৮',
            '9' => '৯',
            _ => c,
        })
        .collect()
}

/// Converts Bengali digits ('০'..='৯') to ASCII digits ('0'..='9')
pub fn bengali_digits_to_ascii(digits: &str) -> String {
    digits
        .chars()
        .map(|c| match c {
            '০' => '0',
            '১' => '1',
            '২' => '2',
            '৩' => '3',
            '৪' => '4',
            '৫' => '5',
            '৬' => '6',
            '৭' => '7',
            '৮' => '8',
            '৯' => '9',
            _ => c,
        })
        .collect()
}

/// Resolves digit prefixes with units, quantifiers, currency, and ordinals
/// e.g. "10ta" -> "১০টা", "500tk" -> "৫০০ টাকা", "1st" -> "১ম"
pub fn resolve_digit_unit_candidates(input: &str) -> Option<(String, Vec<String>)> {
    let digit_count = input.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_count == 0 || digit_count == input.chars().count() {
        return None;
    }

    let (digits, suffix) = input.split_at(digit_count);
    let lower_suffix = suffix.to_lowercase();
    let b_digits = ascii_digits_to_bengali(digits);

    let (primary, candidates) = match lower_suffix.as_str() {
        "ta" => {
            let p = format!("{}টা", b_digits);
            let cands = vec![
                p.clone(),
                input.to_string(),
                format!("{}তা", b_digits),
                digits.to_string(),
            ];
            (p, cands)
        }
        "ti" => {
            let p = format!("{}টি", b_digits);
            let cands = vec![
                p.clone(),
                input.to_string(),
                format!("{}তি", b_digits),
                digits.to_string(),
            ];
            (p, cands)
        }
        "tk" => {
            let p = format!("{} টাকা", b_digits);
            let cands = vec![
                p.clone(),
                format!("৳{}", b_digits),
                input.to_string(),
                format!("{}tk", b_digits),
                b_digits.clone(),
            ];
            (p, cands)
        }
        "taka" => {
            let p = format!("{} টাকা", b_digits);
            let cands = vec![
                p.clone(),
                format!("৳{}", b_digits),
                input.to_string(),
                b_digits.clone(),
            ];
            (p, cands)
        }
        "jon" => {
            let p = format!("{}জন", b_digits);
            let cands = vec![p.clone(), input.to_string(), b_digits.clone()];
            (p, cands)
        }
        "bar" => {
            let p = format!("{}বার", b_digits);
            let cands = vec![p.clone(), input.to_string(), b_digits.clone()];
            (p, cands)
        }
        "din" => {
            let p = format!("{}দিন", b_digits);
            let cands = vec![p.clone(), input.to_string(), b_digits.clone()];
            (p, cands)
        }
        "mas" => {
            let p = format!("{}মাস", b_digits);
            let cands = vec![p.clone(), input.to_string(), b_digits.clone()];
            (p, cands)
        }
        "bochor" | "bosor" => {
            let p = format!("{}বছর", b_digits);
            let cands = vec![p.clone(), input.to_string(), b_digits.clone()];
            (p, cands)
        }
        "kg" => {
            let p = format!("{} কেজি", b_digits);
            let cands = vec![
                p.clone(),
                input.to_string(),
                format!("{}kg", b_digits),
                b_digits.clone(),
            ];
            (p, cands)
        }
        "gm" | "gram" => {
            let p = format!("{} গ্রাম", b_digits);
            let cands = vec![
                p.clone(),
                input.to_string(),
                format!("{}gm", b_digits),
                b_digits.clone(),
            ];
            (p, cands)
        }
        "km" => {
            let p = format!("{} কিমি", b_digits);
            let cands = vec![
                p.clone(),
                input.to_string(),
                format!("{}km", b_digits),
                b_digits.clone(),
            ];
            (p, cands)
        }
        "mb" => {
            let p = format!("{} এমবি", b_digits);
            let cands = vec![p.clone(), format!("{}MB", digits), input.to_string()];
            (p, cands)
        }
        "gb" => {
            let p = format!("{} জিবি", b_digits);
            let cands = vec![p.clone(), format!("{}GB", digits), input.to_string()];
            (p, cands)
        }
        // English ordinals to Bengali ordinals
        "st" => {
            let p = if digits == "1" {
                "১ম".to_string()
            } else {
                format!("{}তম", b_digits)
            };
            let cands = vec![p.clone(), input.to_string(), b_digits.clone()];
            (p, cands)
        }
        "nd" => {
            let p = if digits == "2" {
                "২য়".to_string()
            } else {
                format!("{}তম", b_digits)
            };
            let cands = vec![p.clone(), input.to_string(), b_digits.clone()];
            (p, cands)
        }
        "rd" => {
            let p = if digits == "3" {
                "৩য়".to_string()
            } else {
                format!("{}তম", b_digits)
            };
            let cands = vec![p.clone(), input.to_string(), b_digits.clone()];
            (p, cands)
        }
        "th" => {
            let p = match digits {
                "4" => "৪র্থ".to_string(),
                "5" => "৫ম".to_string(),
                "6" => "৬ষ্ঠ".to_string(),
                "7" => "৭ম".to_string(),
                "8" => "৮ম".to_string(),
                "9" => "৯ম".to_string(),
                "10" => "১০ম".to_string(),
                _ => format!("{}তম", b_digits),
            };
            let cands = vec![p.clone(), input.to_string(), b_digits.clone()];
            (p, cands)
        }
        _ => return None,
    };

    Some((primary, candidates))
}

pub const CLITIC_SUFFIXES: &[(&str, &str)] = &[
    ("gulor", "-গুলোর"),
    ("gulote", "-গুলোতে"),
    ("gulo", "-গুলো"),
    ("gula", "-গুলা"),
    ("guli", "-গুলি"),
    ("derke", "-দেরকে"),
    ("der", "-দের"),
    ("marse", " মারছে"),
    ("marlam", " মারলাম"),
    ("kora", " করা"),
    ("kore", " করে"),
    ("koro", " করো"),
    ("bhittik", "-ভিত্তিক"),
    ("shongkranto", "-সংক্রান্ত"),
    ("er", "-এর"),
    ("te", "-তে"),
    ("ta", "-টা"),
    ("ti", "-টি"),
    ("ke", "-কে"),
    ("ra", "-রা"),
    ("re", "-রে"),
    ("e", "-এ"),
    ("o", "-ও"),
    ("i", "-ই"),
];

/// Resolves bilingual hyphenated enclitics attached to Latin words (e.g. "Facebook-e" -> "Facebook-এ")
pub fn resolve_bilingual_hyphen_clitic(input: &str) -> Option<(String, Vec<String>)> {
    if !input.contains('-') || input.starts_with('-') || input.ends_with('-') {
        return None;
    }

    let dash_idx = input.rfind('-')?;
    let (base, suffix_with_dash) = input.split_at(dash_idx);
    let suffix = &suffix_with_dash[1..];

    if base.is_empty()
        || suffix.is_empty()
        || !base
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return None;
    }

    if base.chars().any(|c| matches!(c, '^' | '`' | ':' | '$')) {
        return None;
    }

    let lower_suffix = suffix.to_lowercase();
    let b_clitic = match lower_suffix.as_str() {
        "e" => "-এ",
        "te" => "-তে",
        "r" => "-র",
        "er" => "-এর",
        "ta" => "-টা",
        "ti" => "-টি",
        "ke" => "-কে",
        "gulo" => "-গুলো",
        "gulor" => "-গুলোর",
        "der" => "-দের",
        "o" => "-ও",
        "i" => "-ই",
        "bhittik" => "-ভিত্তিক",
        "shongkranto" => "-সংক্রান্ত",
        _ => return None,
    };

    let primary = format!("{}{}", base, b_clitic);
    let mut candidates = Vec::with_capacity(4);
    candidates.push(primary.clone());
    candidates.push(input.to_string());
    if let Some(stripped) = b_clitic.strip_prefix('-') {
        candidates.push(format!("{}{}", base, stripped));
    }

    Some((primary, candidates))
}

/// Resolves unhyphenated bilingual enclitics attached to Latin words (e.g. "WhatsAppe" -> "WhatsApp-এ", "filegulo" -> "file-গুলো")
pub fn resolve_unhyphenated_bilingual_clitic(input: &str) -> Option<(String, Vec<String>)> {
    if input.is_empty() || !input.is_ascii() || input.contains('-') || input.len() < 3 {
        return None;
    }

    let lower = input.to_lowercase();

    for &(suf, b_clitic) in CLITIC_SUFFIXES {
        if lower.ends_with(suf) && input.len() > suf.len() {
            let base = &input[..input.len() - suf.len()];
            if base.is_empty() || !base.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                continue;
            }

            if base.chars().any(|c| matches!(c, '^' | '`' | ':' | '$')) {
                continue;
            }

            // Capitalized or multi-case bases are strongly indicative of a Latin token (e.g. "Meetinger", "WhatsAppe", "iPhone-er")
            let has_valid_latin_title =
                base.len() >= 3 && base.chars().next().is_some_and(|c| c.is_ascii_uppercase());
            let is_all_caps = base.len() >= 2 && base.chars().all(|c| c.is_ascii_uppercase());
            if has_valid_latin_title || is_all_caps || is_static_short_english_word(base) {
                let primary = format!("{}{}", base, b_clitic);
                let mut candidates = Vec::with_capacity(4);
                candidates.push(primary.clone());
                candidates.push(input.to_string());
                if let Some(stripped) = b_clitic.strip_prefix('-') {
                    candidates.push(format!("{}{}", base, stripped));
                }
                return Some((primary, candidates));
            }
        }
    }

    None
}

/// Contextual Technical Collocations where the English form is preferred (e.g. "file download", "api call")
pub const TECH_COLLOCATIONS_ENGLISH: &[(&str, &str)] = &[
    ("link", "share"),
    ("file", "download"),
    ("api", "call"),
    ("server", "restart"),
    ("git", "push"),
    ("git", "pull"),
    ("git", "commit"),
    ("git", "merge"),
    ("code", "review"),
    ("pull", "request"),
    ("bug", "fix"),
    ("task", "complete"),
];

/// Contextual Technical Collocations where naturalized Bengali loanword is preferred (e.g. "photo edit" -> "এডিট")
pub const TECH_COLLOCATIONS_BENGALI: &[(&str, &str)] = &[
    ("photo", "edit"),
    ("group", "chat"),
    ("screen", "share"),
    ("status", "update"),
    ("video", "call"),
    ("audio", "message"),
];

#[inline]
pub fn is_english_tech_collocation(prev: &str, curr: &str) -> bool {
    let p = prev.to_ascii_lowercase();
    let c = curr.to_ascii_lowercase();
    TECH_COLLOCATIONS_ENGLISH.iter().any(|&(w1, w2)| w1 == p && w2 == c)
}

#[inline]
pub fn is_bengali_tech_collocation(prev: &str, curr: &str) -> bool {
    let p = prev.to_ascii_lowercase();
    let c = curr.to_ascii_lowercase();
    TECH_COLLOCATIONS_BENGALI.iter().any(|&(w1, w2)| w1 == p && w2 == c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digit_unit_candidates() {
        let (p, cands) = resolve_digit_unit_candidates("10ta").unwrap();
        assert_eq!(p, "১০টা");
        assert!(cands.contains(&"10ta".to_string()));

        let (p, _) = resolve_digit_unit_candidates("500tk").unwrap();
        assert_eq!(p, "৫০০ টাকা");

        let (p, _) = resolve_digit_unit_candidates("1st").unwrap();
        assert_eq!(p, "১ম");

        let (p, _) = resolve_digit_unit_candidates("2nd").unwrap();
        assert_eq!(p, "২য়");

        let (p, _) = resolve_digit_unit_candidates("4th").unwrap();
        assert_eq!(p, "৪র্থ");
    }

    #[test]
    fn test_bilingual_hyphen_clitic() {
        let (p, _) = resolve_bilingual_hyphen_clitic("Facebook-e").unwrap();
        assert_eq!(p, "Facebook-এ");

        let (p, _) = resolve_bilingual_hyphen_clitic("post-er").unwrap();
        assert_eq!(p, "post-এর");
    }

    #[test]
    fn test_bilingual_unhyphenated_clitic() {
        let (p, _) = resolve_unhyphenated_bilingual_clitic("WhatsAppe").unwrap();
        assert_eq!(p, "WhatsApp-এ");

        let (p, _) = resolve_unhyphenated_bilingual_clitic("appgulo").unwrap();
        assert_eq!(p, "app-গুলো");
    }
}
