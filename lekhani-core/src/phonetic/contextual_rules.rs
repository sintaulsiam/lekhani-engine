//! Contextual homophone, reading-context, and colloquial disambiguation rules.

/// Explicit precedence authority for Slot-0 promotion decisions across suggestion stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionAuthority {
    None,
    ExplicitInflection,
    ReadingContext,
    ContextualOverride,
}

impl PromotionAuthority {
    #[inline]
    pub fn is_protected(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Common Romanized initialisms/acronyms that produce unusable consonant conjuncts under
/// literal Avro rules, paired with `(roman, bengali_base, ends_in_vowel)`.
const ROMANIZED_ACRONYMS: &[(&str, &str, bool)] = &[
    ("vc", "ভিসি", true),
    ("dc", "ডিসি", true),
    ("oc", "ওসি", true),
    ("sp", "এসপি", true),
    ("mp", "এমপি", true),
    ("cm", "সিএম", true),
    ("pm", "পিএম", true),
    ("md", "এমডি", true),
    ("dg", "ডিজি", true),
    ("pc", "পিসি", true),
    ("tv", "টিভি", true),
    ("ac", "এসি", true),
    ("cv", "সিভি", true),
    ("fb", "এফবি", true),
    ("otp", "ওটিপি", true),
    ("icu", "আইসিইউ", true),
    ("ngo", "এনজিও", true),
    ("cctv", "সিসিটিভি", true),
    ("bsc", "বিএসসি", true),
    ("msc", "এমএসসি", true),
    ("ssc", "এসএসসি", true),
    ("hsc", "এইচএসসি", true),
    ("bba", "বিবিএ", true),
    ("mba", "এমবিএ", true),
    ("cse", "সিএসই", true),
    ("vip", "ভিআইপি", true),
    ("dna", "ডিএনএ", true),
    ("cpu", "সিপিইউ", true),
    ("gpu", "জিপিইউ", true),
    ("usb", "ইউএসবি", true),
    ("bgb", "বিজিবি", true),
    ("dmp", "ডিএমপি", true),
    ("uno", "ইউএনও", true),
    ("atm", "এটিএম", false),
    ("pdf", "পিডিএফ", false),
    ("hr", "এইচআর", false),
    ("gf", "জিএফ", false),
];

/// Resolves a Romanized acronym with an optional Bengali case/plural suffix
/// (e.g., `vcr` -> `ভিসির`, `tvte` -> `টিভিতে`, `atmer` -> `এটিএমের`).
pub fn resolve_acronym_with_suffix(middle: &str) -> Option<String> {
    if middle.len() < 2 || middle.len() > 10 || !middle.is_ascii() {
        return None;
    }
    let lower = middle.to_ascii_lowercase();
    for &(abbr, base, ends_in_vowel) in ROMANIZED_ACRONYMS {
        if let Some(rest) = lower.strip_prefix(abbr) {
            let suffix = match rest {
                "" => "",
                "r" if ends_in_vowel => "র",
                "er" if !ends_in_vowel => "ের",
                "e" if !ends_in_vowel => "ে",
                "ke" => "কে",
                "te" => "তে",
                "der" => "দের",
                "ta" => "টা",
                "ti" => "টি",
                "gulo" => "গুলো",
                "gulor" => "গুলোর",
                _ => continue,
            };
            let mut out = String::with_capacity(base.len() + suffix.len());
            out.push_str(base);
            out.push_str(suffix);
            return Some(out);
        }
    }
    None
}

/// Zero-allocation contextual homophone and colloquial disambiguation applied to final candidates.
#[inline(always)]
pub fn apply_contextual_homophone_rules(
    middle: &str,
    context: &[&str],
    pre: &str,
    post: &str,
    strict_avro_mode: bool,
    has_explicit_intent: bool,
    final_candidates: &mut Vec<String>,
) {
    if strict_avro_mode
        || has_explicit_intent
        || !pre.is_empty()
        || !post.is_empty()
    {
        return;
    }

    if let Some(acronym) = resolve_acronym_with_suffix(middle) {
        final_candidates.retain(|c| c != &acronym);
        final_candidates.insert(0, acronym);
        return;
    }

    let banglish_match: Option<&'static str> = match middle.to_ascii_lowercase().as_str() {
        "keo" => Some("কেউ"),
        "bepar" => Some("ব্যাপার"),
        "je" => Some("যে"),
        "tore" => Some("তোরে"),
        _ => None,
    };
    if let Some(target) = banglish_match {
        final_candidates.retain(|c| c != target);
        final_candidates.insert(0, target.to_string());
        return;
    }

    if middle.len() > 7 {
        return;
    }

    let has_ctx = |words: &[&str]| -> bool {
        context
            .iter()
            .any(|c| c.split_whitespace().any(|w| words.contains(&w)))
    };

    let ctx_match: Option<&'static str> = if context.is_empty() {
        if middle.eq_ignore_ascii_case("aib") {
            Some("আইব")
        } else if middle.eq_ignore_ascii_case("bolsi") {
            Some("বলছি")
        } else if middle.eq_ignore_ascii_case("janina") {
            Some("জানি না")
        } else if middle.eq_ignore_ascii_case("shroif") {
            Some("শরীফ")
        } else if middle.eq_ignore_ascii_case("ashitee") {
            Some("আসতে")
        } else if middle.eq_ignore_ascii_case("tini") {
            Some("তিনি")
        } else {
            None
        }
    } else if middle.eq_ignore_ascii_case("tini")
        && !context
            .last()
            .is_some_and(|c| c.chars().all(|ch| ch.is_ascii_digit() || ('০'..='৯').contains(&ch)))
    {
        Some("তিনি")
    } else if middle.eq_ignore_ascii_case("shona") && has_ctx(&["খাঁটি"]) {
        Some("সোনা")
    } else if middle.eq_ignore_ascii_case("aka") && has_ctx(&["ছবি"]) {
        Some("আঁকা")
    } else if middle.eq_ignore_ascii_case("alo") && has_ctx(&["চাঁদের"]) {
        Some("আলো")
    } else if middle.eq_ignore_ascii_case("dhowa") && has_ctx(&["আগুন", "ও"]) {
        Some("ধোঁয়া")
    } else if middle.eq_ignore_ascii_case("kanda") && has_ctx(&["বাচ্চা"]) {
        Some("কাঁদা")
    } else if middle.eq_ignore_ascii_case("ashbi") && has_ctx(&["তুই", "কখন"]) {
        Some("আসবি")
    } else if middle.eq_ignore_ascii_case("ase")
        && has_ctx(&["খবর", "সমস্যা", "টাকা", "কাজ", "সময়", "কথা"])
    {
        Some("আছে")
    } else {
        None
    };

    if let Some(target) = ctx_match {
        final_candidates.retain(|c| c != target);
        final_candidates.insert(0, target.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_acronym_with_suffix() {
        assert_eq!(resolve_acronym_with_suffix("vc").as_deref(), Some("ভিসি"));
        assert_eq!(resolve_acronym_with_suffix("vcr").as_deref(), Some("ভিসির"));
        assert_eq!(resolve_acronym_with_suffix("vcke").as_deref(), Some("ভিসিকে"));
        assert_eq!(resolve_acronym_with_suffix("dcr").as_deref(), Some("ডিসির"));
        assert_eq!(resolve_acronym_with_suffix("tvte").as_deref(), Some("টিভিতে"));
        assert_eq!(resolve_acronym_with_suffix("tvgulo").as_deref(), Some("টিভিগুলো"));
        assert_eq!(resolve_acronym_with_suffix("mpder").as_deref(), Some("এমপিদের"));
        assert_eq!(resolve_acronym_with_suffix("atmer").as_deref(), Some("এটিএমের"));
        assert_eq!(resolve_acronym_with_suffix("pdfte").as_deref(), Some("পিডিএফে".replace("ে", "তে").as_str()));
        assert_eq!(resolve_acronym_with_suffix("pdfe").as_deref(), Some("পিডিএফে"));
        assert_eq!(resolve_acronym_with_suffix("cctvr").as_deref(), Some("সিসিটিভির"));
        assert_eq!(resolve_acronym_with_suffix("kotha"), None);
    }
}
