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
        || middle.len() > 7
    {
        return;
    }

    let has_ctx = |words: &[&str]| -> bool {
        context
            .iter()
            .any(|c| c.split_whitespace().any(|w| words.contains(&w)))
    };

    let ctx_match: Option<&'static str> = if middle.eq_ignore_ascii_case("vc") {
        Some("ভিসি")
    } else if middle.eq_ignore_ascii_case("vcr") {
        Some("ভিসির")
    } else if context.is_empty() {
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
    } else {
        None
    };

    if let Some(target) = ctx_match {
        final_candidates.retain(|c| c != target);
        final_candidates.insert(0, target.to_string());
    }
}
