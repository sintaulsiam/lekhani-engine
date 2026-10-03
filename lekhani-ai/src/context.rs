use crate::embeddings::WordEmbeddings;
use crate::lm::LanguageModel;

const OVERLAY_CAPACITY: usize = 4096;
const OVERLAY_MASK: usize = OVERLAY_CAPACITY - 1;

/// Fast, deterministic 64-bit FNV-1a hash for string tokens
#[inline]
pub fn hash_word(word: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in word.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // Reserve 0 as empty slot indicator
    if hash == 0 {
        1
    } else {
        hash
    }
}

/// Zero-allocation, fixed-size hot-vocab overlay table for personalized word boosts
#[derive(Clone)]
pub struct PersonalScoreOverlay {
    /// Fixed-size table of (word_hash, boost_score)
    table: [(u64, f32); OVERLAY_CAPACITY],
    count: usize,
}

impl std::fmt::Debug for PersonalScoreOverlay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersonalScoreOverlay")
            .field("count", &self.count)
            .finish()
    }
}

impl Default for PersonalScoreOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl PersonalScoreOverlay {
    pub fn new() -> Self {
        Self {
            table: [(0, 0.0); OVERLAY_CAPACITY],
            count: 0,
        }
    }

    /// Reset all personalized boost entries
    pub fn clear(&mut self) {
        self.table = [(0, 0.0); OVERLAY_CAPACITY];
        self.count = 0;
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Lookup personalized boost score for a word. Returns 0.0 if not present.
    #[inline]
    pub fn boost_for(&self, word: &str) -> f32 {
        self.boost_for_hash(hash_word(word))
    }

    /// Fast O(1) lookup by pre-computed 64-bit hash
    #[inline]
    pub fn boost_for_hash(&self, word_hash: u64) -> f32 {
        let mut idx = (word_hash as usize) & OVERLAY_MASK;
        for _ in 0..16 {
            let (h, score) = self.table[idx];
            if h == word_hash {
                return score;
            }
            if h == 0 {
                return 0.0;
            }
            idx = (idx + 1) & OVERLAY_MASK;
        }
        0.0
    }

    /// Set an explicit boost score for a word (e.g. from decayed UserStats)
    pub fn set_score(&mut self, word: &str, score: f32) {
        self.set_score_hash(hash_word(word), score);
    }

    pub fn set_score_hash(&mut self, word_hash: u64, score: f32) {
        let mut idx = (word_hash as usize) & OVERLAY_MASK;
        let mut empty_idx = None;

        for _ in 0..16 {
            let (h, _) = self.table[idx];
            if h == word_hash {
                self.table[idx].1 = score;
                return;
            }
            if h == 0 && empty_idx.is_none() {
                empty_idx = Some(idx);
            }
            idx = (idx + 1) & OVERLAY_MASK;
        }

        if let Some(target) = empty_idx {
            self.table[target] = (word_hash, score);
            self.count += 1;
        } else {
            // Collision limit reached, overwrite first slot in probe chain
            let fallback_idx = (word_hash as usize) & OVERLAY_MASK;
            self.table[fallback_idx] = (word_hash, score);
        }
    }

    /// Increment personal score on user word commit (+1.0 boost, capped at 5.0)
    pub fn record_commit(&mut self, word: &str) {
        let h = hash_word(word);
        let curr = self.boost_for_hash(h);
        let new_score = (curr + 1.0).min(5.0);
        self.set_score_hash(h, new_score);
    }

    /// Penalize word score on immediate backspace/undo (-1.5 penalty, clamped at 0.0)
    pub fn penalize(&mut self, word: &str) {
        let h = hash_word(word);
        let curr = self.boost_for_hash(h);
        if curr > 0.0 {
            let new_score = (curr - 1.5).max(0.0);
            self.set_score_hash(h, new_score);
        }
    }
}

/// Truncate preceding context slice at the most recent sentence boundary (। , ? , ! , \n).
/// Words prior to and including the sentence boundary are removed so they do not
/// cross-contaminate next-sentence candidate scoring.
#[inline]
pub fn truncate_at_sentence_boundary<'a>(context: &'a [&'a str]) -> &'a [&'a str] {
    for (idx, token) in context.iter().enumerate().rev() {
        if token.contains('।')
            || token.contains('?')
            || token.contains('!')
            || token.contains('\n')
        {
            return &context[idx + 1..];
        }
    }
    context
}

#[derive(Debug, Clone, Default)]
pub struct ContextScorer {
    lm: LanguageModel,
    personal_overlay: PersonalScoreOverlay,
    embeddings: WordEmbeddings,
}

impl ContextScorer {
    pub fn new() -> Self {
        let mut lm = LanguageModel::new();
        lm.load_from_system_paths();
        Self {
            lm,
            personal_overlay: PersonalScoreOverlay::new(),
            embeddings: WordEmbeddings::new(),
        }
    }

    pub fn with_language_model(lm: LanguageModel) -> Self {
        Self {
            lm,
            personal_overlay: PersonalScoreOverlay::new(),
            embeddings: WordEmbeddings::new(),
        }
    }

    pub fn with_overlay(lm: LanguageModel, personal_overlay: PersonalScoreOverlay) -> Self {
        Self {
            lm,
            personal_overlay,
            embeddings: WordEmbeddings::new(),
        }
    }

    pub fn with_embeddings(
        lm: LanguageModel,
        personal_overlay: PersonalScoreOverlay,
        embeddings: WordEmbeddings,
    ) -> Self {
        Self {
            lm,
            personal_overlay,
            embeddings,
        }
    }

    /// Access the underlying LanguageModel directly without re-allocating
    pub fn lm(&self) -> &LanguageModel {
        &self.lm
    }

    /// Access the personal score overlay
    pub fn personal_overlay(&self) -> &PersonalScoreOverlay {
        &self.personal_overlay
    }

    /// Mutable access to the personal score overlay
    pub fn personal_overlay_mut(&mut self) -> &mut PersonalScoreOverlay {
        &mut self.personal_overlay
    }

    /// Access the static word embeddings engine
    pub fn embeddings(&self) -> &WordEmbeddings {
        &self.embeddings
    }

    /// Mutable access to the static word embeddings engine
    pub fn embeddings_mut(&mut self) -> &mut WordEmbeddings {
        &mut self.embeddings
    }

    /// Record a committed word to personalize scoring
    pub fn record_commit(&mut self, word: &str) {
        self.personal_overlay.record_commit(word);
    }

    /// Penalize a word when reverted on backspace
    pub fn penalize(&mut self, word: &str) {
        self.personal_overlay.penalize(word);
    }

    /// Score and re-rank candidate list based on multi-token preceding context and personal overlay.
    /// This allocates a new Vec. For zero-allocation hot paths, use `rank_candidates_in_place`.
    pub fn rank_candidates(&self, context: &[&str], candidates: &[String]) -> Vec<String> {
        self.rank_candidates_bidirectional(context, None, candidates)
    }

    /// Score and re-rank candidate list with bidirectional context (preceding words + following word).
    pub fn rank_candidates_bidirectional(
        &self,
        context: &[&str],
        right_word: Option<&str>,
        candidates: &[String],
    ) -> Vec<String> {
        let mut cloned = candidates.to_vec();
        self.rank_candidates_in_place_bidirectional(context, right_word, &mut cloned);
        cloned
    }

    /// Zero-allocation candidate ranking. Modifies the candidates slice in-place.
    pub fn rank_candidates_in_place(&self, context: &[&str], candidates: &mut [String]) {
        self.rank_candidates_in_place_bidirectional(context, None, candidates);
    }

    /// Bi-directional zero-allocation candidate ranking.
    /// Modifies `candidates` in-place taking into account preceding context and following word.
    pub fn rank_candidates_in_place_bidirectional(
        &self,
        context: &[&str],
        right_word: Option<&str>,
        candidates: &mut [String],
    ) {
        if candidates.len() <= 1 {
            return;
        }

        let clean_context = truncate_at_sentence_boundary(context);
        let prev1 = clean_context.last().copied();
        let prev2 = if clean_context.len() >= 2 {
            Some(clean_context[clean_context.len() - 2])
        } else {
            None
        };

        let clean_next = right_word.and_then(|w| {
            let trimmed = w.trim_matches(|c: char| {
                c.is_ascii_punctuation()
                    || c == '।'
                    || c == '—'
                    || c == '‘'
                    || c == '’'
                    || c == '“'
                    || c == '”'
                    || c == '\''
                    || c == '"'
                    || c == ','
            });
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        });

        // Stack-allocated array for zero-allocation scoring (up to 32 candidates)
        let len = candidates.len().min(32);
        let mut scores = [0.0; 32];
        for i in 0..len {
            let cand = &candidates[i];
            let lm_score = if prev1.is_some() {
                self.lm.score_candidate(prev2, prev1, cand)
            } else {
                self.lm.score_candidate(None, None, cand)
            };
            let personal_boost = self.personal_overlay.boost_for(cand);
            let grammar_boost = compute_honorific_agreement_boost(prev2, prev1, cand);
            let right_boost = if let Some(next) = clean_next {
                // Score transition from candidate to following word: P(next | cand)
                let p = self.lm.score_candidate(None, Some(cand), next);
                (p + 3.0) * 0.4
            } else {
                0.0
            };
            let semantic_boost = match (prev1, clean_next) {
                (Some(prev), Some(next)) => {
                    self.embeddings.semantic_boost(prev, cand)
                        + self.embeddings.semantic_boost(next, cand) * 0.5
                }
                (Some(prev), None) => self.embeddings.semantic_boost(prev, cand),
                (None, Some(next)) => self.embeddings.semantic_boost(next, cand) * 0.5,
                (None, None) => 0.0,
            };
            let position_penalty = i as f32 * 0.05;
            scores[i] = lm_score + personal_boost + grammar_boost + right_boost + semantic_boost - position_penalty;
        }

        // Simple insertion sort to sort `candidates` and `scores` in tandem.
        // For N <= 32 this is extremely fast and entirely stack-based.
        for i in 1..len {
            let mut j = i;
            while j > 0 && scores[j] > scores[j - 1] {
                scores.swap(j, j - 1);
                candidates.swap(j, j - 1);
                j -= 1;
            }
        }
    }

    /// Compute context score boost for homophone pairs
    pub fn score_homophone_boost(&self, context: &[&str], candidate: &str) -> i32 {
        let clean_context = truncate_at_sentence_boundary(context);
        let personal = self.personal_overlay.boost_for(candidate);
        let personal_boost = (personal * 200.0) as i32;

        if clean_context.is_empty() {
            return personal_boost;
        }

        let prev1 = clean_context.last().copied();
        let prev2 = if clean_context.len() >= 2 {
            Some(clean_context[clean_context.len() - 2])
        } else {
            None
        };

        let sem_boost = if let Some(p) = prev1 {
            (self.embeddings.semantic_boost(p, candidate) * 400.0) as i32
        } else {
            0
        };

        let score = self.lm.score_candidate(prev2, prev1, candidate);
        if score > -1.0 {
            1000 + personal_boost + sem_boost
        } else if score > -2.0 {
            500 + personal_boost + sem_boost
        } else {
            personal_boost + sem_boost
        }
    }
}

/// Computes grammar boost / penalty based on Bengali honorific concordance.
///
/// Ensures formal pronouns (e.g. আপনি) boost formal verb endings (e.g. করবেন, আছেন)
/// and penalize familiar/intimate inflections (e.g. করবে, করবি).
#[inline]
pub fn compute_honorific_agreement_boost(prev2: Option<&str>, prev1: Option<&str>, cand: &str) -> f32 {
    let subject = match (prev1, prev2) {
        (Some("আপনি" | "আপনারা" | "তিনি" | "তাঁরা"), _) => Some(1), // Formal
        (_, Some("আপনি" | "আপনারা" | "তিনি" | "তাঁরা")) => Some(1),
        (Some("তুমি" | "তোমরা"), _) => Some(2), // Familiar
        (_, Some("তুমি" | "তোমরা")) => Some(2),
        (Some("তুই" | "তোরা"), _) => Some(3), // Intimate
        (_, Some("তুই" | "তোরা")) => Some(3),
        _ => None,
    };

    let Some(honorific_tier) = subject else {
        return 0.0;
    };

    let is_formal_verb = cand.ends_with("েন")
        || cand.ends_with("বেন")
        || cand.ends_with("ছেন")
        || cand.ends_with("লেন")
        || cand.ends_with("তেন")
        || cand.ends_with("ন");
    let is_familiar_verb = cand.ends_with("বে")
        || cand.ends_with("ছো")
        || cand.ends_with("লে")
        || cand.ends_with("তে")
        || cand.ends_with("ও")
        || cand.ends_with("রো")
        || cand.ends_with("লো");
    let is_intimate_verb = cand.ends_with("বি")
        || cand.ends_with("ছিস")
        || cand.ends_with("লি")
        || cand.ends_with("তিস")
        || cand.ends_with("িস");

    match honorific_tier {
        1 => {
            // Formal
            if is_formal_verb { 2.5 }
            else if is_intimate_verb { -4.0 }
            else if is_familiar_verb { -2.5 }
            else { 0.0 }
        }
        2 => {
            // Familiar
            if is_familiar_verb { 2.5 }
            else if is_formal_verb { -2.5 }
            else if is_intimate_verb { -3.5 }
            else { 0.0 }
        }
        3 => {
            // Intimate
            if is_intimate_verb { 3.0 }
            else if is_formal_verb { -4.0 }
            else if is_familiar_verb { -3.0 }
            else { 0.0 }
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_ranking() {
        let scorer = ContextScorer::new();
        let candidates = vec!["পড়া".to_string(), "পরা".to_string()];

        // After "শার্ট", "পরা" should be ranked #1
        let ranked_shirt = scorer.rank_candidates(&["শার্ট"], &candidates);
        assert_eq!(ranked_shirt[0], "পরা");

        // After "বই", "পড়া" should be ranked #1
        let ranked_book = scorer.rank_candidates(&["বই"], &candidates);
        assert_eq!(ranked_book[0], "পড়া");
    }

    #[test]
    fn test_bidirectional_ranking() {
        let scorer = ContextScorer::new();
        let mut candidates = vec!["পড়া".to_string(), "পরা".to_string()];
        // With right context "হচ্ছে", bigram "পড়া হচ্ছে" is much more plausible than "পরা হচ্ছে"
        scorer.rank_candidates_in_place_bidirectional(&["একটি", "বই"], Some("হচ্ছে"), &mut candidates);
        assert_eq!(candidates[0], "পড়া");
    }

    #[test]
    fn test_sentence_boundary_reset() {
        let context = &["তুমি", "কেমন", "আছো?", "আমি"];
        let truncated = truncate_at_sentence_boundary(context);
        assert_eq!(truncated, &["আমি"]);

        let context_dari = &["আমি", "ভাত", "খেয়েছি।"];
        let truncated_dari = truncate_at_sentence_boundary(context_dari);
        assert_eq!(truncated_dari, &[] as &[&str]);
    }

    #[test]
    fn test_personal_score_overlay() {
        let mut overlay = PersonalScoreOverlay::new();
        assert_eq!(overlay.boost_for("বিশেষ"), 0.0);

        overlay.record_commit("বিশেষ");
        assert!(overlay.boost_for("বিশেষ") >= 1.0);

        overlay.penalize("বিশেষ");
        assert!(overlay.boost_for("বিশেষ") < 1.0);
    }

    #[test]
    fn test_honorific_agreement() {
        let scorer = ContextScorer::new();
        let candidates = vec!["করবে".to_string(), "করবেন".to_string()];

        // After formal "আপনি", "করবেন" must outrank "করবে"
        let formal_ranked = scorer.rank_candidates(&["আপনি"], &candidates);
        assert_eq!(formal_ranked[0], "করবেন");

        // After familiar "তুমি", "করবে" must outrank "করবেন"
        let familiar_ranked = scorer.rank_candidates(&["তুমি"], &candidates);
        assert_eq!(familiar_ranked[0], "করবে");
    }

    #[test]
    fn test_semantic_ranking_with_embeddings() {
        let scorer = ContextScorer::new();

        // 1. Semantic boost for temporal words:
        // After "আজ", "কাল" should get a significant semantic boost over unrelated "ভাত"
        let mut temporal_cands = vec!["ভাত".to_string(), "কাল".to_string()];
        scorer.rank_candidates_in_place_bidirectional(&["আজ"], None, &mut temporal_cands);
        assert_eq!(temporal_cands[0], "কাল");

        // 2. Homophone boost with semantic context:
        // In context of "বই", "পড়া" (reading) gets higher homophone boost than "পরা" (wearing)
        let boost_reading = scorer.score_homophone_boost(&["বই"], "পড়া");
        let boost_wearing = scorer.score_homophone_boost(&["বই"], "পরা");
        assert!(boost_reading > boost_wearing, "Expected reading boost {} > wearing boost {}", boost_reading, boost_wearing);

        // In context of "শার্ট", "পরা" (wearing) gets higher homophone boost than "পড়া" (reading)
        let boost_shirt_wearing = scorer.score_homophone_boost(&["শার্ট"], "পরা");
        let boost_shirt_reading = scorer.score_homophone_boost(&["শার্ট"], "পড়া");
        assert!(boost_shirt_wearing > boost_shirt_reading, "Expected shirt wearing boost {} > reading boost {}", boost_shirt_wearing, boost_shirt_reading);
    }
}
