//! Multi-Token Deep Context Engine & Homophone Disambiguator

use crate::lm::LanguageModel;

#[derive(Debug, Clone, Default)]
pub struct ContextScorer {
    lm: LanguageModel,
}

impl ContextScorer {
    pub fn new() -> Self {
        Self {
            lm: LanguageModel::new(),
        }
    }

    /// Score and re-rank candidate list based on multi-token preceding context
    pub fn rank_candidates(
        &self,
        context: &[&str],
        candidates: &[String],
    ) -> Vec<String> {
        if candidates.is_empty() {
            return Vec::new();
        }

        if context.is_empty() {
            return candidates.to_vec();
        }

        let prev1 = context.last().copied();
        let prev2 = if context.len() >= 2 {
            Some(context[context.len() - 2])
        } else {
            None
        };

        let mut scored_candidates: Vec<(String, f32, usize)> = candidates
            .iter()
            .enumerate()
            .map(|(orig_idx, cand)| {
                let lm_score = self.lm.score_candidate(prev2, prev1, cand);
                // Combine original ranking priority with LM score
                let position_penalty = orig_idx as f32 * 0.15;
                let total_score = lm_score - position_penalty;
                (cand.clone(), total_score, orig_idx)
            })
            .collect();

        // Sort by total score descending, preserving stable order on close ties
        scored_candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(a.2.cmp(&b.2))
        });

        scored_candidates.into_iter().map(|(cand, _, _)| cand).collect()
    }

    /// Compute context score boost for homophone pairs
    pub fn score_homophone_boost(&self, context: &[&str], candidate: &str) -> i32 {
        if context.is_empty() {
            return 0;
        }

        let prev1 = context.last().copied();
        let prev2 = if context.len() >= 2 {
            Some(context[context.len() - 2])
        } else {
            None
        };

        let score = self.lm.score_candidate(prev2, prev1, candidate);
        if score > -1.0 {
            1000
        } else if score > -2.0 {
            500
        } else {
            0
        }
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
}
