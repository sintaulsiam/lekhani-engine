//! Global Beam Search Sequence Decoder for Bengali Sentences

use crate::lm::LanguageModel;

#[derive(Copy, Clone, Debug)]
struct BeamNode<'a> {
    cand: &'a str,
    parent_idx: Option<usize>,
    score: f32,
    cand_idx: usize,
}

#[derive(Debug, Clone, Default)]
pub struct BeamSearchDecoder {
    lm: LanguageModel,
    beam_width: usize,
}

impl BeamSearchDecoder {
    pub fn new() -> Self {
        Self {
            lm: LanguageModel::new(),
            beam_width: 4,
        }
    }

    pub fn with_beam_width(beam_width: usize) -> Self {
        Self {
            lm: LanguageModel::new(),
            beam_width,
        }
    }

    pub fn with_language_model(lm: LanguageModel, beam_width: usize) -> Self {
        Self { lm, beam_width }
    }

    /// Decode a sequence of token candidate lists into the globally optimal sentence path
    /// using a zero-allocation parent-pointer arena across beam layers.
    pub fn decode(&self, sequence_candidates: &[Vec<String>]) -> Vec<String> {
        if sequence_candidates.is_empty() {
            return Vec::new();
        }

        let mut layers: Vec<Vec<BeamNode<'_>>> = Vec::with_capacity(sequence_candidates.len());

        // Initialize layer 0 with first token options
        if let Some(first_cands) = sequence_candidates.first() {
            if first_cands.is_empty() {
                return Vec::new();
            }
            let mut layer0: Vec<BeamNode<'_>> = Vec::with_capacity(self.beam_width);
            for (cand_idx, cand) in first_cands.iter().take(self.beam_width).enumerate() {
                let score = self.lm.score_candidate(None, None, cand);
                let rank_penalty = cand_idx as f32 * 0.1;
                layer0.push(BeamNode {
                    cand: cand.as_str(),
                    parent_idx: None,
                    score: score - rank_penalty,
                    cand_idx,
                });
            }
            layers.push(layer0);
        }

        // Iterate through remaining tokens
        for t in 1..sequence_candidates.len() {
            let token_cands = &sequence_candidates[t];
            if token_cands.is_empty() {
                continue;
            }

            let prev_layer = &layers[t - 1];
            let mut next_layer: Vec<BeamNode<'_>> =
                Vec::with_capacity(prev_layer.len() * token_cands.len());

            for (parent_idx, parent_node) in prev_layer.iter().enumerate() {
                let prev1 = Some(parent_node.cand);
                let prev2 = if t >= 2 {
                    parent_node.parent_idx.map(|p_idx| layers[t - 2][p_idx].cand)
                } else {
                    None
                };

                for (cand_idx, cand) in token_cands.iter().enumerate() {
                    let trans_score = self.lm.score_candidate(prev2, prev1, cand);
                    let rank_penalty = cand_idx as f32 * 0.1;
                    let total_score = parent_node.score + trans_score - rank_penalty;

                    next_layer.push(BeamNode {
                        cand: cand.as_str(),
                        parent_idx: Some(parent_idx),
                        score: total_score,
                        cand_idx,
                    });
                }
            }

            // Prune to top beam_width
            next_layer.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or_else(|| a.cand_idx.cmp(&b.cand_idx))
            });
            next_layer.truncate(self.beam_width);
            layers.push(next_layer);
        }

        // Backtrack to reconstruct winning path
        let last_layer = match layers.last() {
            Some(l) if !l.is_empty() => l,
            _ => return Vec::new(),
        };

        let mut path: Vec<String> = Vec::with_capacity(layers.len());
        let mut curr_node = &last_layer[0];
        path.push(curr_node.cand.to_string());

        for t in (1..layers.len()).rev() {
            if let Some(parent_idx) = curr_node.parent_idx {
                curr_node = &layers[t - 1][parent_idx];
                path.push(curr_node.cand.to_string());
            } else {
                break;
            }
        }

        path.reverse();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_search_decoding() {
        let decoder = BeamSearchDecoder::new();

        let token_options = vec![
            vec!["আমি".to_string()],
            vec!["শার্ট".to_string(), "শিরত".to_string()],
            vec!["পড়া".to_string(), "পরা".to_string()], // should choose "পরা" following "শার্ট"
        ];

        let decoded = decoder.decode(&token_options);
        assert_eq!(decoded, vec!["আমি", "শার্ট", "পরা"]);
    }
}
