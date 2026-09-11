//! Global Beam Search Sequence Decoder for Bengali Sentences

use crate::lm::LanguageModel;

#[derive(Debug, Clone)]
struct BeamCandidate {
    path: Vec<String>,
    score: f32,
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

    /// Decode a sequence of token candidate lists into the globally optimal sentence path
    pub fn decode(&self, sequence_candidates: &[Vec<String>]) -> Vec<String> {
        if sequence_candidates.is_empty() {
            return Vec::new();
        }

        let mut beams: Vec<BeamCandidate> = Vec::new();

        // Initialize with first token options
        if let Some(first_cands) = sequence_candidates.first() {
            for cand in first_cands.iter().take(self.beam_width) {
                let score = self.lm.score_candidate(None, None, cand);
                beams.push(BeamCandidate {
                    path: vec![cand.clone()],
                    score,
                });
            }
        }

        // Iterate through remaining tokens
        for token_cands in sequence_candidates.iter().skip(1) {
            let mut next_beams: Vec<BeamCandidate> = Vec::with_capacity(beams.len() * token_cands.len());

            for beam in &beams {
                let prev1 = beam.path.last().map(|s| s.as_str());
                let prev2 = if beam.path.len() >= 2 {
                    Some(beam.path[beam.path.len() - 2].as_str())
                } else {
                    None
                };

                for (cand_idx, cand) in token_cands.iter().enumerate() {
                    let trans_score = self.lm.score_candidate(prev2, prev1, cand);
                    let rank_penalty = cand_idx as f32 * 0.1;
                    let total_score = beam.score + trans_score - rank_penalty;

                    let mut new_path = beam.path.clone();
                    new_path.push(cand.clone());

                    next_beams.push(BeamCandidate {
                        path: new_path,
                        score: total_score,
                    });
                }
            }

            // Prune to top beam_width
            next_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            next_beams.truncate(self.beam_width);
            beams = next_beams;
        }

        beams
            .first()
            .map(|b| b.path.clone())
            .unwrap_or_else(Vec::new)
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
