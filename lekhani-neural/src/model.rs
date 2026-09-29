//! Pure Rust Micro-GRU Inference Model
//!
//! Compact 2-layer Gated Recurrent Unit (GRU) with tied word embeddings.
//! Designed for low-latency background prediction (< 20 ms) with strictly bounded memory (< 15 MB).

use serde::{Deserialize, Serialize};

/// Activation function: Sigmoid
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Activation function: Tanh
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    x.tanh()
}

/// Weights for a single GRU layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GruLayerWeights {
    pub input_dim: usize,
    pub hidden_dim: usize,
    // Reset gate weights: W_ir (hidden x input), W_hr (hidden x hidden), b_r (hidden)
    pub w_ir: Vec<f32>,
    pub w_hr: Vec<f32>,
    pub b_r: Vec<f32>,
    // Update gate weights: W_iz, W_hz, b_z
    pub w_iz: Vec<f32>,
    pub w_hz: Vec<f32>,
    pub b_z: Vec<f32>,
    // Candidate gate weights: W_in, W_hn, b_in, b_hn
    pub w_in: Vec<f32>,
    pub w_hn: Vec<f32>,
    pub b_in: Vec<f32>,
    pub b_hn: Vec<f32>,
}

impl GruLayerWeights {
    pub fn new_zeros(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            w_ir: vec![0.0; hidden_dim * input_dim],
            w_hr: vec![0.0; hidden_dim * hidden_dim],
            b_r: vec![0.0; hidden_dim],
            w_iz: vec![0.0; hidden_dim * input_dim],
            w_hz: vec![0.0; hidden_dim * hidden_dim],
            b_z: vec![0.0; hidden_dim],
            w_in: vec![0.0; hidden_dim * input_dim],
            w_hn: vec![0.0; hidden_dim * hidden_dim],
            b_in: vec![0.0; hidden_dim],
            b_hn: vec![0.0; hidden_dim],
        }
    }

    /// Single GRU step: takes input x and prev hidden state h_prev, writes to h_out
    pub fn step(&self, x: &[f32], h_prev: &[f32], h_out: &mut [f32]) {
        let h_dim = self.hidden_dim;
        let in_dim = self.input_dim;

        for i in 0..h_dim {
            // 1. Reset gate: r = sigmoid(W_ir * x + W_hr * h_prev + b_r)
            let mut r_val = self.b_r[i];
            let w_ir_row = &self.w_ir[i * in_dim..(i + 1) * in_dim];
            for (&w, &val) in w_ir_row.iter().zip(x.iter()) {
                r_val += w * val;
            }
            let w_hr_row = &self.w_hr[i * h_dim..(i + 1) * h_dim];
            for (&w, &val) in w_hr_row.iter().zip(h_prev.iter()) {
                r_val += w * val;
            }
            let r = sigmoid(r_val);

            // 2. Update gate: z = sigmoid(W_iz * x + W_hz * h_prev + b_z)
            let mut z_val = self.b_z[i];
            let w_iz_row = &self.w_iz[i * in_dim..(i + 1) * in_dim];
            for (&w, &val) in w_iz_row.iter().zip(x.iter()) {
                z_val += w * val;
            }
            let w_hz_row = &self.w_hz[i * h_dim..(i + 1) * h_dim];
            for (&w, &val) in w_hz_row.iter().zip(h_prev.iter()) {
                z_val += w * val;
            }
            let z = sigmoid(z_val);

            // 3. Candidate hidden state: n = tanh(W_in * x + b_in + r * (W_hn * h_prev + b_hn))
            let mut n_in = self.b_in[i];
            let w_in_row = &self.w_in[i * in_dim..(i + 1) * in_dim];
            for (&w, &val) in w_in_row.iter().zip(x.iter()) {
                n_in += w * val;
            }
            let mut n_hn = self.b_hn[i];
            let w_hn_row = &self.w_hn[i * h_dim..(i + 1) * h_dim];
            for (&w, &val) in w_hn_row.iter().zip(h_prev.iter()) {
                n_hn += w * val;
            }
            let n = fast_tanh(n_in + r * n_hn);

            // 4. Output state: h = (1 - z) * n + z * h_prev
            h_out[i] = (1.0 - z) * n + z * h_prev[i];
        }
    }
}

/// 2-Layer Micro-GRU Model with Tied Embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroGruModel {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    /// Embedding matrix: vocab_size x embedding_dim
    pub embeddings: Vec<f32>,
    /// Layer 1: input_dim = embedding_dim, hidden_dim
    pub layer1: GruLayerWeights,
    /// Layer 2: input_dim = hidden_dim, hidden_dim
    pub layer2: GruLayerWeights,
}

impl MicroGruModel {
    pub fn new(vocab_size: usize, embedding_dim: usize, hidden_dim: usize) -> Self {
        Self {
            vocab_size,
            embedding_dim,
            hidden_dim,
            embeddings: vec![0.0; vocab_size * embedding_dim],
            layer1: GruLayerWeights::new_zeros(embedding_dim, hidden_dim),
            layer2: GruLayerWeights::new_zeros(hidden_dim, hidden_dim),
        }
    }

    /// Forward pass through sequence of tokens.
    /// Returns the projected logits over the vocabulary for the next token.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let mut h1 = vec![0.0; self.hidden_dim];
        let mut h2 = vec![0.0; self.hidden_dim];
        let mut next_h1 = vec![0.0; self.hidden_dim];
        let mut next_h2 = vec![0.0; self.hidden_dim];

        let mut x = vec![0.0; self.embedding_dim];

        for &id in token_ids {
            let id = (id as usize).min(self.vocab_size.saturating_sub(1));
            // Read embedding row
            let emb_start = id * self.embedding_dim;
            let emb_end = emb_start + self.embedding_dim;
            if emb_end <= self.embeddings.len() {
                x.copy_from_slice(&self.embeddings[emb_start..emb_end]);
            }

            // Layer 1
            self.layer1.step(&x, &h1, &mut next_h1);
            h1.copy_from_slice(&next_h1);

            // Layer 2
            self.layer2.step(&h1, &h2, &mut next_h2);
            h2.copy_from_slice(&next_h2);
        }

        // Project final hidden state h2 onto vocabulary embeddings (Tied embeddings projection)
        let mut logits = vec![0.0; self.vocab_size];
        for (v, logit) in logits.iter_mut().enumerate() {
            let emb_start = v * self.embedding_dim;
            let emb_end = emb_start + self.embedding_dim;
            if emb_end <= self.embeddings.len() {
                let mut dot = 0.0;
                let emb_slice = &self.embeddings[emb_start..emb_end];
                for (&h_val, &emb_val) in h2.iter().zip(emb_slice.iter()).take(self.embedding_dim.min(self.hidden_dim)) {
                    dot += h_val * emb_val;
                }
                *logit = dot;
            }
        }


        logits
    }

    /// Top-k next token IDs and log-probabilities
    pub fn predict_top_k(&self, token_ids: &[u32], k: usize) -> Vec<(u32, f32)> {
        if token_ids.is_empty() {
            return Vec::new();
        }

        let logits = self.forward(token_ids);

        // Compute log-softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let log_sum_exp = max_logit + sum_exp.ln();

        let mut scored: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &l)| (idx as u32, l - log_sum_exp))
            .collect();

        // Sort descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_gru_forward_dimension_safety() {
        let model = MicroGruModel::new(64, 16, 16);
        let top = model.predict_top_k(&[1, 5, 12], 5);
        assert_eq!(top.len(), 5);
        assert!(top[0].1 <= 0.0);
    }
}
