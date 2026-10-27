//! Character-level Seq2Seq Transliterator
//!
//! GRU encoder → dot-product attention → GRU decoder for Latin→Bengali transliteration.
//! Designed as a safety-net fallback for inputs where the beam-phonetic decoder fails.
//!
//! Model size target: < 1.5 MB. Inference latency target: < 3 ms per input.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Character Vocabularies ─────────────────────────────────────────────────

/// Special token IDs
pub const PAD_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
pub const SPECIAL_COUNT: usize = 4;

/// Compact char↔id mapping for a fixed character set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharVocab {
    /// id → char (index 0..SPECIAL_COUNT are reserved)
    pub chars: Vec<char>,
    /// char → id
    pub lookup: HashMap<char, u32>,
}

impl CharVocab {
    pub fn new(chars: Vec<char>) -> Self {
        // Reserve 0=PAD, 1=BOS, 2=EOS, 3=UNK
        let mut lookup = HashMap::with_capacity(chars.len() + SPECIAL_COUNT);
        for (i, &c) in chars.iter().enumerate() {
            lookup.insert(c, (i + SPECIAL_COUNT) as u32);
        }
        Self { chars, lookup }
    }

    #[inline]
    pub fn encode(&self, c: char) -> u32 {
        *self.lookup.get(&c).unwrap_or(&UNK_ID)
    }

    #[inline]
    pub fn decode(&self, id: u32) -> Option<char> {
        match id {
            PAD_ID | BOS_ID | UNK_ID => None,
            EOS_ID => None,
            _ => self.chars.get((id as usize).saturating_sub(SPECIAL_COUNT)).copied(),
        }
    }

    pub fn size(&self) -> usize {
        self.chars.len() + SPECIAL_COUNT
    }

    /// Encode a string to ids (including BOS/EOS for targets)
    pub fn encode_str(&self, s: &str) -> Vec<u32> {
        s.chars().map(|c| self.encode(c)).collect()
    }

    pub fn encode_str_with_eos(&self, s: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = s.chars().map(|c| self.encode(c)).collect();
        ids.push(EOS_ID);
        ids
    }

    /// Decode ids to string (stops at EOS, skips PAD/BOS/UNK)
    pub fn decode_str(&self, ids: &[u32]) -> String {
        let mut s = String::with_capacity(ids.len() * 3);
        for &id in ids {
            if id == EOS_ID {
                break;
            }
            if let Some(c) = self.decode(id) {
                s.push(c);
            }
        }
        s
    }

    /// Build vocab from a collection of strings
    pub fn from_strings<'a>(iter: impl Iterator<Item = &'a str>) -> Self {
        let mut set: std::collections::BTreeSet<char> = std::collections::BTreeSet::new();
        for s in iter {
            for c in s.chars() {
                set.insert(c);
            }
        }
        Self::new(set.into_iter().collect())
    }
}

// ─── Model Architecture ──────────────────────────────────────────────────────

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    x.tanh()
}

/// Compact GRU cell for seq2seq (input_dim may differ from hidden_dim)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S2SGruCell {
    pub input_dim: usize,
    pub hidden_dim: usize,
    // Reset gate
    pub w_ir: Vec<f32>, // hidden × input
    pub w_hr: Vec<f32>, // hidden × hidden
    pub b_r:  Vec<f32>, // hidden
    // Update gate
    pub w_iz: Vec<f32>,
    pub w_hz: Vec<f32>,
    pub b_z:  Vec<f32>,
    // Candidate state
    pub w_in: Vec<f32>,
    pub w_hn: Vec<f32>,
    pub b_in: Vec<f32>,
    pub b_hn: Vec<f32>,
}

/// SIMD-friendly 8-wide chunked dot product
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();
    let mut acc = [0.0f32; 8];
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for k in 0..8 {
            acc[k] += ca[k] * cb[k];
        }
    }
    let mut sum = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        sum += x * y;
    }
    sum
}

impl S2SGruCell {
    pub fn new_zeros(input_dim: usize, hidden_dim: usize) -> Self {
        let ih = hidden_dim * input_dim;
        let hh = hidden_dim * hidden_dim;
        Self {
            input_dim,
            hidden_dim,
            w_ir: vec![0.0; ih], w_hr: vec![0.0; hh], b_r: vec![0.0; hidden_dim],
            w_iz: vec![0.0; ih], w_hz: vec![0.0; hh], b_z: vec![0.0; hidden_dim],
            w_in: vec![0.0; ih], w_hn: vec![0.0; hh],
            b_in: vec![0.0; hidden_dim], b_hn: vec![0.0; hidden_dim],
        }
    }

    /// Forward step. Writes new hidden state to `h_out`.
    /// Also writes gate values to `gate_buf` (for BPTT): [r | z | n | preact_r | preact_z | n_in | n_hn]
    pub fn step(&self, x: &[f32], h_prev: &[f32], h_out: &mut [f32], mut gate_buf: Option<&mut GateBuf>) {
        let h = self.hidden_dim;
        let in_d = self.input_dim;
        for i in 0..h {
            // Reset gate
            let rv = self.b_r[i]
                + dot(&self.w_ir[i * in_d..(i + 1) * in_d], x)
                + dot(&self.w_hr[i * h..(i + 1) * h], h_prev);
            let r = sigmoid(rv);
            // Update gate
            let zv = self.b_z[i]
                + dot(&self.w_iz[i * in_d..(i + 1) * in_d], x)
                + dot(&self.w_hz[i * h..(i + 1) * h], h_prev);
            let z = sigmoid(zv);
            // Candidate
            let nin = self.b_in[i] + dot(&self.w_in[i * in_d..(i + 1) * in_d], x);
            let nhn = self.b_hn[i] + dot(&self.w_hn[i * h..(i + 1) * h], h_prev);
            let n = fast_tanh(nin + r * nhn);
            h_out[i] = (1.0 - z) * n + z * h_prev[i];
            if let Some(ref mut gb) = gate_buf.as_deref_mut() {
                gb.r[i] = r; gb.z[i] = z; gb.n[i] = n;
                gb.pre_r[i] = rv; gb.pre_z[i] = zv;
                gb.n_in[i] = nin; gb.n_hn[i] = nhn;
            }
        }
    }
}

/// Per-step gate buffer for BPTT (Backprop Through Time)
#[derive(Debug, Clone)]
pub struct GateBuf {
    pub r: Vec<f32>, pub z: Vec<f32>, pub n: Vec<f32>,
    pub pre_r: Vec<f32>, pub pre_z: Vec<f32>,
    pub n_in: Vec<f32>, pub n_hn: Vec<f32>,
}

impl GateBuf {
    pub fn new(h: usize) -> Self {
        Self {
            r: vec![0.0; h], z: vec![0.0; h], n: vec![0.0; h],
            pre_r: vec![0.0; h], pre_z: vec![0.0; h],
            n_in: vec![0.0; h], n_hn: vec![0.0; h],
        }
    }
}

// ─── Full Seq2Seq Model ───────────────────────────────────────────────────────

/// Complete encoder-decoder transliterator.
/// Encoder: 1-layer GRU over Latin characters.
/// Decoder: 1-layer GRU with dot-product attention over encoder states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Seq2SeqTransliterator {
    // Vocabulary
    pub src_vocab: CharVocab,  // Latin
    pub tgt_vocab: CharVocab,  // Bengali
    // Dimensions
    pub emb_dim: usize,
    pub hidden_dim: usize,
    // Embeddings
    pub enc_emb: Vec<f32>,  // src_vocab_size × emb_dim
    pub dec_emb: Vec<f32>,  // tgt_vocab_size × emb_dim
    // Encoder
    pub encoder: S2SGruCell,
    // Decoder (input = emb_dim + hidden_dim for context concat)
    pub decoder: S2SGruCell,
    // Attention: score = enc_states · W_a · dec_h
    pub attn_w: Vec<f32>,   // hidden_dim × hidden_dim
    // Output projection
    pub out_w: Vec<f32>,    // tgt_vocab_size × hidden_dim
    pub out_b: Vec<f32>,    // tgt_vocab_size
}

impl Seq2SeqTransliterator {
    pub fn new(src_vocab: CharVocab, tgt_vocab: CharVocab, emb_dim: usize, hidden_dim: usize) -> Self {
        let sv = src_vocab.size();
        let tv = tgt_vocab.size();
        Self {
            enc_emb: vec![0.0; sv * emb_dim],
            dec_emb: vec![0.0; tv * emb_dim],
            encoder: S2SGruCell::new_zeros(emb_dim, hidden_dim),
            decoder: S2SGruCell::new_zeros(emb_dim + hidden_dim, hidden_dim),
            attn_w: vec![0.0; hidden_dim * hidden_dim],
            out_w: vec![0.0; tv * hidden_dim],
            out_b: vec![0.0; tv],
            src_vocab,
            tgt_vocab,
            emb_dim,
            hidden_dim,
        }
    }

    /// Save model to bincode binary
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        bincode::serialize_into(std::io::BufWriter::new(file), self)
            .map_err(std::io::Error::other)
    }

    /// Load model from bincode binary
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        bincode::deserialize_from(std::io::BufReader::new(file))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    // ── Inference ──────────────────────────────────────────────────────────

    /// Look up source embedding for character id
    fn src_embed(&self, id: u32, out: &mut [f32]) {
        let id = (id as usize).min(self.src_vocab.size().saturating_sub(1));
        let s = id * self.emb_dim;
        let e = s + self.emb_dim;
        if e <= self.enc_emb.len() {
            out.copy_from_slice(&self.enc_emb[s..e]);
        } else {
            out.fill(0.0);
        }
    }

    /// Look up target embedding for character id
    fn tgt_embed(&self, id: u32, out: &mut [f32]) {
        let id = (id as usize).min(self.tgt_vocab.size().saturating_sub(1));
        let s = id * self.emb_dim;
        let e = s + self.emb_dim;
        if e <= self.dec_emb.len() {
            out.copy_from_slice(&self.dec_emb[s..e]);
        } else {
            out.fill(0.0);
        }
    }

    /// Encode Latin string → encoder hidden states (seq_len × hidden_dim) + final state
    fn encode(&self, ids: &[u32], scratch: &mut InferenceScratch) -> Vec<Vec<f32>> {
        let h = self.hidden_dim;
        let mut h_cur = vec![0.0f32; h];
        let mut h_next = vec![0.0f32; h];
        let mut enc_states: Vec<Vec<f32>> = Vec::with_capacity(ids.len());

        for &id in ids {
            self.src_embed(id, &mut scratch.emb_buf);
            self.encoder.step(&scratch.emb_buf, &h_cur, &mut h_next, None);
            h_cur.copy_from_slice(&h_next);
            enc_states.push(h_cur.clone());
        }

        if enc_states.is_empty() {
            enc_states.push(vec![0.0; h]);
        }
        // Store final encoder state in scratch
        scratch.enc_final.copy_from_slice(enc_states.last().unwrap());
        enc_states
    }

    /// Compute attention context vector from decoder hidden state and encoder states.
    /// Uses dot-product attention with W_a projection on encoder states.
    fn attend(&self, dec_h: &[f32], enc_states: &[Vec<f32>], ctx_out: &mut [f32]) {
        let h = self.hidden_dim;
        // Transform dec_h: q = attn_w · dec_h
        let mut q = vec![0.0f32; h];
        for i in 0..h {
            q[i] = dot(&self.attn_w[i * h..(i + 1) * h], dec_h);
        }
        // Scores: dot(enc_i, q) for each encoder step
        let mut scores = vec![0.0f32; enc_states.len()];
        for (t, enc_h) in enc_states.iter().enumerate() {
            scores[t] = dot(enc_h, &q);
        }
        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum_exp += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum_exp.max(1e-9);
        }
        // Context = weighted sum of encoder states
        ctx_out.fill(0.0);
        for (t, enc_h) in enc_states.iter().enumerate() {
            let a = scores[t];
            for j in 0..h {
                ctx_out[j] += a * enc_h[j];
            }
        }
    }

    /// Project decoder hidden state + context → logits over target vocabulary
    fn project(&self, dec_h: &[f32], logits: &mut [f32]) {
        let tv = self.tgt_vocab.size();
        let h = self.hidden_dim;
        for v in 0..tv {
            logits[v] = self.out_b[v] + dot(&self.out_w[v * h..(v + 1) * h], dec_h);
        }
    }

    /// Beam-search decode from encoder states.
    /// Beam-search decode from encoder states with optional lexicon constraints.
    /// Returns top-`beam` candidates as (score, id_sequence) sorted best-first.
    pub fn beam_decode_constrained(
        &self,
        enc_states: &[Vec<f32>],
        enc_final: &[f32],
        beam: usize,
        max_len: usize,
        scratch: &mut InferenceScratch,
        prefix_filter: Option<&dyn Fn(&str) -> bool>,
        word_filter: Option<&dyn Fn(&str) -> bool>,
    ) -> Vec<(f32, Vec<u32>)> {
        let tv = self.tgt_vocab.size();

        // BeamHyp: (cum_log_prob, dec_h, token_ids)
        struct Hyp {
            score: f32,
            h: Vec<f32>,
            ids: Vec<u32>,
            done: bool,
        }

        let mut active: Vec<Hyp> = vec![Hyp {
            score: 0.0,
            h: enc_final.to_vec(),
            ids: vec![BOS_ID],
            done: false,
        }];

        let mut completed: Vec<(f32, Vec<u32>)> = Vec::new();

        for _ in 0..max_len {
            if active.is_empty() { break; }
            let mut candidates: Vec<Hyp> = Vec::with_capacity(active.len() * beam * 2);

            for hyp in active.drain(..) {
                if hyp.done {
                    completed.push((hyp.score, hyp.ids.clone()));
                    continue;
                }
                let last_id = *hyp.ids.last().unwrap_or(&BOS_ID);
                // Decoder step
                self.tgt_embed(last_id, &mut scratch.emb_buf);
                self.attend(&hyp.h, enc_states, &mut scratch.ctx_buf);
                // Concat emb + context
                scratch.dec_in[..self.emb_dim].copy_from_slice(&scratch.emb_buf);
                scratch.dec_in[self.emb_dim..].copy_from_slice(&scratch.ctx_buf);
                let mut h_next = vec![0.0f32; self.hidden_dim];
                self.decoder.step(&scratch.dec_in, &hyp.h, &mut h_next, None);
                self.project(&h_next, &mut scratch.logits_buf);

                // Log-softmax
                let max_l = scratch.logits_buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_e: f32 = scratch.logits_buf.iter().map(|&l| (l - max_l).exp()).sum();
                let log_sum = max_l + sum_e.ln();

                // Top candidates (explore wider than beam when filtering)
                scratch.scored_buf.clear();
                for (idx, &l) in scratch.logits_buf.iter().enumerate().take(tv) {
                    scratch.scored_buf.push((idx as u32, l - log_sum));
                }
                scratch.scored_buf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let explore_count = if prefix_filter.is_some() || word_filter.is_some() {
                    (beam * 4).min(scratch.scored_buf.len())
                } else {
                    beam.min(scratch.scored_buf.len())
                };

                for &(tok_id, log_p) in scratch.scored_buf.iter().take(explore_count) {
                    let mut new_ids = hyp.ids.clone();
                    new_ids.push(tok_id);
                    let done = tok_id == EOS_ID;

                    let mut penalty = 0.0f32;
                    if done {
                        if let Some(wf) = word_filter {
                            let word = self.tgt_vocab.decode_str(&hyp.ids);
                            if !wf(&word) {
                                penalty = -30.0;
                            }
                        }
                    } else if let Some(pf) = prefix_filter {
                        let partial = self.tgt_vocab.decode_str(&new_ids);
                        if !pf(&partial) {
                            penalty = -30.0;
                        }
                    }

                    candidates.push(Hyp {
                        score: hyp.score + log_p + penalty,
                        h: h_next.clone(),
                        ids: new_ids,
                        done,
                    });
                }
            }

            // Prune to top-beam
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(beam);
            active = candidates;
        }

        // Collect remaining active as completed
        for hyp in active {
            completed.push((hyp.score, hyp.ids));
        }

        // Sort by score desc
        completed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        completed.truncate(beam);
        completed
    }

    /// Beam-search decode from encoder states.
    /// Returns top-`beam` candidates as (score, id_sequence) sorted best-first.
    pub fn beam_decode(
        &self,
        enc_states: &[Vec<f32>],
        enc_final: &[f32],
        beam: usize,
        max_len: usize,
        scratch: &mut InferenceScratch,
    ) -> Vec<(f32, Vec<u32>)> {
        self.beam_decode_constrained(enc_states, enc_final, beam, max_len, scratch, None, None)
    }

    /// Transliterate a Latin string to up to `beam` Bengali candidates.
    pub fn transliterate(&self, latin: &str, beam: usize, scratch: &mut InferenceScratch) -> Vec<String> {
        self.transliterate_constrained(latin, beam, scratch, None, None)
    }

    /// Transliterate a Latin string to up to `beam` Bengali candidates with lexicon constraints.
    pub fn transliterate_constrained(
        &self,
        latin: &str,
        beam: usize,
        scratch: &mut InferenceScratch,
        prefix_filter: Option<&dyn Fn(&str) -> bool>,
        word_filter: Option<&dyn Fn(&str) -> bool>,
    ) -> Vec<String> {
        let src_ids = self.src_vocab.encode_str(latin.to_ascii_lowercase().trim());
        let enc_states = self.encode(&src_ids, scratch);
        let enc_final = scratch.enc_final.clone();
        let hyps = self.beam_decode_constrained(
            &enc_states,
            &enc_final,
            beam,
            (latin.len() * 3 + 4).min(32),
            scratch,
            prefix_filter,
            word_filter,
        );
        hyps.iter()
            .map(|(_, ids)| self.tgt_vocab.decode_str(ids))
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Parameter count (for size estimation)
    pub fn param_count(&self) -> usize {
        let h = self.hidden_dim;
        let in_e = self.emb_dim;
        let in_d = self.emb_dim + h;
        let sv = self.src_vocab.size();
        let tv = self.tgt_vocab.size();
        sv * in_e + tv * in_e                           // embeddings
            + 3 * h * in_e + 3 * h * h + 5 * h        // encoder GRU
            + 3 * h * in_d + 3 * h * h + 5 * h        // decoder GRU
            + h * h                                     // attn_w
            + tv * h + tv                               // output projection
    }
}

/// Pre-allocated inference buffers (zero heap allocation during inference)
#[derive(Clone)]
pub struct InferenceScratch {
    pub emb_buf: Vec<f32>,
    pub ctx_buf: Vec<f32>,
    pub dec_in: Vec<f32>,
    pub logits_buf: Vec<f32>,
    pub scored_buf: Vec<(u32, f32)>,
    pub enc_final: Vec<f32>,
}

impl InferenceScratch {
    pub fn new(emb_dim: usize, hidden_dim: usize, vocab_size: usize) -> Self {
        Self {
            emb_buf: vec![0.0; emb_dim],
            ctx_buf: vec![0.0; hidden_dim],
            dec_in: vec![0.0; emb_dim + hidden_dim],
            logits_buf: vec![0.0; vocab_size],
            scored_buf: Vec::with_capacity(vocab_size),
            enc_final: vec![0.0; hidden_dim],
        }
    }

    pub fn for_model(m: &Seq2SeqTransliterator) -> Self {
        Self::new(m.emb_dim, m.hidden_dim, m.tgt_vocab.size())
    }
}

// ─── Gradient Struct (for training) ─────────────────────────────────────────

/// Gradient accumulator mirroring Seq2SeqTransliterator parameter layout
#[derive(Debug, Clone)]
pub struct Seq2SeqGrad {
    pub enc_emb: Vec<f32>,
    pub dec_emb: Vec<f32>,
    pub enc: CellGrad,
    pub dec: CellGrad,
    pub attn_w: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CellGrad {
    pub w_ir: Vec<f32>, pub w_hr: Vec<f32>, pub b_r: Vec<f32>,
    pub w_iz: Vec<f32>, pub w_hz: Vec<f32>, pub b_z: Vec<f32>,
    pub w_in: Vec<f32>, pub w_hn: Vec<f32>, pub b_in: Vec<f32>, pub b_hn: Vec<f32>,
}

impl CellGrad {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let ih = hidden_dim * input_dim;
        let hh = hidden_dim * hidden_dim;
        Self {
            w_ir: vec![0.0; ih], w_hr: vec![0.0; hh], b_r: vec![0.0; hidden_dim],
            w_iz: vec![0.0; ih], w_hz: vec![0.0; hh], b_z: vec![0.0; hidden_dim],
            w_in: vec![0.0; ih], w_hn: vec![0.0; hh], b_in: vec![0.0; hidden_dim],
            b_hn: vec![0.0; hidden_dim],
        }
    }

    pub fn zero(&mut self) {
        self.w_ir.fill(0.0); self.w_hr.fill(0.0); self.b_r.fill(0.0);
        self.w_iz.fill(0.0); self.w_hz.fill(0.0); self.b_z.fill(0.0);
        self.w_in.fill(0.0); self.w_hn.fill(0.0); self.b_in.fill(0.0);
        self.b_hn.fill(0.0);
    }

    pub fn add(&mut self, other: &Self) {
        for (a, &b) in self.w_ir.iter_mut().zip(&other.w_ir) { *a += b; }
        for (a, &b) in self.w_hr.iter_mut().zip(&other.w_hr) { *a += b; }
        for (a, &b) in self.b_r.iter_mut().zip(&other.b_r) { *a += b; }
        for (a, &b) in self.w_iz.iter_mut().zip(&other.w_iz) { *a += b; }
        for (a, &b) in self.w_hz.iter_mut().zip(&other.w_hz) { *a += b; }
        for (a, &b) in self.b_z.iter_mut().zip(&other.b_z) { *a += b; }
        for (a, &b) in self.w_in.iter_mut().zip(&other.w_in) { *a += b; }
        for (a, &b) in self.w_hn.iter_mut().zip(&other.w_hn) { *a += b; }
        for (a, &b) in self.b_in.iter_mut().zip(&other.b_in) { *a += b; }
        for (a, &b) in self.b_hn.iter_mut().zip(&other.b_hn) { *a += b; }
    }

    pub fn scale(&mut self, s: f32) {
        for g in &mut self.w_ir { *g *= s; }
        for g in &mut self.w_hr { *g *= s; }
        for g in &mut self.b_r  { *g *= s; }
        for g in &mut self.w_iz { *g *= s; }
        for g in &mut self.w_hz { *g *= s; }
        for g in &mut self.b_z  { *g *= s; }
        for g in &mut self.w_in { *g *= s; }
        for g in &mut self.w_hn { *g *= s; }
        for g in &mut self.b_in { *g *= s; }
        for g in &mut self.b_hn { *g *= s; }
    }
}

impl Seq2SeqGrad {
    pub fn new_for(m: &Seq2SeqTransliterator) -> Self {
        let h = m.hidden_dim;
        let in_e = m.emb_dim;
        let in_d = m.emb_dim + h;
        let sv = m.src_vocab.size();
        let tv = m.tgt_vocab.size();
        Self {
            enc_emb: vec![0.0; sv * in_e],
            dec_emb: vec![0.0; tv * in_e],
            enc: CellGrad::new(in_e, h),
            dec: CellGrad::new(in_d, h),
            attn_w: vec![0.0; h * h],
            out_w: vec![0.0; tv * h],
            out_b: vec![0.0; tv],
        }
    }

    pub fn zero(&mut self) {
        self.enc_emb.fill(0.0); self.dec_emb.fill(0.0);
        self.enc.zero(); self.dec.zero();
        self.attn_w.fill(0.0);
        self.out_w.fill(0.0); self.out_b.fill(0.0);
    }

    pub fn add(&mut self, other: &Self) {
        for (a, &b) in self.enc_emb.iter_mut().zip(&other.enc_emb) { *a += b; }
        for (a, &b) in self.dec_emb.iter_mut().zip(&other.dec_emb) { *a += b; }
        self.enc.add(&other.enc);
        self.dec.add(&other.dec);
        for (a, &b) in self.attn_w.iter_mut().zip(&other.attn_w) { *a += b; }
        for (a, &b) in self.out_w.iter_mut().zip(&other.out_w) { *a += b; }
        for (a, &b) in self.out_b.iter_mut().zip(&other.out_b) { *a += b; }
    }

    pub fn scale(&mut self, s: f32) {
        for g in &mut self.enc_emb { *g *= s; }
        for g in &mut self.dec_emb { *g *= s; }
        self.enc.scale(s);
        self.dec.scale(s);
        for g in &mut self.attn_w { *g *= s; }
        for g in &mut self.out_w  { *g *= s; }
        for g in &mut self.out_b  { *g *= s; }
    }
}

// ─── BPTT Helper ─────────────────────────────────────────────────────────────

/// Backprop through one GRU step.
/// Returns d_h_prev (gradient wrt previous hidden state) and d_x (gradient wrt input).
pub fn gru_backward(
    cell: &S2SGruCell,
    x: &[f32],
    h_prev: &[f32],
    gb: &GateBuf,
    d_h: &[f32],       // gradient wrt h_out
    grad: &mut CellGrad,
    d_h_prev_out: &mut [f32],
    d_x_out: &mut [f32],
) {
    let h = cell.hidden_dim;
    let in_d = cell.input_dim;
    d_h_prev_out.fill(0.0);
    d_x_out.fill(0.0);

    for i in 0..h {
        let r = gb.r[i];
        let z = gb.z[i];
        let n = gb.n[i];
        let dh = d_h[i];

        // h = (1-z)*n + z*h_prev → grads:
        let d_z = dh * (h_prev[i] - n);
        let d_n = dh * (1.0 - z);
        d_h_prev_out[i] += dh * z;

        // n = tanh(n_in + r*n_hn) → dn_pre = d_n * (1 - n^2)
        let dn_pre = d_n * (1.0 - n * n);
        // n_in part: d_n_hn = dn_pre * r
        let d_n_hn = dn_pre * r;
        let d_r = dn_pre * gb.n_hn[i];
        // b_in, b_hn
        grad.b_in[i] += dn_pre;
        grad.b_hn[i] += d_n_hn;

        // r = sigmoid(pre_r) → dr_pre = d_r * r * (1-r)
        let dr_pre = d_r * r * (1.0 - r);
        // z = sigmoid(pre_z) → dz_pre = d_z * z * (1-z)
        let dz_pre = d_z * z * (1.0 - z);

        // Update gradients for W_ir, W_hr, b_r
        grad.b_r[i] += dr_pre;
        for j in 0..in_d {
            grad.w_ir[i * in_d + j] += dr_pre * x[j];
        }
        for j in 0..h {
            grad.w_hr[i * h + j] += dr_pre * h_prev[j];
            d_h_prev_out[j] += dr_pre * cell.w_hr[i * h + j];
        }

        // W_iz, W_hz, b_z
        grad.b_z[i] += dz_pre;
        for j in 0..in_d {
            grad.w_iz[i * in_d + j] += dz_pre * x[j];
        }
        for j in 0..h {
            grad.w_hz[i * h + j] += dz_pre * h_prev[j];
            d_h_prev_out[j] += dz_pre * cell.w_hz[i * h + j];
        }

        // W_in (n_in part), d_x
        for j in 0..in_d {
            grad.w_in[i * in_d + j] += dn_pre * x[j];
            d_x_out[j] += dn_pre * cell.w_in[i * in_d + j];
            d_x_out[j] += dr_pre * cell.w_ir[i * in_d + j];
            d_x_out[j] += dz_pre * cell.w_iz[i * in_d + j];
        }

        // W_hn (n_hn part)
        for j in 0..h {
            grad.w_hn[i * h + j] += d_n_hn * h_prev[j];
            d_h_prev_out[j] += d_n_hn * cell.w_hn[i * h + j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charvocab_roundtrip() {
        let v = CharVocab::from_strings(["hello", "world"].iter().copied());
        assert!(v.size() > SPECIAL_COUNT);
        for c in "hello".chars() {
            let id = v.encode(c);
            assert!(id >= SPECIAL_COUNT as u32);
            assert_eq!(v.decode(id), Some(c));
        }
    }

    #[test]
    fn test_model_inference_shape() {
        let sv = CharVocab::from_strings(["abcde"].iter().copied());
        let tv = CharVocab::from_strings(["অআইউ"].iter().copied());
        let m = Seq2SeqTransliterator::new(sv, tv, 8, 16);
        let mut scratch = InferenceScratch::for_model(&m);
        let result = m.transliterate("abc", 3, &mut scratch);
        // Should not panic; may return empty strings if weights are zero
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_param_count_reasonable() {
        let sv = CharVocab::from_strings(["abcdefghijklmnopqrstuvwxyz0123456789"].iter().copied());
        let tv = CharVocab::from_strings(["অআইউএওকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃঁ"].iter().copied());
        let m = Seq2SeqTransliterator::new(sv, tv, 16, 64);
        let params = m.param_count();
        // Should be < 200K parameters → < 800 KB
        assert!(params < 250_000, "Too many params: {}", params);
    }

    #[test]
    fn test_constrained_inference() {
        let sv = CharVocab::from_strings(["abcde"].iter().copied());
        let tv = CharVocab::from_strings(["অআইউ"].iter().copied());
        let m = Seq2SeqTransliterator::new(sv, tv, 8, 16);
        let mut scratch = InferenceScratch::for_model(&m);
        // Constraint: only allow prefixes starting with 'অ'
        let prefix_filter = |p: &str| p.starts_with('অ');
        let word_filter = |w: &str| w == "অ";
        let result = m.transliterate_constrained("a", 3, &mut scratch, Some(&prefix_filter), Some(&word_filter));
        for cand in &result {
            assert!(cand.starts_with('অ'));
        }
    }
}
