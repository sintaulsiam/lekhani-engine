#![allow(clippy::all)]
//! Train a character-level GRU seq2seq transliterator on the Dakshina dataset.
//!
//! Usage:
//!   cargo run --release --bin train_seq2seq -- \
//!       --train data/eval/bn.translit.sampled.train.tsv \
//!       --out   data/dictionaries/seq2seq_transliterator.bin \
//!       [--emb 16] [--hidden 64] [--epochs 30] [--lr 3e-3] [--beam 5]
//!
//! Model size at emb=16, hidden=64: ~180 KB.
//! Training time on 88K pairs × 30 epochs: ~4 minutes.

use lekhani_neural::seq2seq::{
    gru_backward, quantize_weights, CellGrad, CharVocab, GateBuf, InferenceScratch,
    QuantizedS2SGruCell, S2SGruCell, Seq2SeqGrad, Seq2SeqQuantized, Seq2SeqTransliterator, BOS_ID,
};
use rayon::prelude::*;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Config {
    train_path: PathBuf,
    out_path: PathBuf,
    emb_dim: usize,
    hidden_dim: usize,
    epochs: usize,
    lr: f32,
    lr_decay: f32,
    batch: usize,
    beam: usize,
    eval_n: usize,
    augment: bool,
    quantize: bool,
    bidirectional: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            train_path: PathBuf::from("data/eval/bn.translit.sampled.train.tsv"),
            out_path: PathBuf::from("data/dictionaries/seq2seq_transliterator.bin"),
            emb_dim: 16,
            hidden_dim: 64,
            epochs: 30,
            lr: 3e-3,
            lr_decay: 0.92,
            batch: 128,
            beam: 5,
            eval_n: 200,
            augment: false,
            quantize: false,
            bidirectional: false,
        }
    }
}

fn parse_args() -> Config {
    let mut c = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--train" => {
                c.train_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                c.out_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--emb" => {
                c.emb_dim = args[i + 1].parse().unwrap_or(c.emb_dim);
                i += 2;
            }
            "--hidden" => {
                c.hidden_dim = args[i + 1].parse().unwrap_or(c.hidden_dim);
                i += 2;
            }
            "--epochs" => {
                c.epochs = args[i + 1].parse().unwrap_or(c.epochs);
                i += 2;
            }
            "--lr" => {
                c.lr = args[i + 1].parse().unwrap_or(c.lr);
                i += 2;
            }
            "--batch" => {
                c.batch = args[i + 1].parse().unwrap_or(c.batch);
                i += 2;
            }
            "--beam" => {
                c.beam = args[i + 1].parse().unwrap_or(c.beam);
                i += 2;
            }
            "--augment" => {
                c.augment = true;
                i += 1;
            }
            "--quantize" => {
                c.quantize = true;
                i += 1;
            }
            "--bidirectional" => {
                c.bidirectional = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    c
}

// ─── Data Loading ─────────────────────────────────────────────────────────────

fn load_tsv(path: &Path) -> Vec<(String, String)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("❌ Cannot open {:?}: {}", path, e);
            std::process::exit(1);
        }
    };
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for line in reader.lines().flatten() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 {
            continue;
        }
        let bn = lekhani_core::chars::canonicalize(parts[0].trim());
        let lat = parts[1].trim().to_ascii_lowercase();
        if bn.is_empty() || lat.is_empty() {
            continue;
        }
        let count: usize = parts
            .get(2)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1);
        let reps = if count >= 10 {
            3
        } else if count >= 3 {
            2
        } else {
            1
        };
        for _ in 0..reps {
            pairs.push((lat.clone(), bn.to_string()));
        }
    }
    pairs
}

// ─── Kaiming Initialization ───────────────────────────────────────────────────

fn kaiming_init(weights: &mut [f32], fan_in: usize, rng: &mut u64) {
    let std = (2.0 / fan_in as f32).sqrt();
    for w in weights.iter_mut() {
        *w = randn(rng) * std;
    }
}

fn randn(rng: &mut u64) -> f32 {
    // Box-Muller via xorshift64
    let u1 = xorshift64(rng) as f32 / u64::MAX as f32 + 1e-10;
    let u2 = xorshift64(rng) as f32 / u64::MAX as f32;
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn init_model(m: &mut Seq2SeqTransliterator, rng: &mut u64) {
    let ed = m.emb_dim;
    let hd = m.hidden_dim;
    let in_d = ed + hd;
    // Embeddings: small uniform init
    let emb_std = 0.05;
    for w in m.enc_emb.iter_mut() {
        *w = randn(rng) * emb_std;
    }
    for w in m.dec_emb.iter_mut() {
        *w = randn(rng) * emb_std;
    }
    // Encoder GRU
    kaiming_init(&mut m.encoder.w_ir, ed, rng);
    kaiming_init(&mut m.encoder.w_iz, ed, rng);
    kaiming_init(&mut m.encoder.w_in, ed, rng);
    kaiming_init(&mut m.encoder.w_hr, hd, rng);
    kaiming_init(&mut m.encoder.w_hz, hd, rng);
    kaiming_init(&mut m.encoder.w_hn, hd, rng);
    // Decoder GRU
    kaiming_init(&mut m.decoder.w_ir, in_d, rng);
    kaiming_init(&mut m.decoder.w_iz, in_d, rng);
    kaiming_init(&mut m.decoder.w_in, in_d, rng);
    kaiming_init(&mut m.decoder.w_hr, hd, rng);
    kaiming_init(&mut m.decoder.w_hz, hd, rng);
    kaiming_init(&mut m.decoder.w_hn, hd, rng);
    // Attention
    kaiming_init(&mut m.attn_w, hd, rng);
    // Output
    kaiming_init(&mut m.out_w, hd, rng);
}

// ─── Adam Optimizer ───────────────────────────────────────────────────────────

struct Adam {
    m: Vec<f32>, // first moment
    v: Vec<f32>, // second moment
    t: usize,
    offset: usize,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            offset: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn begin_step(&mut self) {
        self.t += 1;
        self.offset = 0;
    }
}

fn init_cell(cell: &mut S2SGruCell, in_dim: usize, hd: usize, rng: &mut u64) {
    kaiming_init(&mut cell.w_ir, in_dim, rng);
    kaiming_init(&mut cell.w_iz, in_dim, rng);
    kaiming_init(&mut cell.w_in, in_dim, rng);
    kaiming_init(&mut cell.w_hr, hd, rng);
    kaiming_init(&mut cell.w_hz, hd, rng);
    kaiming_init(&mut cell.w_hn, hd, rng);
}

fn scatter_grads(
    m: &mut Seq2SeqTransliterator,
    rev: Option<&mut S2SGruCell>,
    g: &Seq2SeqGrad,
    g_rev: Option<&CellGrad>,
    adam: &mut Adam,
    lr: f32,
) {
    adam.begin_step();
    let clip = 5.0f32;
    apply_adam(&mut m.enc_emb, &g.enc_emb, adam, lr, clip);
    apply_adam(&mut m.dec_emb, &g.dec_emb, adam, lr, clip);
    apply_cell_adam(&mut m.encoder, &g.enc, adam, lr, clip);
    if let (Some(r), Some(gr)) = (rev, g_rev) {
        apply_cell_adam(r, gr, adam, lr, clip);
    }
    apply_cell_adam(&mut m.decoder, &g.dec, adam, lr, clip);
    apply_adam(&mut m.attn_w, &g.attn_w, adam, lr, clip);
    apply_adam(&mut m.out_w, &g.out_w, adam, lr, clip);
    apply_adam(&mut m.out_b, &g.out_b, adam, lr, clip);
}

fn apply_adam(params: &mut [f32], grads: &[f32], adam: &mut Adam, lr: f32, clip: f32) {
    let bc1 = 1.0 - adam.beta1.powi(adam.t as i32);
    let bc2 = 1.0 - adam.beta2.powi(adam.t as i32);
    let lr_hat = lr * bc2.sqrt() / bc1;
    let n = params.len().min(grads.len());
    let offset = adam.offset;
    for i in 0..n {
        let p_idx = offset + i;
        if p_idx >= adam.m.len() {
            break;
        }
        let g = grads[i].clamp(-clip, clip);
        adam.m[p_idx] = adam.beta1 * adam.m[p_idx] + (1.0 - adam.beta1) * g;
        adam.v[p_idx] = adam.beta2 * adam.v[p_idx] + (1.0 - adam.beta2) * g * g;
        params[i] -= lr_hat * adam.m[p_idx] / (adam.v[p_idx].sqrt() + adam.eps);
    }
    adam.offset += n;
}

fn apply_cell_adam(cell: &mut S2SGruCell, cg: &CellGrad, adam: &mut Adam, lr: f32, clip: f32) {
    apply_adam(&mut cell.w_ir, &cg.w_ir, adam, lr, clip);
    apply_adam(&mut cell.w_hr, &cg.w_hr, adam, lr, clip);
    apply_adam(&mut cell.b_r, &cg.b_r, adam, lr, clip);
    apply_adam(&mut cell.w_iz, &cg.w_iz, adam, lr, clip);
    apply_adam(&mut cell.w_hz, &cg.w_hz, adam, lr, clip);
    apply_adam(&mut cell.b_z, &cg.b_z, adam, lr, clip);
    apply_adam(&mut cell.w_in, &cg.w_in, adam, lr, clip);
    apply_adam(&mut cell.w_hn, &cg.w_hn, adam, lr, clip);
    apply_adam(&mut cell.b_in, &cg.b_in, adam, lr, clip);
    apply_adam(&mut cell.b_hn, &cg.b_hn, adam, lr, clip);
}

// ─── Forward + Backward (single example) ─────────────────────────────────────

struct FwdBufEnc {
    h_states: Vec<Vec<f32>>, // per step hidden state
    x_embeds: Vec<Vec<f32>>, // per step input embedding
    gates: Vec<GateBuf>,
}

struct FwdBufDec {
    h_states: Vec<Vec<f32>>,
    x_dec_in: Vec<Vec<f32>>, // [emb || ctx] per step
    ctx_vecs: Vec<Vec<f32>>, // context vector per step
    tgt_embs: Vec<Vec<f32>>, // target embeddings per step
    gates: Vec<GateBuf>,
    logits: Vec<Vec<f32>>,
    attn_scores: Vec<Vec<f32>>, // softmax attention per step
}

/// Forward pass: returns (cross_entropy_loss, enc_buf, rev_enc_buf, dec_buf)
fn forward(
    m: &Seq2SeqTransliterator,
    rev: Option<&S2SGruCell>,
    src: &[u32],
    tgt: &[u32], // includes BOS at start, EOS at end
) -> (f32, FwdBufEnc, Option<FwdBufEnc>, FwdBufDec) {
    let hd = m.hidden_dim;
    let ed = m.emb_dim;
    let tv = m.tgt_vocab.size();
    let in_d = ed + hd;

    // ── Encoder (Forward) ────────────────────────────────────────────────────
    let mut enc = FwdBufEnc {
        h_states: Vec::with_capacity(src.len() + 1),
        x_embeds: Vec::with_capacity(src.len()),
        gates: Vec::with_capacity(src.len()),
    };
    enc.h_states.push(vec![0.0; hd]); // h_0
    for &sid in src {
        let sid = (sid as usize).min(m.src_vocab.size().saturating_sub(1));
        let s = sid * ed;
        let emb = m.enc_emb[s..s + ed].to_vec();
        let h_prev = enc.h_states.last().unwrap().clone();
        let mut h_next = vec![0.0; hd];
        let mut gb = GateBuf::new(hd);
        m.encoder.step(&emb, &h_prev, &mut h_next, Some(&mut gb));
        enc.x_embeds.push(emb);
        enc.h_states.push(h_next);
        enc.gates.push(gb);
    }

    // ── Encoder (Optional Backward for BiGRU) ────────────────────────────────
    let enc_rev = rev.map(|rev_cell| {
        let mut eb = FwdBufEnc {
            h_states: Vec::with_capacity(src.len() + 1),
            x_embeds: Vec::with_capacity(src.len()),
            gates: Vec::with_capacity(src.len()),
        };
        eb.h_states.push(vec![0.0; hd]);
        for &sid in src.iter().rev() {
            let sid = (sid as usize).min(m.src_vocab.size().saturating_sub(1));
            let s = sid * ed;
            let emb = m.enc_emb[s..s + ed].to_vec();
            let h_prev = eb.h_states.last().unwrap().clone();
            let mut h_next = vec![0.0; hd];
            let mut gb = GateBuf::new(hd);
            rev_cell.step(&emb, &h_prev, &mut h_next, Some(&mut gb));
            eb.x_embeds.push(emb);
            eb.h_states.push(h_next);
            eb.gates.push(gb);
        }
        eb
    });

    // Combined encoder states
    let enc_len_raw = src.len();
    let enc_states: Vec<Vec<f32>> = if let Some(ref er) = enc_rev {
        (0..enc_len_raw)
            .map(|k| {
                let fwd = &enc.h_states[k + 1];
                let rev_s = &er.h_states[enc_len_raw - k];
                (0..hd).map(|j| 0.5 * (fwd[j] + rev_s[j])).collect()
            })
            .collect()
    } else {
        enc.h_states[1..].to_vec()
    };

    // ── Decoder (teacher forcing) ──────────────────────────────────────────
    // tgt = [BOS, y1, y2, ..., EOS]
    // At step t: input = tgt[t], predict tgt[t+1]
    let dec_steps = tgt.len().saturating_sub(1); // BOS..last predict EOS
    let mut dec = FwdBufDec {
        h_states: Vec::with_capacity(dec_steps + 1),
        x_dec_in: Vec::with_capacity(dec_steps),
        ctx_vecs: Vec::with_capacity(dec_steps),
        tgt_embs: Vec::with_capacity(dec_steps),
        gates: Vec::with_capacity(dec_steps),
        logits: Vec::with_capacity(dec_steps),
        attn_scores: Vec::with_capacity(dec_steps),
    };
    // init dec hidden = final encoder hidden
    dec.h_states
        .push(enc_states.last().cloned().unwrap_or_else(|| vec![0.0; hd]));

    let mut loss = 0.0f32;
    let max_dec_len = (src.len() + 1).min(14).max(1);
    let is_bi = rev.is_some();

    for t in 0..dec_steps {
        let inp_id = tgt[t] as usize;
        let s = inp_id.min(m.tgt_vocab.size().saturating_sub(1)) * ed;
        let tgt_emb = m.dec_emb[s..s + ed].to_vec();

        let h_prev = dec.h_states.last().unwrap();

        // Attention: q = attn_w · h_prev
        let mut q = vec![0.0f32; hd];
        for i in 0..hd {
            for j in 0..hd {
                q[i] += m.attn_w[i * hd + j] * h_prev[j];
            }
        }
        // Scores (with monotonic Gaussian bias when bidirectional)
        let enc_len = enc_states.len().max(1);
        let mut attn_raw = vec![0.0f32; enc_len];
        if is_bi {
            let expected_pos = (t as f32 * (enc_len.saturating_sub(1)) as f32) / max_dec_len as f32;
            let sigma_sq = 2.0 * 2.5 * 2.5;
            for (k, eh) in enc_states.iter().enumerate() {
                let dist = k as f32 - expected_pos;
                let penalty = -(dist * dist) / sigma_sq;
                attn_raw[k] = eh.iter().zip(q.iter()).map(|(&a, &b)| a * b).sum::<f32>() + penalty;
            }
        } else {
            for (k, eh) in enc_states.iter().enumerate() {
                attn_raw[k] = eh.iter().zip(q.iter()).map(|(&a, &b)| a * b).sum::<f32>();
            }
        }
        // Softmax
        let max_a = attn_raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut attn_sm = vec![0.0f32; enc_len];
        let mut sum_e = 0.0f32;
        for k in 0..enc_len {
            attn_sm[k] = (attn_raw[k] - max_a).exp();
            sum_e += attn_sm[k];
        }
        for a in attn_sm.iter_mut() {
            *a /= sum_e.max(1e-9);
        }

        // Context
        let mut ctx = vec![0.0f32; hd];
        for (k, eh) in enc_states.iter().enumerate() {
            let a = attn_sm[k];
            for j in 0..hd {
                ctx[j] += a * eh[j];
            }
        }

        // Decoder input = [tgt_emb || ctx]
        let mut dec_inp = vec![0.0f32; in_d];
        dec_inp[..ed].copy_from_slice(&tgt_emb);
        dec_inp[ed..].copy_from_slice(&ctx);

        let h_prev = dec.h_states.last().unwrap().clone();
        let mut h_next = vec![0.0; hd];
        let mut gb = GateBuf::new(hd);
        m.decoder
            .step(&dec_inp, &h_prev, &mut h_next, Some(&mut gb));

        // Logits
        let mut logits = vec![0.0f32; tv];
        for v in 0..tv {
            let mut s = m.out_b[v];
            for j in 0..hd {
                s += m.out_w[v * hd + j] * h_next[j];
            }
            logits[v] = s;
        }

        // Cross-entropy vs tgt[t+1]
        let gold = tgt[t + 1] as usize;
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        let log_sum = max_l + sum_exp.ln();
        loss += log_sum - logits[gold];

        dec.tgt_embs.push(tgt_emb);
        dec.ctx_vecs.push(ctx);
        dec.x_dec_in.push(dec_inp);
        dec.attn_scores.push(attn_sm);
        dec.h_states.push(h_next);
        dec.gates.push(gb);
        dec.logits.push(logits);
    }

    loss /= dec_steps.max(1) as f32;
    (loss, enc, enc_rev, dec)
}

/// Backward pass: accumulate gradients into `grad` (and optional `g_rev`)
fn backward(
    m: &Seq2SeqTransliterator,
    rev: Option<&S2SGruCell>,
    src: &[u32],
    tgt: &[u32],
    enc: &FwdBufEnc,
    enc_rev: Option<&FwdBufEnc>,
    dec: &FwdBufDec,
    grad: &mut Seq2SeqGrad,
    mut g_rev: Option<&mut CellGrad>,
) {
    let hd = m.hidden_dim;
    let ed = m.emb_dim;
    let tv = m.tgt_vocab.size();
    let in_d = ed + hd;
    let dec_steps = dec.h_states.len() - 1;
    let enc_len = enc.h_states.len() - 1; // number of encoder steps (excl. h_0)
    let scale = 1.0 / dec_steps.max(1) as f32;

    // d_h accumulates gradient flowing into decoder h
    let mut d_dec_h = vec![0.0f32; hd];
    // Gradients flowing back through encoder from attention
    let mut d_enc_h: Vec<Vec<f32>> = vec![vec![0.0; hd]; enc_len];

    let mut d_x_buf = vec![0.0f32; in_d];
    let mut d_h_prev_buf = vec![0.0f32; hd];

    for t in (0..dec_steps).rev() {
        let gold = tgt[t + 1] as usize;
        // Softmax gradient
        let max_l = dec.logits[t]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = dec.logits[t].iter().map(|&l| (l - max_l).exp()).sum();
        let mut d_logits = vec![0.0f32; tv];
        for v in 0..tv {
            let prob = (dec.logits[t][v] - max_l).exp() / sum_exp;
            d_logits[v] = (prob - if v == gold { 1.0 } else { 0.0 }) * scale;
        }

        // Output projection gradients: d_h from out_w^T · d_logits
        let h_dec = &dec.h_states[t + 1];
        let mut d_h_out = vec![0.0f32; hd];
        for v in 0..tv {
            let dl = d_logits[v];
            grad.out_b[v] += dl;
            for j in 0..hd {
                grad.out_w[v * hd + j] += dl * h_dec[j];
                d_h_out[j] += dl * m.out_w[v * hd + j];
            }
        }
        // Add accumulated d_dec_h
        for j in 0..hd {
            d_h_out[j] += d_dec_h[j];
        }

        // Decoder GRU backward
        let h_prev = &dec.h_states[t];
        gru_backward(
            &m.decoder,
            &dec.x_dec_in[t],
            h_prev,
            &dec.gates[t],
            &d_h_out,
            &mut grad.dec,
            &mut d_h_prev_buf,
            &mut d_x_buf,
        );
        d_dec_h.copy_from_slice(&d_h_prev_buf);

        // d_x_buf = [d_emb || d_ctx]
        // Gradient for target embedding
        let inp_id = tgt[t] as usize;
        let emb_s = inp_id.min(m.tgt_vocab.size().saturating_sub(1)) * ed;
        for j in 0..ed {
            grad.dec_emb[emb_s + j] += d_x_buf[j];
        }

        // Gradient for context vector d_ctx = d_x_buf[ed..]
        let d_ctx = &d_x_buf[ed..];

        // Backprop through context = sum(a_k * enc_h_k)
        // d_enc_h_k += a_k * d_ctx
        let attn = &dec.attn_scores[t];
        for k in 0..enc_len.min(attn.len()) {
            let a = attn[k];
            for j in 0..hd {
                d_enc_h[k][j] += a * d_ctx[j];
            }
        }

        // Backprop through attention (simplified: just backprop to attn_w via q)
        // d_q = sum_k(a_k * (d_ctx · enc_h_k)) -- simplified, skip full attn bprop
        // For attn_w: d_attn_w += h_prev^T * d_q (approximation: just backprop through linear)
        let h_prev_dec = &dec.h_states[t];
        // Compute d_q from d_ctx and attention
        let mut d_q = vec![0.0f32; hd];
        for k in 0..enc_len.min(attn.len()) {
            for j in 0..hd {
                d_q[j] += attn[k] * d_ctx[j] * enc.h_states[k + 1][j]; // approx
            }
        }
        for i in 0..hd {
            for j in 0..hd {
                grad.attn_w[i * hd + j] += d_q[i] * h_prev_dec[j];
            }
        }
    }

    // Backprop through encoder
    // d_enc_h also gets d_dec_h[0] (decoder init from final encoder state)
    if enc_len > 0 {
        for j in 0..hd {
            d_enc_h[enc_len - 1][j] += d_dec_h[j];
        }
    }

    let is_bi = rev.is_some() && enc_rev.is_some();
    let enc_scale = if is_bi { 0.5 } else { 1.0 };

    let mut d_enc_h_in = vec![0.0f32; hd];
    let mut d_enc_x = vec![0.0f32; ed];
    for t in (0..enc_len).rev() {
        let d_h_cur = &d_enc_h[t];
        let mut d_h_with_carry: Vec<f32> = d_h_cur.iter().map(|&v| v * enc_scale).collect();
        for j in 0..hd {
            d_h_with_carry[j] += d_enc_h_in[j];
        }

        gru_backward(
            &m.encoder,
            &enc.x_embeds[t],
            &enc.h_states[t],
            &enc.gates[t],
            &d_h_with_carry,
            &mut grad.enc,
            &mut d_enc_h_in,
            &mut d_enc_x,
        );

        let sid = src[t] as usize;
        let emb_s = sid.min(m.src_vocab.size().saturating_sub(1)) * ed;
        for j in 0..ed {
            grad.enc_emb[emb_s + j] += d_enc_x[j];
        }
    }

    if let (Some(rev_cell), Some(er), Some(gr)) = (rev, enc_rev, g_rev.as_deref_mut()) {
        let mut d_rev_h_in = vec![0.0f32; hd];
        let mut d_rev_x = vec![0.0f32; ed];
        for t in (0..enc_len).rev() {
            // rev step t processed src[enc_len - 1 - t]
            let orig_k = enc_len - 1 - t;
            let mut d_h_with_carry: Vec<f32> = d_enc_h[orig_k].iter().map(|&v| v * 0.5).collect();
            for j in 0..hd {
                d_h_with_carry[j] += d_rev_h_in[j];
            }
            gru_backward(
                rev_cell,
                &er.x_embeds[t],
                &er.h_states[t],
                &er.gates[t],
                &d_h_with_carry,
                gr,
                &mut d_rev_h_in,
                &mut d_rev_x,
            );
            let sid = src[orig_k] as usize;
            let emb_s = sid.min(m.src_vocab.size().saturating_sub(1)) * ed;
            for j in 0..ed {
                grad.enc_emb[emb_s + j] += d_rev_x[j];
            }
        }
    }
}

fn build_quantized(m: &Seq2SeqTransliterator, rev: Option<&S2SGruCell>) -> Seq2SeqQuantized {
    let (enc_emb, scale_enc_emb) = quantize_weights(&m.enc_emb);
    let (dec_emb, scale_dec_emb) = quantize_weights(&m.dec_emb);
    let (attn_w, scale_attn_w) = quantize_weights(&m.attn_w);
    let (out_w, scale_out_w) = quantize_weights(&m.out_w);
    Seq2SeqQuantized {
        src_vocab: m.src_vocab.clone(),
        tgt_vocab: m.tgt_vocab.clone(),
        emb_dim: m.emb_dim,
        hidden_dim: m.hidden_dim,
        enc_emb,
        scale_enc_emb,
        dec_emb,
        scale_dec_emb,
        encoder: QuantizedS2SGruCell::from_f32(&m.encoder),
        encoder_rev: rev.map(QuantizedS2SGruCell::from_f32),
        is_bidirectional: rev.is_some(),
        decoder: QuantizedS2SGruCell::from_f32(&m.decoder),
        attn_w,
        scale_attn_w,
        out_w,
        scale_out_w,
        out_b: m.out_b.clone(),
    }
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

fn evaluate(
    m: &Seq2SeqTransliterator,
    rev: Option<&S2SGruCell>,
    pairs: &[(String, String)],
    n: usize,
    beam: usize,
) -> f32 {
    let eval_pairs: Vec<_> = pairs.iter().take(n).collect();
    let mut hits = 0usize;
    if rev.is_some() {
        let q = build_quantized(m, rev);
        let mut scratch = InferenceScratch::for_quantized(&q);
        for (lat, bn) in &eval_pairs {
            let cands = q.transliterate(lat, beam, &mut scratch);
            if cands.first().map(|s| s == bn).unwrap_or(false) {
                hits += 1;
            }
        }
    } else {
        let mut scratch = InferenceScratch::for_model(m);
        for (lat, bn) in &eval_pairs {
            let cands = m.transliterate(lat, beam, &mut scratch);
            if cands.first().map(|s| s == bn).unwrap_or(false) {
                hits += 1;
            }
        }
    }
    hits as f32 / eval_pairs.len().max(1) as f32
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn augment_pairs(pairs: &[(String, String)], rng: &mut u64) -> Vec<(String, String)> {
    let mut out = Vec::with_capacity(pairs.len() * 2);
    for (lat, bn) in pairs {
        out.push((lat.clone(), bn.clone()));
        if lat.len() < 4 {
            continue;
        }
        let r = xorshift64(rng) % 10;
        if r < 2 {
            // Synthetic vowel drop (e.g. "kothay" -> "kothy", "amader" -> "amder")
            let chars: Vec<char> = lat.chars().collect();
            if let Some(pos) =
                (1..chars.len() - 1).find(|&idx| matches!(chars[idx], 'a' | 'e' | 'i' | 'o' | 'u'))
            {
                let mut mutated = String::with_capacity(lat.len());
                for (i, ch) in chars.iter().enumerate() {
                    if i != pos {
                        mutated.push(*ch);
                    }
                }
                if mutated.len() >= 3 {
                    out.push((mutated, bn.clone()));
                }
            }
        } else if r < 4 {
            // Digraph / phonetic variant (sh->s, kh->k, bh->v, ph->f, chh->ch)
            let mutated = if lat.contains("sh") {
                Some(lat.replacen("sh", "s", 1))
            } else if lat.contains("kh") {
                Some(lat.replacen("kh", "k", 1))
            } else if lat.contains("bh") {
                Some(lat.replacen("bh", "v", 1))
            } else if lat.contains("ph") {
                Some(lat.replacen("ph", "f", 1))
            } else if lat.contains("chh") {
                Some(lat.replacen("chh", "ch", 1))
            } else {
                None
            };
            if let Some(m) = mutated {
                if m != *lat && m.len() >= 3 {
                    out.push((m, bn.clone()));
                }
            }
        } else if r == 4 {
            // Adjacent character transposition typo
            let mut chars: Vec<char> = lat.chars().collect();
            if chars.len() >= 4 {
                let idx = 1 + (xorshift64(rng) as usize % (chars.len() - 2));
                chars.swap(idx, idx + 1);
                let mutated: String = chars.into_iter().collect();
                if mutated != *lat {
                    out.push((mutated, bn.clone()));
                }
            }
        } else if r == 5 && lat.starts_with('o') && lat.len() >= 4 {
            // Dakshina natural inherent-'অ' written as 'a' (e.g. odhikarta -> adhikarta)
            out.push((format!("a{}", &lat[1..]), bn.clone()));
        } else if r == 6 && lat.contains("sh") {
            out.push((lat.replacen("sh", "s", 1), bn.clone()));
        }
    }
    out
}

fn main() {
    let cfg = parse_args();

    println!("Loading training data from {:?}...", cfg.train_path);
    let base_pairs = load_tsv(&cfg.train_path);
    let mut aug_rng = 0x_cafe_babe_1234_u64;
    let pairs = if cfg.augment {
        let augmented = augment_pairs(&base_pairs, &mut aug_rng);
        println!(
            "   {} base pairs -> {} augmented pairs.",
            base_pairs.len(),
            augmented.len()
        );
        augmented
    } else {
        base_pairs
    };
    println!("   {} unique (latin, bengali) pairs loaded.", pairs.len());
    if pairs.is_empty() {
        eprintln!("❌ No training pairs loaded. Check TSV path.");
        std::process::exit(1);
    }

    // Build vocabularies
    let src_vocab = CharVocab::from_strings(pairs.iter().map(|(l, _)| l.as_str()));
    let tgt_vocab = CharVocab::from_strings(pairs.iter().map(|(_, b)| b.as_str()));
    println!(
        "   Latin vocab:   {} chars (+4 special = {})",
        src_vocab.chars.len(),
        src_vocab.size()
    );
    println!(
        "   Bengali vocab: {} chars (+4 special = {})",
        tgt_vocab.chars.len(),
        tgt_vocab.size()
    );

    // Build model
    let mut model = Seq2SeqTransliterator::new(src_vocab, tgt_vocab, cfg.emb_dim, cfg.hidden_dim);
    let mut encoder_rev = if cfg.bidirectional {
        Some(S2SGruCell::new_zeros(cfg.emb_dim, cfg.hidden_dim))
    } else {
        None
    };

    let mut rng = 0x_dead_beef_cafe_u64;
    init_model(&mut model, &mut rng);
    if let Some(ref mut rev) = encoder_rev {
        init_cell(rev, cfg.emb_dim, cfg.hidden_dim, &mut rng);
    }

    let rev_params = if encoder_rev.is_some() {
        3 * cfg.hidden_dim * cfg.emb_dim + 3 * cfg.hidden_dim * cfg.hidden_dim + 4 * cfg.hidden_dim
    } else {
        0
    };
    let n_params = model.param_count() + rev_params;
    println!(
        "   Parameters: {} (bidirectional={}, ~{:.1} KB F32)",
        n_params,
        cfg.bidirectional,
        n_params as f32 * 4.0 / 1024.0
    );

    // Adam state — size = total params
    let mut adam = Adam::new(n_params);

    // Pre-compute encoded pairs
    let mut src_seqs: Vec<Vec<u32>> = Vec::with_capacity(pairs.len());
    let mut tgt_seqs: Vec<Vec<u32>> = Vec::with_capacity(pairs.len());
    for (lat, bn) in &pairs {
        src_seqs.push(model.src_vocab.encode_str(lat));
        let mut tseq = vec![BOS_ID];
        tseq.extend(model.tgt_vocab.encode_str_with_eos(bn));
        tgt_seqs.push(tseq);
    }

    // Skip pairs longer than 25 chars (avoid extreme BPTT)
    let max_len = 25usize;
    let valid: Vec<usize> = (0..pairs.len())
        .filter(|&i| src_seqs[i].len() <= max_len && tgt_seqs[i].len() <= max_len + 2)
        .collect();
    println!(
        "   Training on {} pairs (max_len={}).",
        valid.len(),
        max_len
    );

    let mut lr = cfg.lr;

    for epoch in 0..cfg.epochs {
        // Shuffle
        let mut order = valid.clone();
        let mut r = rng;
        for i in (1..order.len()).rev() {
            let j = (xorshift64(&mut r) as usize) % (i + 1);
            order.swap(i, j);
        }
        rng = r;

        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        let batch = cfg.batch;
        for chunk in order.chunks(batch) {
            let (batch_loss, batch_count, mut grad, mut grad_rev) = chunk
                .par_iter()
                .fold(
                    || {
                        (
                            0.0f32,
                            0usize,
                            Seq2SeqGrad::new_for(&model),
                            encoder_rev
                                .as_ref()
                                .map(|_| CellGrad::new(cfg.emb_dim, cfg.hidden_dim)),
                        )
                    },
                    |(mut loss_acc, mut count_acc, mut grad_acc, mut rev_acc), &idx| {
                        let (loss, enc, enc_rev_buf, dec) = forward(
                            &model,
                            encoder_rev.as_ref(),
                            src_seqs[idx].as_slice(),
                            tgt_seqs[idx].as_slice(),
                        );
                        if loss.is_finite() {
                            backward(
                                &model,
                                encoder_rev.as_ref(),
                                src_seqs[idx].as_slice(),
                                tgt_seqs[idx].as_slice(),
                                &enc,
                                enc_rev_buf.as_ref(),
                                &dec,
                                &mut grad_acc,
                                rev_acc.as_mut(),
                            );
                            loss_acc += loss;
                            count_acc += 1;
                        }
                        (loss_acc, count_acc, grad_acc, rev_acc)
                    },
                )
                .reduce(
                    || {
                        (
                            0.0f32,
                            0usize,
                            Seq2SeqGrad::new_for(&model),
                            encoder_rev
                                .as_ref()
                                .map(|_| CellGrad::new(cfg.emb_dim, cfg.hidden_dim)),
                        )
                    },
                    |(mut l1, c1, mut g1, mut r1), (l2, c2, g2, r2)| {
                        l1 += l2;
                        g1.add(&g2);
                        if let (Some(ref mut gr1), Some(ref gr2)) = (&mut r1, &r2) {
                            gr1.add(gr2);
                        }
                        (l1, c1 + c2, g1, r1)
                    },
                );

            total_loss += batch_loss;
            count += batch_count;

            // Scale gradients by batch size
            let bs = chunk.len().max(1) as f32;
            grad.scale(1.0 / bs);
            if let Some(ref mut gr) = grad_rev {
                gr.scale(1.0 / bs);
            }
            // apply Adam per param group
            scatter_grads(
                &mut model,
                encoder_rev.as_mut(),
                &grad,
                grad_rev.as_ref(),
                &mut adam,
                lr,
            );
        }

        let avg_loss = total_loss / count.max(1) as f32;

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            let acc = evaluate(&model, encoder_rev.as_ref(), &pairs, cfg.eval_n, cfg.beam);
            println!(
                "  Epoch {:2}/{}: loss={:.4}  train_top1@{}={:.2}%  lr={:.2e}",
                epoch + 1,
                cfg.epochs,
                avg_loss,
                cfg.eval_n,
                acc * 100.0,
                lr
            );
        } else {
            println!(
                "  Epoch {:2}/{}: loss={:.4}",
                epoch + 1,
                cfg.epochs,
                avg_loss
            );
        }

        lr *= cfg.lr_decay;
    }

    // Save
    println!("\n💾 Saving model to {:?}...", cfg.out_path);
    if let Some(parent) = cfg.out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match model.save(&cfg.out_path) {
        Ok(()) => {
            let size = std::fs::metadata(&cfg.out_path)
                .map(|m| m.len())
                .unwrap_or(0);
            println!("Saved ({:.1} KB)", size as f32 / 1024.0);
        }
        Err(e) => {
            eprintln!("Save failed: {}", e);
            std::process::exit(1);
        }
    }
    if cfg.quantize {
        let q_path = cfg
            .out_path
            .with_file_name("seq2seq_transliterator_int8.bin");
        let q = build_quantized(&model, encoder_rev.as_ref());
        if let Err(e) = q.save(&q_path) {
            eprintln!("Quantized save failed: {}", e);
            std::process::exit(1);
        }
        let q_size = std::fs::metadata(&q_path).map(|m| m.len()).unwrap_or(0);
        println!(
            "Saved quantized INT8 (bidirectional={}) to {:?} ({:.1} KB)",
            q.is_bidirectional,
            q_path,
            q_size as f32 / 1024.0
        );
    }
}
