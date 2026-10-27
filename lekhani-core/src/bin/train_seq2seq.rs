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
    gru_backward, CharVocab, CellGrad, GateBuf, InferenceScratch, S2SGruCell,
    Seq2SeqGrad, Seq2SeqTransliterator, BOS_ID,
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
        }
    }
}

fn parse_args() -> Config {
    let mut c = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--train" => { c.train_path = PathBuf::from(&args[i + 1]); i += 2; }
            "--out"   => { c.out_path = PathBuf::from(&args[i + 1]); i += 2; }
            "--emb"   => { c.emb_dim = args[i + 1].parse().unwrap_or(c.emb_dim); i += 2; }
            "--hidden" => { c.hidden_dim = args[i + 1].parse().unwrap_or(c.hidden_dim); i += 2; }
            "--epochs" => { c.epochs = args[i + 1].parse().unwrap_or(c.epochs); i += 2; }
            "--lr"    => { c.lr = args[i + 1].parse().unwrap_or(c.lr); i += 2; }
            "--batch" => { c.batch = args[i + 1].parse().unwrap_or(c.batch); i += 2; }
            "--beam"  => { c.beam = args[i + 1].parse().unwrap_or(c.beam); i += 2; }
            _ => { i += 1; }
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
        if parts.len() < 2 { continue; }
        let bn = lekhani_core::chars::canonicalize(parts[0].trim());
        let lat = parts[1].trim().to_ascii_lowercase();
        if bn.is_empty() || lat.is_empty() { continue; }
        // Deduplicate by keeping first occurrence (TSV may repeat pairs with counts)
        pairs.push((lat, bn.to_string()));
    }
    // Deduplicate
    pairs.sort_unstable();
    pairs.dedup();
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
    for w in m.enc_emb.iter_mut() { *w = randn(rng) * emb_std; }
    for w in m.dec_emb.iter_mut() { *w = randn(rng) * emb_std; }
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

fn scatter_grads(m: &mut Seq2SeqTransliterator, g: &Seq2SeqGrad, adam: &mut Adam, lr: f32) {
    adam.begin_step();
    let clip = 5.0f32;
    apply_adam(&mut m.enc_emb, &g.enc_emb, adam, lr, clip);
    apply_adam(&mut m.dec_emb, &g.dec_emb, adam, lr, clip);
    apply_cell_adam(&mut m.encoder, &g.enc, adam, lr, clip);
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
    apply_adam(&mut cell.b_r,  &cg.b_r,  adam, lr, clip);
    apply_adam(&mut cell.w_iz, &cg.w_iz, adam, lr, clip);
    apply_adam(&mut cell.w_hz, &cg.w_hz, adam, lr, clip);
    apply_adam(&mut cell.b_z,  &cg.b_z,  adam, lr, clip);
    apply_adam(&mut cell.w_in, &cg.w_in, adam, lr, clip);
    apply_adam(&mut cell.w_hn, &cg.w_hn, adam, lr, clip);
    apply_adam(&mut cell.b_in, &cg.b_in, adam, lr, clip);
    apply_adam(&mut cell.b_hn, &cg.b_hn, adam, lr, clip);
}

// ─── Forward + Backward (single example) ─────────────────────────────────────

struct FwdBufEnc {
    h_states: Vec<Vec<f32>>,    // per step hidden state
    x_embeds: Vec<Vec<f32>>,    // per step input embedding
    gates:    Vec<GateBuf>,
}

struct FwdBufDec {
    h_states: Vec<Vec<f32>>,
    x_dec_in: Vec<Vec<f32>>,    // [emb || ctx] per step
    ctx_vecs: Vec<Vec<f32>>,    // context vector per step
    tgt_embs: Vec<Vec<f32>>,    // target embeddings per step
    gates:    Vec<GateBuf>,
    logits:   Vec<Vec<f32>>,
    attn_scores: Vec<Vec<f32>>, // softmax attention per step
}

/// Forward pass: returns (cross_entropy_loss, enc_buf, dec_buf)
fn forward(
    m: &Seq2SeqTransliterator,
    src: &[u32],
    tgt: &[u32], // includes BOS at start, EOS at end
) -> (f32, FwdBufEnc, FwdBufDec) {
    let hd = m.hidden_dim;
    let ed = m.emb_dim;
    let tv = m.tgt_vocab.size();
    let in_d = ed + hd;

    // ── Encoder ──────────────────────────────────────────────────────────────
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

    // Encoder hidden states as key/value matrix (skip h_0)
    let enc_states: Vec<&Vec<f32>> = enc.h_states[1..].iter().collect();

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
    dec.h_states.push(enc.h_states.last().unwrap().clone());

    let mut loss = 0.0f32;

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
        // Scores
        let enc_len = enc_states.len().max(1);
        let mut attn_raw = vec![0.0f32; enc_len];
        for (k, eh) in enc_states.iter().enumerate() {
            attn_raw[k] = eh.iter().zip(q.iter()).map(|(&a, &b)| a * b).sum::<f32>();
        }
        // Softmax
        let max_a = attn_raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut attn_sm = vec![0.0f32; enc_len];
        let mut sum_e = 0.0f32;
        for k in 0..enc_len {
            attn_sm[k] = (attn_raw[k] - max_a).exp();
            sum_e += attn_sm[k];
        }
        for a in attn_sm.iter_mut() { *a /= sum_e.max(1e-9); }

        // Context
        let mut ctx = vec![0.0f32; hd];
        for (k, eh) in enc_states.iter().enumerate() {
            let a = attn_sm[k];
            for j in 0..hd { ctx[j] += a * eh[j]; }
        }

        // Decoder input = [tgt_emb || ctx]
        let mut dec_inp = vec![0.0f32; in_d];
        dec_inp[..ed].copy_from_slice(&tgt_emb);
        dec_inp[ed..].copy_from_slice(&ctx);

        let h_prev = dec.h_states.last().unwrap().clone();
        let mut h_next = vec![0.0; hd];
        let mut gb = GateBuf::new(hd);
        m.decoder.step(&dec_inp, &h_prev, &mut h_next, Some(&mut gb));

        // Logits
        let mut logits = vec![0.0f32; tv];
        for v in 0..tv {
            let mut s = m.out_b[v];
            for j in 0..hd { s += m.out_w[v * hd + j] * h_next[j]; }
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
    (loss, enc, dec)
}

/// Backward pass: accumulate gradients into `grad`
fn backward(
    m: &Seq2SeqTransliterator,
    src: &[u32],
    tgt: &[u32],
    enc: &FwdBufEnc,
    dec: &FwdBufDec,
    grad: &mut Seq2SeqGrad,
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
        let max_l = dec.logits[t].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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
        for j in 0..hd { d_h_out[j] += d_dec_h[j]; }

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
            for j in 0..hd { d_enc_h[k][j] += a * d_ctx[j]; }
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
        for j in 0..hd { d_enc_h[enc_len - 1][j] += d_dec_h[j]; }
    }

    let mut d_enc_h_in = vec![0.0f32; hd];
    let mut d_enc_x = vec![0.0f32; ed];
    for t in (0..enc_len).rev() {
        let d_h_cur = &d_enc_h[t];
        let mut d_h_with_carry = d_h_cur.to_vec();
        if t < enc_len - 1 {
            // d_enc_h_in carries grad from next step's h_prev
        }
        // Add carry from inner gru_backward's d_h_prev
        for j in 0..hd { d_h_with_carry[j] += d_enc_h_in[j]; }

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

        // Gradient for source embedding
        let sid = src[t] as usize;
        let emb_s = sid.min(m.src_vocab.size().saturating_sub(1)) * ed;
        for j in 0..ed {
            grad.enc_emb[emb_s + j] += d_enc_x[j];
        }
    }
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

fn evaluate(m: &Seq2SeqTransliterator, pairs: &[(String, String)], n: usize, beam: usize) -> f32 {
    let eval_pairs: Vec<_> = pairs.iter().take(n).collect();
    let mut hits = 0usize;
    let mut scratch = InferenceScratch::for_model(m);
    for (lat, bn) in &eval_pairs {
        let cands = m.transliterate(lat, beam, &mut scratch);
        if cands.first().map(|s| s == bn).unwrap_or(false) {
            hits += 1;
        }
    }
    hits as f32 / eval_pairs.len().max(1) as f32
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    println!("📚 Loading training data from {:?}...", cfg.train_path);
    let pairs = load_tsv(&cfg.train_path);
    println!("   {} unique (latin, bengali) pairs loaded.", pairs.len());
    if pairs.is_empty() {
        eprintln!("❌ No training pairs loaded. Check TSV path.");
        std::process::exit(1);
    }

    // Build vocabularies
    let src_vocab = CharVocab::from_strings(pairs.iter().map(|(l, _)| l.as_str()));
    let tgt_vocab = CharVocab::from_strings(pairs.iter().map(|(_, b)| b.as_str()));
    println!("   Latin vocab:   {} chars (+4 special = {})", src_vocab.chars.len(), src_vocab.size());
    println!("   Bengali vocab: {} chars (+4 special = {})", tgt_vocab.chars.len(), tgt_vocab.size());

    // Build model
    let mut model = Seq2SeqTransliterator::new(src_vocab, tgt_vocab, cfg.emb_dim, cfg.hidden_dim);
    println!("   Parameters: {} (~{:.1} KB)", model.param_count(), model.param_count() as f32 * 4.0 / 1024.0);

    let mut rng = 0x_dead_beef_cafe_u64;
    init_model(&mut model, &mut rng);

    // Adam state — size = total params
    let n_params = model.param_count();
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
    println!("   Training on {} pairs (max_len={}).", valid.len(), max_len);

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
            let (batch_loss, batch_count, mut grad) = chunk
                .par_iter()
                .fold(
                    || (0.0f32, 0usize, Seq2SeqGrad::new_for(&model)),
                    |(mut loss_acc, mut count_acc, mut grad_acc), &idx| {
                        let (loss, enc, dec) = forward(&model, &src_seqs[idx], &tgt_seqs[idx]);
                        if loss.is_finite() {
                            backward(&model, &src_seqs[idx], &tgt_seqs[idx], &enc, &dec, &mut grad_acc);
                            loss_acc += loss;
                            count_acc += 1;
                        }
                        (loss_acc, count_acc, grad_acc)
                    },
                )
                .reduce(
                    || (0.0f32, 0usize, Seq2SeqGrad::new_for(&model)),
                    |(mut l1, c1, mut g1), (l2, c2, g2)| {
                        l1 += l2;
                        g1.add(&g2);
                        (l1, c1 + c2, g1)
                    },
                );

            total_loss += batch_loss;
            count += batch_count;

            // Scale gradients by batch size
            let bs = chunk.len().max(1) as f32;
            grad.scale(1.0 / bs);
            // apply Adam per param group
            scatter_grads(&mut model, &grad, &mut adam, lr);
        }

        let avg_loss = total_loss / count.max(1) as f32;

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            let acc = evaluate(&model, &pairs, cfg.eval_n, cfg.beam);
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
            println!("  Epoch {:2}/{}: loss={:.4}", epoch + 1, cfg.epochs, avg_loss);
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
            let size = std::fs::metadata(&cfg.out_path).map(|m| m.len()).unwrap_or(0);
            println!("✅ Saved ({:.1} KB)", size as f32 / 1024.0);
        }
        Err(e) => {
            eprintln!("❌ Save failed: {}", e);
            std::process::exit(1);
        }
    }
}
