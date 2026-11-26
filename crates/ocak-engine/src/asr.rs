//! Native `Qwen3-ASR-1.7B` audio frontend (`qwen3a` mmproj + `qwen3vl` decoder):
//! - Zero-dependency PCM/float WAV parser + 16 kHz mono resampler
//! - Whisper-style 128-bin Slaney log-mel spectrogram (n_fft=400, hop=160, center reflection pad)
//! - 3x Conv2D(3x3, stride=2, pad=1) + GELU_ERF + conv_out + per-chunk pos_embd
//! - 24-layer Pre-LN Transformer encoder + `a.post_ln` + 2-layer GELU_ERF projector (`1024 -> 2048`)
//! - Feeds projected audio embeddings directly into `NativeEngine` (`VkExec`) for transcription.

use crate::engine::{BackendKind, EngineOptions, NativeEngine};
use crate::exec::EngineError;
use ocak_cpu::gemv;
use ocak_gguf::{GgufFile, TensorInfo};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

#[inline]
fn gelu_erf(x: f32) -> f32 {
    // Abramowitz & Stegun erf approximation (max err < 1.5e-7)
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-a * a).exp();
    0.5 * x * (1.0 + sign * y)
}

#[inline]
fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().enumerate().map(|(i, &v)| (v - mean) * inv * w[i] + b[i]).collect()
}

/// Parses a RIFF WAVE file (16-bit PCM or 32-bit float, mono/stereo) into 16 kHz mono f32 samples in [-1, 1].
pub fn parse_wav_mono_16k(data: &[u8]) -> Result<Vec<f32>, EngineError> {
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(EngineError::Unsupported("invalid WAV header (expected RIFF/WAVE)".into()));
    }
    let mut off = 12usize;
    let mut audio_fmt = 1u16;
    let mut channels = 1u16;
    let mut sample_rate = 16000u32;
    let mut bits_per_sample = 16u16;
    let mut pcm: Option<&[u8]> = None;

    while off + 8 <= data.len() {
        let id = &data[off..off + 4];
        let sz = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
        let body_start = off + 8;
        let body_end = (body_start + sz).min(data.len());
        if id == b"fmt " && body_end >= body_start + 16 {
            audio_fmt = u16::from_le_bytes(data[body_start..body_start + 2].try_into().unwrap());
            channels = u16::from_le_bytes(data[body_start + 2..body_start + 4].try_into().unwrap()).max(1);
            sample_rate = u32::from_le_bytes(data[body_start + 4..body_start + 8].try_into().unwrap()).max(1);
            bits_per_sample = u16::from_le_bytes(data[body_start + 14..body_start + 16].try_into().unwrap());
        } else if id == b"data" {
            pcm = Some(&data[body_start..body_end]);
            break;
        }
        off = body_start + sz + (sz & 1);
    }
    let pcm = pcm.ok_or_else(|| EngineError::Unsupported("WAV missing data chunk".into()))?;
    let mut mono = Vec::new();
    let ch = channels as usize;
    if audio_fmt == 1 && bits_per_sample == 16 {
        let frame_bytes = 2 * ch;
        for frame in pcm.chunks_exact(frame_bytes) {
            let mut sum = 0f32;
            for c in 0..ch {
                let s = i16::from_le_bytes([frame[2 * c], frame[2 * c + 1]]) as f32 / 32768.0;
                sum += s;
            }
            mono.push(sum / ch as f32);
        }
    } else if audio_fmt == 3 && bits_per_sample == 32 {
        let frame_bytes = 4 * ch;
        for frame in pcm.chunks_exact(frame_bytes) {
            let mut sum = 0f32;
            for c in 0..ch {
                let s = f32::from_le_bytes(frame[4 * c..4 * c + 4].try_into().unwrap());
                sum += s;
            }
            mono.push(sum / ch as f32);
        }
    } else {
        return Err(EngineError::Unsupported(format!(
            "unsupported WAV format={audio_fmt} bits={bits_per_sample} (expected 16-bit PCM or 32-bit float)"
        )));
    }
    if sample_rate == 16000 || mono.is_empty() {
        return Ok(mono);
    }
    let ratio = sample_rate as f64 / 16000.0;
    let out_len = ((mono.len() as f64) / ratio).round() as usize;
    let mut resampled = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let idx = (src.floor() as usize).min(mono.len() - 1);
        let frac = (src - idx as f64) as f32;
        let s0 = mono[idx];
        let s1 = mono[(idx + 1).min(mono.len() - 1)];
        resampled.push(s0 + frac * (s1 - s0));
    }
    Ok(resampled)
}

fn hz_to_mel_slaney(hz: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        (hz - f_min) / f_sp
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

/// Computes Whisper/Qwen3A-compatible 128-bin log-mel spectrogram windows (`[n_mel=128, n_frames_padded]`, multiple of 100).
pub fn compute_qwen3a_mel_windows(samples: &[f32]) -> Vec<(Vec<f32>, usize)> {
    const N_FFT: usize = 400;
    const HOP: usize = 160;
    const N_MEL: usize = 128;
    const N_BINS: usize = N_FFT / 2 + 1;
    const CHUNK: usize = 100;
    const WINDOW: usize = 800;

    if samples.is_empty() {
        return Vec::new();
    }
    let pad = N_FFT / 2;
    let mut padded = vec![0f32; samples.len() + 2 * pad];
    for i in 0..pad {
        let src = pad - i;
        padded[i] = if src < samples.len() { samples[src] } else { 0.0 };
    }
    padded[pad..pad + samples.len()].copy_from_slice(samples);
    for i in 0..pad {
        let src = samples.len() as isize - 2 - i as isize;
        padded[pad + samples.len() + i] = if src >= 0 { samples[src as usize] } else { 0.0 };
    }

    // Periodic Hann window
    let hann: Vec<f32> = (0..N_FFT).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / N_FFT as f32).cos())).collect();

    // Slaney mel filterbank [N_MEL, N_BINS]
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(8000.0);
    let mel_pts: Vec<f64> = (0..N_MEL + 2)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (N_MEL + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..N_BINS).map(|k| k as f64 * 16000.0 / N_FFT as f64).collect();
    let mut fb = vec![0f32; N_MEL * N_BINS];
    for m in 0..N_MEL {
        let (f_l, f_c, f_r) = (mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]);
        let enorm = 2.0 / (f_r - f_l);
        for k in 0..N_BINS {
            let f = fft_freqs[k];
            let lo = (f - f_l) / (f_c - f_l);
            let hi = (f_r - f) / (f_r - f_c);
            let w = lo.min(hi).max(0.0) * enorm;
            fb[m * N_BINS + k] = w as f32;
        }
    }

    // Precompute DFT cos/sin tables [N_BINS, N_FFT]
    let mut cos_t = vec![0f32; N_BINS * N_FFT];
    let mut sin_t = vec![0f32; N_BINS * N_FFT];
    for k in 0..N_BINS {
        let base = 2.0 * PI * k as f32 / N_FFT as f32;
        for n in 0..N_FFT {
            let a = base * n as f32;
            cos_t[k * N_FFT + n] = a.cos();
            sin_t[k * N_FFT + n] = a.sin();
        }
    }

    let n_frames_full = if padded.len() >= N_FFT { (padded.len() - N_FFT) / HOP + 1 } else { 0 };
    let n_eff = n_frames_full.min(samples.len() / HOP + 1);
    if n_eff == 0 {
        return Vec::new();
    }

    // Compute power spectrum -> mel -> log10 in parallel across frames
    let frame_mels: Vec<Vec<f32>> = (0..n_eff)
        .into_par_iter()
        .map(|f| {
            let off = f * HOP;
            let mut win = [0f32; N_FFT];
            for n in 0..N_FFT {
                win[n] = padded[off + n] * hann[n];
            }
            let mut mag2 = [0f32; N_BINS];
            for k in 0..N_BINS {
                let (mut re, mut im) = (0f32, 0f32);
                let cb = &cos_t[k * N_FFT..(k + 1) * N_FFT];
                let sb = &sin_t[k * N_FFT..(k + 1) * N_FFT];
                for n in 0..N_FFT {
                    re += win[n] * cb[n];
                    im -= win[n] * sb[n];
                }
                mag2[k] = re * re + im * im;
            }
            let mut mel = vec![0f32; N_MEL];
            for m in 0..N_MEL {
                let row = &fb[m * N_BINS..(m + 1) * N_BINS];
                let mut s = 0f32;
                for k in 0..N_BINS {
                    s += row[k] * mag2[k];
                }
                mel[m] = s.max(1e-10).log10();
            }
            mel
        })
        .collect();

    let mut mmax = f32::NEG_INFINITY;
    for fm in &frame_mels {
        for &v in fm {
            if v > mmax { mmax = v; }
        }
    }
    let floor = mmax - 8.0;

    // Split into <=800-frame windows padded to multiple of 100 frames
    let mut windows = Vec::new();
    let mut off = 0usize;
    while off < n_eff {
        let win_eff = WINDOW.min(n_eff - off);
        let n_chunks = (win_eff + CHUNK - 1) / CHUNK;
        let n_pad = n_chunks * CHUNK;
        let mut data = vec![0f32; N_MEL * n_pad];
        for t in 0..win_eff {
            let fm = &frame_mels[off + t];
            for m in 0..N_MEL {
                data[m * n_pad + t] = (fm[m].max(floor) + 4.0) / 4.0;
            }
        }
        windows.push((data, n_pad));
        off += WINDOW;
    }
    windows
}

struct EncLayer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    wq: TensorInfo,
    bq: Vec<f32>,
    wk: TensorInfo,
    bk: Vec<f32>,
    wv: TensorInfo,
    bv: Vec<f32>,
    wo: TensorInfo,
    bo: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    ff_up: TensorInfo,
    ff_up_b: Vec<f32>,
    ff_dn: TensorInfo,
    ff_dn_b: Vec<f32>,
}

pub struct Qwen3AEncoder {
    g: Arc<GgufFile>,
    d_model: usize,
    d_ff: usize,
    d_proj: usize,
    n_head: usize,
    eps: f32,
    c1_w: Vec<f32>,
    c1_b: Vec<f32>,
    c2_w: Vec<f32>,
    c2_b: Vec<f32>,
    c3_w: Vec<f32>,
    c3_b: Vec<f32>,
    conv_out_w: TensorInfo,
    pos_embd: Vec<f32>,
    layers: Vec<EncLayer>,
    post_ln_w: Vec<f32>,
    post_ln_b: Vec<f32>,
    mm1_w: TensorInfo,
    mm1_b: Vec<f32>,
    mm2_w: TensorInfo,
    mm2_b: Vec<f32>,
}

fn req_t(g: &GgufFile, name: &str) -> Result<TensorInfo, EngineError> {
    g.tensor(name).cloned().ok_or_else(|| EngineError::Unsupported(format!("qwen3a missing {name}")))
}
fn req_f32(g: &GgufFile, name: &str) -> Result<Vec<f32>, EngineError> {
    Ok(g.tensor_to_f32(&req_t(g, name)?))
}

fn conv2d_3x3_s2_p1_gelu(
    inp: &[f32], // [c_in, h_in, w_in]
    w: &[f32],   // ggml [3, 3, c_in, c_out] -> idx = ((oc * c_in + ic) * 3 + ky) * 3 + kx
    b: &[f32],   // [c_out]
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h_in + 2 - 3) / 2 + 1;
    let w_out = (w_in + 2 - 3) / 2 + 1;
    let out_ch: Vec<Vec<f32>> = (0..c_out)
        .into_par_iter()
        .map(|oc| {
            let mut plane = vec![0f32; h_out * w_out];
            let bias = b[oc];
            for oy in 0..h_out {
                for ox in 0..w_out {
                    let mut acc = bias;
                    for ic in 0..c_in {
                        let w_base = (oc * c_in + ic) * 9;
                        for ky in 0..3 {
                            let iy = oy as isize * 2 + ky as isize - 1;
                            if iy < 0 || iy >= h_in as isize { continue; }
                            let in_row = (ic * h_in + iy as usize) * w_in;
                            for kx in 0..3 {
                                let ix = ox as isize * 2 + kx as isize - 1;
                                if ix < 0 || ix >= w_in as isize { continue; }
                                acc += inp[in_row + ix as usize] * w[w_base + ky * 3 + kx];
                            }
                        }
                    }
                    plane[oy * w_out + ox] = gelu_erf(acc);
                }
            }
            plane
        })
        .collect();
    let mut out = Vec::with_capacity(c_out * h_out * w_out);
    for p in out_ch { out.extend_from_slice(&p); }
    (out, h_out, w_out)
}

impl Qwen3AEncoder {
    pub fn load(mmproj_path: impl AsRef<Path>) -> Result<Self, EngineError> {
        let g = Arc::new(GgufFile::open(mmproj_path).map_err(|e| EngineError::Gguf(e.to_string()))?);
        let d_model = g.get_u64("clip.audio.embedding_length").unwrap_or(1024) as usize;
        let d_ff = g.get_u64("clip.audio.feed_forward_length").unwrap_or(4096) as usize;
        let d_proj = g.get_u64("clip.audio.projection_dim").unwrap_or(2048) as usize;
        let n_head = g.get_u64("clip.audio.attention.head_count").unwrap_or(16) as usize;
        let n_blk = g.get_u64("clip.audio.block_count").unwrap_or(24) as usize;
        let eps = g.get_f64("clip.audio.attention.layer_norm_epsilon").unwrap_or(1e-5) as f32;

        let mut layers = Vec::with_capacity(n_blk);
        for i in 0..n_blk {
            let p = |s: &str| format!("a.blk.{i}.{s}");
            layers.push(EncLayer {
                ln1_w: req_f32(&g, &p("ln1.weight"))?,
                ln1_b: req_f32(&g, &p("ln1.bias"))?,
                wq: req_t(&g, &p("attn_q.weight"))?,
                bq: req_f32(&g, &p("attn_q.bias"))?,
                wk: req_t(&g, &p("attn_k.weight"))?,
                bk: req_f32(&g, &p("attn_k.bias"))?,
                wv: req_t(&g, &p("attn_v.weight"))?,
                bv: req_f32(&g, &p("attn_v.bias"))?,
                wo: req_t(&g, &p("attn_out.weight"))?,
                bo: req_f32(&g, &p("attn_out.bias"))?,
                ln2_w: req_f32(&g, &p("ln2.weight"))?,
                ln2_b: req_f32(&g, &p("ln2.bias"))?,
                ff_up: req_t(&g, &p("ffn_up.weight"))?,
                ff_up_b: req_f32(&g, &p("ffn_up.bias"))?,
                ff_dn: req_t(&g, &p("ffn_down.weight"))?,
                ff_dn_b: req_f32(&g, &p("ffn_down.bias"))?,
            });
        }
        Ok(Self {
            c1_w: req_f32(&g, "a.conv2d.1.weight")?,
            c1_b: req_f32(&g, "a.conv2d.1.bias")?,
            c2_w: req_f32(&g, "a.conv2d.2.weight")?,
            c2_b: req_f32(&g, "a.conv2d.2.bias")?,
            c3_w: req_f32(&g, "a.conv2d.3.weight")?,
            c3_b: req_f32(&g, "a.conv2d.3.bias")?,
            conv_out_w: req_t(&g, "a.conv_out.weight")?,
            pos_embd: req_f32(&g, "a.position_embd.weight")?,
            post_ln_w: req_f32(&g, "a.post_ln.weight")?,
            post_ln_b: req_f32(&g, "a.post_ln.bias")?,
            mm1_w: req_t(&g, "mm.a.mlp.1.weight")?,
            mm1_b: req_f32(&g, "mm.a.mlp.1.bias")?,
            mm2_w: req_t(&g, "mm.a.mlp.2.weight")?,
            mm2_b: req_f32(&g, "mm.a.mlp.2.bias")?,
            d_model, d_ff, d_proj, n_head, eps, layers, g,
        })
    }

    fn mv(&self, ti: &TensorInfo, x: &[f32], y: &mut [f32]) {
        gemv(ti.ty, self.g.tensor_data(ti), x, y, ti.n_cols(), ti.n_rows());
    }

    /// Encodes 16 kHz mono samples into projected `[N_tokens][2048]` embeddings for the Qwen3-ASR decoder.
    pub fn encode_samples(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let windows = compute_qwen3a_mel_windows(samples);
        let mut all_out = Vec::new();
        for (mel, n_frames) in windows {
            let n_chunks = n_frames / 100;
            let mut tokens: Vec<Vec<f32>> = Vec::new();
            for ch in 0..n_chunks {
                // Extract [1, 128, 100]
                let mut patch = vec![0f32; 128 * 100];
                for m in 0..128 {
                    patch[m * 100..(m + 1) * 100].copy_from_slice(&mel[m * n_frames + ch * 100..m * n_frames + (ch + 1) * 100]);
                }
                let (h1, hh1, ww1) = conv2d_3x3_s2_p1_gelu(&patch, &self.c1_w, &self.c1_b, 1, 128, 100, 480);
                let (h2, hh2, ww2) = conv2d_3x3_s2_p1_gelu(&h1, &self.c2_w, &self.c2_b, 480, hh1, ww1, 480);
                let (h3, hh3, ww3) = conv2d_3x3_s2_p1_gelu(&h2, &self.c3_w, &self.c3_b, 480, hh2, ww2, 480);
                // h3: [480, 16, 13] -> for each ox in 0..ww3: feat[c * 16 + oy] -> conv_out -> + pos_embd[ox]
                for ox in 0..ww3 {
                    let mut feat = vec![0f32; 480 * hh3];
                    for c in 0..480 {
                        for oy in 0..hh3 {
                            feat[c * hh3 + oy] = h3[(c * hh3 + oy) * ww3 + ox];
                        }
                    }
                    let mut tok = vec![0f32; self.d_model];
                    self.mv(&self.conv_out_w, &feat, &mut tok);
                    for i in 0..self.d_model {
                        tok[i] += self.pos_embd[ox * self.d_model + i];
                    }
                    tokens.push(tok);
                }
            }
            let n_pos = tokens.len();
            let hd = self.d_model / self.n_head;
            let scale = 1.0 / (hd as f32).sqrt();

            // 24-layer bidirectional Transformer encoder
            for l in &self.layers {
                let normed: Vec<Vec<f32>> = tokens.iter().map(|t| layer_norm(t, &l.ln1_w, &l.ln1_b, self.eps)).collect();
                let mut qs = vec![vec![0f32; self.d_model]; n_pos];
                let mut ks = vec![vec![0f32; self.d_model]; n_pos];
                let mut vs = vec![vec![0f32; self.d_model]; n_pos];
                for t in 0..n_pos {
                    self.mv(&l.wq, &normed[t], &mut qs[t]);
                    self.mv(&l.wk, &normed[t], &mut ks[t]);
                    self.mv(&l.wv, &normed[t], &mut vs[t]);
                    for i in 0..self.d_model {
                        qs[t][i] += l.bq[i];
                        ks[t][i] += l.bk[i];
                        vs[t][i] += l.bv[i];
                    }
                }
                let attn_all: Vec<Vec<f32>> = (0..n_pos)
                    .into_par_iter()
                    .map(|t| {
                        let mut out = vec![0f32; self.d_model];
                        for h in 0..self.n_head {
                            let qh = &qs[t][h * hd..(h + 1) * hd];
                            let mut sc = vec![0f32; n_pos];
                            let mut mx = f32::NEG_INFINITY;
                            for s in 0..n_pos {
                                let kh = &ks[s][h * hd..(h + 1) * hd];
                                let mut dot = 0f32;
                                for d in 0..hd { dot += qh[d] * kh[d]; }
                                sc[s] = dot * scale;
                                if sc[s] > mx { mx = sc[s]; }
                            }
                            let mut denom = 0f32;
                            for v in &mut sc { *v = (*v - mx).exp(); denom += *v; }
                            let inv = 1.0 / denom;
                            for s in 0..n_pos {
                                let w = sc[s] * inv;
                                let vh = &vs[s][h * hd..(h + 1) * hd];
                                for d in 0..hd { out[h * hd + d] += w * vh[d]; }
                            }
                        }
                        out
                    })
                    .collect();
                for t in 0..n_pos {
                    let mut o = vec![0f32; self.d_model];
                    self.mv(&l.wo, &attn_all[t], &mut o);
                    for i in 0..self.d_model { tokens[t][i] += o[i] + l.bo[i]; }
                    let n2 = layer_norm(&tokens[t], &l.ln2_w, &l.ln2_b, self.eps);
                    let mut up = vec![0f32; self.d_ff];
                    self.mv(&l.ff_up, &n2, &mut up);
                    for i in 0..self.d_ff { up[i] = gelu_erf(up[i] + l.ff_up_b[i]); }
                    let mut dn = vec![0f32; self.d_model];
                    self.mv(&l.ff_dn, &up, &mut dn);
                    for i in 0..self.d_model { tokens[t][i] += dn[i] + l.ff_dn_b[i]; }
                }
            }

            // post_ln + 2-layer GELU_ERF projector
            for t in 0..n_pos {
                let pn = layer_norm(&tokens[t], &self.post_ln_w, &self.post_ln_b, self.eps);
                let mut m1 = vec![0f32; self.d_model];
                self.mv(&self.mm1_w, &pn, &mut m1);
                for i in 0..self.d_model { m1[i] = gelu_erf(m1[i] + self.mm1_b[i]); }
                let mut m2 = vec![0f32; self.d_proj];
                self.mv(&self.mm2_w, &m1, &mut m2);
                for i in 0..self.d_proj { m2[i] += self.mm2_b[i]; }
                all_out.push(m2);
            }
        }
        all_out
    }
}

pub struct Qwen3AsrEngine {
    pub encoder: Qwen3AEncoder,
    pub decoder: NativeEngine,
}

impl Qwen3AsrEngine {
    pub fn load(model_path: impl AsRef<Path>, mmproj_path: impl AsRef<Path>, use_vulkan: bool) -> Result<Self, EngineError> {
        let encoder = Qwen3AEncoder::load(mmproj_path)?;
        let decoder = NativeEngine::load(
            model_path,
            EngineOptions {
                backend: if use_vulkan { BackendKind::Vulkan } else { BackendKind::CpuAvx2 },
                max_ctx: 4096,
                kv_pages: 256,
            },
        )?;
        Ok(Self { encoder, decoder })
    }

    pub fn transcribe_wav(&mut self, wav_bytes: &[u8], max_tokens: usize) -> Result<String, EngineError> {
        let samples = parse_wav_mono_16k(wav_bytes)?;
        let audio_embds = self.encoder.encode_samples(&samples);
        let prefix = self.decoder.tokenizer.encode("<|im_start|>user\n<|audio_start|>", false, true);
        let suffix = self.decoder.tokenizer.encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n", false, true);
        self.decoder.generate_multimodal_greedy(&prefix, &audio_embds, &suffix, max_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_parser_and_mel_spectrogram_smoke() {
        // Synthesize 0.25s 440 Hz sine wave WAV (16 kHz, 16-bit mono PCM)
        let sr = 16000u32;
        let n = 4000usize;
        let data_bytes = (n * 2) as u32;
        let mut wav = Vec::with_capacity(44 + data_bytes as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&sr.to_le_bytes());
        wav.extend_from_slice(&(sr * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_bytes.to_le_bytes());
        for i in 0..n {
            let s = ((2.0 * PI * 440.0 * i as f32 / sr as f32).sin() * 16000.0) as i16;
            wav.extend_from_slice(&s.to_le_bytes());
        }
        let samples = parse_wav_mono_16k(&wav).unwrap();
        assert_eq!(samples.len(), n);
        let wins = compute_qwen3a_mel_windows(&samples);
        assert_eq!(wins.len(), 1);
        assert_eq!(wins[0].1, 100); // padded to 1 chunk of 100 frames
        assert_eq!(wins[0].0.len(), 128 * 100);
    }
}
