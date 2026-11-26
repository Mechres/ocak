//! Continuous multi-slot request batcher for `ocak-server`.
//! Concurrent requests targeting the same active pure-attention native model are dynamically
//! admitted into the active slot pool and decoded together via `NativeEngine::step_slots`
//! (`forward_multi_slot`) in a single batched Vulkan submit per decode step.
use crate::AppState;
use ocak_engine::{ActiveSlot, GenRequest, GenResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};

pub const MAX_BATCH_SLOTS: usize = 8;

pub struct BatchJob {
    pub model_id: String,
    pub greq: GenRequest,
    pub token_tx: Option<mpsc::Sender<String>>,
    pub done_tx: oneshot::Sender<Result<GenResult, String>>,
}

pub struct BatchCoordinator {
    queue: Mutex<VecDeque<BatchJob>>,
    running: AtomicBool,
}

impl Default for BatchCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchCoordinator {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            running: AtomicBool::new(false),
        }
    }

    pub fn submit(self: &Arc<Self>, st: Arc<AppState>, job: BatchJob) {
        self.queue.lock().unwrap().push_back(job);
        if self
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let coord = Arc::clone(self);
            tokio::task::spawn_blocking(move || coord.run_loop(st));
        }
    }

    fn run_loop(&self, st: Arc<AppState>) {
        struct LiveJob {
            slot: ActiveSlot,
            token_tx: Option<mpsc::Sender<String>>,
            done_tx: oneshot::Sender<Result<GenResult, String>>,
        }

        loop {
            let first = {
                let mut q = self.queue.lock().unwrap();
                q.pop_front()
            };
            let Some(first_job) = first else {
                self.running.store(false, Ordering::Release);
                // Double-check against a race between pop_front and store(false)
                if self.queue.lock().unwrap().is_empty() {
                    break;
                }
                if self
                    .running
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    break;
                }
                continue;
            };

            let target_model = first_job.model_id.clone();
            let mut guard = st.native.blocking_lock();
            if guard.as_ref().map(|a| a.id.as_str()) != Some(target_model.as_str()) {
                drop(guard);
                if let Some(entry) = st.find_model(&target_model).cloned() {
                    if let Err(e) = tokio::runtime::Handle::current().block_on(st.ensure_native(&entry)) {
                        let _ = first_job.done_tx.send(Err(e));
                        continue;
                    }
                }
                guard = st.native.blocking_lock();
            }
            let Some(active) = guard.as_mut() else {
                let _ = first_job.done_tx.send(Err("native engine not loaded".into()));
                continue;
            };
            if active.id != target_model {
                let _ = first_job.done_tx.send(Err(format!(
                    "model mismatch: requested '{target_model}' but '{}' is currently loaded",
                    active.id
                )));
                continue;
            }
            if !active.engine.supports_multi_slot() {
                let mut token_tx = first_job.token_tx;
                let res = active.engine.generate(&first_job.greq, |_, piece| {
                    if piece.is_empty() {
                        return true;
                    }
                    if let Some(tx) = &mut token_tx {
                        match tx.try_send(piece.to_string()) {
                            Ok(()) => true,
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                tracing::warn!(model = %target_model, "SSE client channel full; cancelling slow stream");
                                false
                            }
                            Err(mpsc::error::TrySendError::Closed(_)) => false,
                        }
                    } else {
                        true
                    }
                });
                let _ = first_job.done_tx.send(res.map_err(|e| e.to_string()));
                st.governor.blocking_lock().touch(&target_model, Instant::now());
                continue;
            }

            let mut live: Vec<LiveJob> = Vec::new();
            match active.engine.open_slot(first_job.greq) {
                Ok(slot) => live.push(LiveJob {
                    slot,
                    token_tx: first_job.token_tx,
                    done_tx: first_job.done_tx,
                }),
                Err(e) => {
                    let _ = first_job.done_tx.send(Err(e.to_string()));
                    continue;
                }
            }

            // Continuous batching loop: admit newly arrived compatible jobs between every decode step!
            while !live.is_empty() {
                // Admit any queued jobs for the same model up to MAX_BATCH_SLOTS
                while live.len() < MAX_BATCH_SLOTS {
                    let next_opt = {
                        let mut q = self.queue.lock().unwrap();
                        if q.front().map(|j| j.model_id == target_model).unwrap_or(false) {
                            q.pop_front()
                        } else {
                            None
                        }
                    };
                    let Some(job) = next_opt else { break };
                    match active.engine.open_slot(job.greq) {
                        Ok(slot) => live.push(LiveJob {
                            slot,
                            token_tx: job.token_tx,
                            done_tx: job.done_tx,
                        }),
                        Err(e) => {
                            let _ = job.done_tx.send(Err(e.to_string()));
                        }
                    }
                }

                // Extract slots slice temporarily for step_slots
                let mut slots_tmp: Vec<ActiveSlot> = Vec::with_capacity(live.len());
                let mut metas: Vec<(Option<mpsc::Sender<String>>, oneshot::Sender<Result<GenResult, String>>)> =
                    Vec::with_capacity(live.len());
                for lj in live.drain(..) {
                    slots_tmp.push(lj.slot);
                    metas.push((lj.token_tx, lj.done_tx));
                }

                let step_res = active.engine.step_slots(&mut slots_tmp, |idx, _tok, piece| {
                    if piece.is_empty() {
                        return true;
                    }
                    if let Some(tx) = &metas[idx].0 {
                        match tx.try_send(piece.to_string()) {
                            Ok(()) => true,
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                tracing::warn!(model = %target_model, slot = idx, "SSE client channel full; decoupling slow slot");
                                false
                            }
                            Err(mpsc::error::TrySendError::Closed(_)) => false,
                        }
                    } else {
                        true
                    }
                });

                match step_res {
                    Ok(done_flags) => {
                        for ((slot, (token_tx, done_tx)), is_done) in
                            slots_tmp.into_iter().zip(metas).zip(done_flags)
                        {
                            if is_done {
                                let gr = active.engine.close_slot(slot);
                                st.tokens_generated_total
                                    .fetch_add(gr.timings.predicted_n as u64, Ordering::Relaxed);
                                st.kv_radix_hits_total
                                    .fetch_add(gr.timings.cache_n as u64, Ordering::Relaxed);
                                let _ = done_tx.send(Ok(gr));
                            } else {
                                live.push(LiveJob { slot, token_tx, done_tx });
                            }
                        }
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        for (slot, (_tx, done_tx)) in slots_tmp.into_iter().zip(metas) {
                            let _ = active.engine.close_slot(slot);
                            let _ = done_tx.send(Err(msg.clone()));
                        }
                        break;
                    }
                }
            }
            st.governor.blocking_lock().touch(&target_model, Instant::now());
        }
    }
}
