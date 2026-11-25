/// Enforces a reasoning-token budget: once the model has spent `max_reasoning_tokens`
/// tokens inside `<think> ... </think>`, the caller should force the close token.
#[derive(Debug, Clone)]
pub struct ThinkingGovernor {
    open: u32,
    close: u32,
    max: usize,
    inside: bool,
    used: usize,
}

impl ThinkingGovernor {
    pub fn new(think_open_id: u32, think_close_id: u32, max_reasoning_tokens: usize) -> Self {
        Self { open: think_open_id, close: think_close_id, max: max_reasoning_tokens, inside: false, used: 0 }
    }

    /// Starts in the "inside `<think>`" state (for prompts that already end with `<think>\n`).
    pub fn start_inside(mut self) -> Self {
        self.inside = true;
        self
    }

    /// Feed every token (prompt tail and generated) in order.
    pub fn observe(&mut self, tok: u32) {
        if tok == self.open {
            self.inside = true;
            self.used = 0;
        } else if tok == self.close {
            self.inside = false;
            self.used = 0;
        } else if self.inside {
            self.used += 1;
        }
    }

    /// `Some(close_id)` once the reasoning budget is used up while inside `<think>`.
    pub fn forced_next(&mut self) -> Option<u32> {
        if self.inside && self.used >= self.max { Some(self.close) } else { None }
    }

    pub fn in_thinking(&self) -> bool { self.inside }
    pub fn reasoning_tokens(&self) -> usize { self.used }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forces_close_after_budget() {
        let mut g = ThinkingGovernor::new(10, 11, 3);
        assert_eq!(g.forced_next(), None);
        g.observe(5); // outside: not counted
        assert!(!g.in_thinking());
        g.observe(10);
        assert!(g.in_thinking());
        for t in [1, 2] {
            g.observe(t);
            assert_eq!(g.forced_next(), None);
        }
        g.observe(3);
        assert_eq!(g.forced_next(), Some(11));
        assert_eq!(g.forced_next(), Some(11)); // stays forced until the close token is observed
        g.observe(11);
        assert!(!g.in_thinking());
        assert_eq!(g.forced_next(), None);
        g.observe(7);
        assert_eq!(g.forced_next(), None);
    }

    #[test]
    fn natural_close_before_budget_and_reopen() {
        let mut g = ThinkingGovernor::new(10, 11, 2);
        g.observe(10);
        g.observe(1);
        g.observe(11);
        assert_eq!(g.forced_next(), None);
        g.observe(10);
        assert_eq!(g.reasoning_tokens(), 0);
        g.observe(1);
        g.observe(2);
        assert_eq!(g.forced_next(), Some(11));
    }

    #[test]
    fn zero_budget_and_start_inside() {
        let mut g = ThinkingGovernor::new(10, 11, 0).start_inside();
        assert_eq!(g.forced_next(), Some(11));
        let mut g = ThinkingGovernor::new(10, 11, 2).start_inside();
        g.observe(1);
        assert_eq!(g.forced_next(), None);
        g.observe(1);
        assert_eq!(g.forced_next(), Some(11));
    }
}
