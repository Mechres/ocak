use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateFamily {
    Zephyr,
    ChatMl,
    Llama2,
    Llama3,
    Xing,
    Inkling,
    Raw,
}

pub fn detect_family(chat_template: Option<&str>) -> TemplateFamily {
    let Some(t) = chat_template else {
        return TemplateFamily::Raw;
    };
    if t.contains("<|im_start|>") {
        TemplateFamily::ChatMl
    } else if t.contains("<|start_header_id|>") {
        TemplateFamily::Llama3
    } else if t.contains("<|message_user|>") || t.contains("<|message_model|>") {
        TemplateFamily::Inkling
    } else if t.contains("<_user>") || t.contains("<_bot>") {
        TemplateFamily::Xing
    } else if t.contains("<|user|>") {
        TemplateFamily::Zephyr
    } else if t.contains("[INST]") {
        TemplateFamily::Llama2
    } else {
        TemplateFamily::Raw
    }
}

/// Neutralizes role-framing special tokens inside untrusted user/tool content so
/// `parse_special=true` cannot be exploited for prompt/role injection.
fn sanitize_content(role: &str, text: &str) -> String {
    if role == "system" {
        return text.to_string();
    }
    const ROLE_TOKENS: &[&str] = &[
        "<|im_start|>",
        "<|im_end|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<_user>",
        "<_bot>",
        "<_system>",
        "<_end>",
        "<_observation>",
        "<|message_user|>",
        "<|message_model|>",
        "<|message_system|>",
        "<|message_tool|>",
        "<|end_message|>",
        "<|content_model_end_sampling|>",
    ];
    let mut s = text.to_string();
    for &tok in ROLE_TOKENS {
        if s.contains(tok) {
            let safe = format!("{}\u{200B}{}", &tok[..1], &tok[1..]);
            s = s.replace(tok, &safe);
        }
    }
    s
}

pub fn render(
    family: TemplateFamily,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut out = String::new();
    match family {
        TemplateFamily::Zephyr => {
            for m in messages {
                out.push_str("<|");
                out.push_str(&m.role);
                out.push_str("|>\n");
                out.push_str(&sanitize_content(&m.role, &m.content));
                out.push_str("</s>\n");
            }
            if add_generation_prompt {
                out.push_str("<|assistant|>\n");
            }
        }
        TemplateFamily::ChatMl => {
            for m in messages {
                out.push_str("<|im_start|>");
                out.push_str(&m.role);
                out.push('\n');
                out.push_str(&sanitize_content(&m.role, &m.content));
                out.push_str("<|im_end|>\n");
            }
            if add_generation_prompt {
                out.push_str("<|im_start|>assistant\n");
                if !enable_thinking {
                    out.push_str("<think>\n\n</think>\n\n");
                }
            }
        }
        TemplateFamily::Llama3 => {
            for m in messages {
                out.push_str("<|start_header_id|>");
                out.push_str(&m.role);
                out.push_str("<|end_header_id|>\n\n");
                out.push_str(&sanitize_content(&m.role, &m.content));
                out.push_str("<|eot_id|>");
            }
            if add_generation_prompt {
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
        }
        TemplateFamily::Llama2 => {
            let mut sys = String::new();
            let mut idx = 0;
            if let Some(first) = messages.first() {
                if first.role == "system" {
                    sys = first.content.clone();
                    idx = 1;
                }
            }
            let mut first_turn = true;
            for m in &messages[idx..] {
                let content = sanitize_content(&m.role, &m.content);
                if m.role == "user" {
                    out.push_str("[INST] ");
                    if first_turn && !sys.is_empty() {
                        out.push_str("<<SYS>>\n");
                        out.push_str(&sys);
                        out.push_str("\n<</SYS>>\n\n");
                    }
                    first_turn = false;
                    out.push_str(&content);
                    out.push_str(" [/INST]");
                } else if m.role == "assistant" {
                    out.push(' ');
                    out.push_str(&content);
                    out.push_str(" </s>");
                }
            }
        }
        TemplateFamily::Xing => {
            let mut idx = 0;
            if let Some(first) = messages.first() {
                if first.role == "system" {
                    out.push_str("<_system>");
                    out.push_str(&first.content);
                    out.push('\n');
                    idx = 1;
                }
            }
            if idx == 0 {
                out.push_str("<_system>You are Xing, a helpful AI assistant.\n");
            }
            for m in &messages[idx..] {
                let content = sanitize_content(&m.role, &m.content);
                match m.role.as_str() {
                    "user" => {
                        out.push_str("<_user>");
                        out.push_str(&content);
                    }
                    "assistant" | "bot" => {
                        out.push_str("<_bot>");
                        if !enable_thinking {
                            out.push_str("</think>");
                        }
                        out.push_str(&content);
                        out.push_str("<_end>\n");
                    }
                    "tool" => {
                        out.push_str("<_observation><tool_response>");
                        out.push_str(&content);
                        out.push_str("</tool_response>");
                    }
                    "system" => {
                        out.push_str("<_system>");
                        out.push_str(&m.content);
                    }
                    _ => {
                        out.push_str("<_user>");
                        out.push_str(&content);
                    }
                }
            }
            if add_generation_prompt {
                out.push_str("<_bot>");
                if enable_thinking {
                    out.push_str("<think>\n");
                } else {
                    out.push_str("</think>");
                }
            }
        }
        TemplateFamily::Inkling => {
            let effort = if enable_thinking { "0.9" } else { "0" };
            let mut effort_emitted = false;
            for m in messages {
                if !effort_emitted && m.role != "system" {
                    out.push_str("<|message_system|><|content_text|>Thinking effort level: ");
                    out.push_str(effort);
                    out.push_str("<|end_message|>");
                    effort_emitted = true;
                }
                let content = sanitize_content(&m.role, &m.content);
                match m.role.as_str() {
                    "system" => {
                        out.push_str("<|message_system|><|content_text|>");
                        out.push_str(&m.content);
                        out.push_str("<|end_message|>");
                    }
                    "user" => {
                        out.push_str("<|message_user|><|content_text|>");
                        out.push_str(&content);
                        out.push_str("<|end_message|>");
                    }
                    "assistant" => {
                        out.push_str("<|message_model|><|content_text|>");
                        out.push_str(&content);
                        out.push_str("<|end_message|><|content_model_end_sampling|>");
                    }
                    "tool" => {
                        out.push_str("<|message_tool|><|content_text|>");
                        out.push_str(&content);
                        out.push_str("<|end_message|>");
                    }
                    _ => {
                        out.push_str("<|message_user|><|content_text|>");
                        out.push_str(&content);
                        out.push_str("<|end_message|>");
                    }
                }
            }
            if !effort_emitted {
                out.push_str("<|message_system|><|content_text|>Thinking effort level: ");
                out.push_str(effort);
                out.push_str("<|end_message|>");
            }
        }
        TemplateFamily::Raw => {
            for m in messages {
                out.push_str(&m.content);
                out.push('\n');
            }
        }
    }
    out
}
