//! Native OpenAI & Anthropic tool-calling prompt injection and `<tool_call>` parser.
use ocak_tok::ChatMessage;
use serde_json::{json, Value};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

pub fn extract_openai_tools(raw: &[Value]) -> Vec<ToolDef> {
    raw.iter()
        .filter_map(|t| {
            let f = t.get("function").unwrap_or(t);
            let name = f.get("name")?.as_str()?.to_string();
            let description = f.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let parameters = f.get("parameters").cloned().unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            Some(ToolDef { name, description, parameters })
        })
        .collect()
}

pub fn extract_anthropic_tools(raw: &[Value]) -> Vec<ToolDef> {
    raw.iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?.to_string();
            let description = t.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let parameters = t
                .get("input_schema")
                .or_else(|| t.get("parameters"))
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            Some(ToolDef { name, description, parameters })
        })
        .collect()
}

pub fn format_tools_system(tools: &[ToolDef]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for t in tools {
        let entry = json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
        });
        s.push_str(&entry.to_string());
        s.push('\n');
    }
    s.push_str(
        "</tools>\n\nFor each function call, return a json object with function name and arguments within \
         <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    s
}

pub fn inject_tools_system(msgs: &mut Vec<ChatMessage>, tools: &[ToolDef]) {
    if tools.is_empty() {
        return;
    }
    let block = format_tools_system(tools);
    if let Some(first) = msgs.first_mut() {
        if first.role == "system" {
            if !first.content.is_empty() {
                first.content.push_str("\n\n");
            }
            first.content.push_str(&block);
            return;
        }
    }
    msgs.insert(0, ChatMessage { role: "system".into(), content: block });
}

fn value_content_to_string(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|p| p.get("text").or_else(|| p.get("content")).and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

pub fn openai_to_chat(messages: &[Value], tools: &[ToolDef]) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    for m in messages {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let mut content = value_content_to_string(m.get("content"));
        if role == "tool" {
            let wrapped = format!("<tool_response>\n{content}\n</tool_response>");
            out.push(ChatMessage { role: "user".into(), content: wrapped });
            continue;
        }
        if role == "assistant" {
            if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tcs {
                    let f = tc.get("function").unwrap_or(tc);
                    if let Some(name) = f.get("name").and_then(|v| v.as_str()) {
                        let args: Value = match f.get("arguments") {
                            Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                            Some(v) => v.clone(),
                            None => json!({}),
                        };
                        if !content.is_empty() {
                            content.push('\n');
                        }
                        content.push_str(&format!(
                            "<tool_call>\n{}\n</tool_call>",
                            json!({ "name": name, "arguments": args })
                        ));
                    }
                }
            }
        }
        out.push(ChatMessage { role: role.to_string(), content });
    }
    inject_tools_system(&mut out, tools);
    out
}

pub fn anthropic_messages_to_chat(system: Option<&Value>, messages: &[Value], tools: &[ToolDef]) -> Vec<ChatMessage> {
    let mut out = Vec::new();
    if let Some(sys) = system {
        let s = match sys {
            Value::String(s) => s.clone(),
            Value::Array(a) => a.iter().filter_map(|v| v.get("text").and_then(|t| t.as_str())).collect::<Vec<_>>().join("\n"),
            _ => String::new(),
        };
        if !s.is_empty() {
            out.push(ChatMessage { role: "system".into(), content: s });
        }
    }
    for m in messages {
        let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user").to_string();
        let content = match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(arr)) => {
                let mut parts = Vec::new();
                for b in arr {
                    match b.get("type").and_then(|v| v.as_str()) {
                        Some("tool_use") => {
                            if let Some(name) = b.get("name").and_then(|v| v.as_str()) {
                                let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
                                parts.push(format!(
                                    "<tool_call>\n{}\n</tool_call>",
                                    json!({ "name": name, "arguments": input })
                                ));
                            }
                        }
                        Some("tool_result") => {
                            let body = value_content_to_string(b.get("content"));
                            parts.push(format!("<tool_response>\n{body}\n</tool_response>"));
                        }
                        _ => {
                            if let Some(t) = b.get("text").or_else(|| b.get("content")).and_then(|t| t.as_str()) {
                                parts.push(t.to_string());
                            }
                        }
                    }
                }
                parts.join("\n")
            }
            _ => String::new(),
        };
        out.push(ChatMessage { role, content });
    }
    inject_tools_system(&mut out, tools);
    out
}

/// Parses `<tool_call>...</tool_call>` blocks from model output, returning `(remaining_text, tool_calls)`.
pub fn parse_tool_calls(text: &str, id_prefix: &str) -> (String, Vec<ParsedToolCall>) {
    let mut calls = Vec::new();
    let mut clean = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        clean.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            // Unclosed tag: leave untouched
            clean.push_str(&rest[start..]);
            rest = "";
            break;
        };
        let mut inner = after[..end].trim();
        if let Some(stripped) = inner.strip_prefix("```json").or_else(|| inner.strip_prefix("```")) {
            inner = stripped.trim().trim_end_matches("```").trim();
        }
        if let Ok(v) = serde_json::from_str::<Value>(inner) {
            if let Some(name) = v.get("name").and_then(|n| n.as_str()) {
                let arguments = match v.get("arguments").or_else(|| v.get("input")) {
                    Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                    Some(obj) => obj.clone(),
                    None => json!({}),
                };
                let id = format!("{id_prefix}{}", &Uuid::new_v4().simple().to_string()[..16]);
                calls.push(ParsedToolCall { id, name: name.to_string(), arguments });
                rest = &after[end + "</tool_call>".len()..];
                continue;
            }
        }
        clean.push_str(&rest[start..start + "<tool_call>".len() + end + "</tool_call>".len()]);
        rest = &after[end + "</tool_call>".len()..];
    }
    clean.push_str(rest);
    (clean.trim().to_string(), calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_roundtrip_and_parsing() {
        let oa = extract_openai_tools(&[json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
            }
        })]);
        assert_eq!(oa.len(), 1);
        assert_eq!(oa[0].name, "get_weather");

        let ant = extract_anthropic_tools(&[json!({
            "name": "read_file",
            "description": "Read a file",
            "input_schema": { "type": "object", "properties": { "path": { "type": "string" } } }
        })]);
        assert_eq!(ant.len(), 1);
        assert_eq!(ant[0].name, "read_file");

        let msgs = openai_to_chat(
            &[
                json!({ "role": "user", "content": "Weather in Istanbul?" }),
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Istanbul\"}" } }]
                }),
                json!({ "role": "tool", "tool_call_id": "call_1", "content": "22C sunny" }),
            ],
            &oa,
        );
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.contains("<tools>"));
        assert!(msgs[2].content.contains("<tool_call>"));
        assert!(msgs[3].content.contains("<tool_response>"));

        let raw = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Istanbul\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": \"{\\\"path\\\": \\\"README.md\\\"}\"}\n</tool_call>";
        let (clean, calls) = parse_tool_calls(raw, "call_");
        assert_eq!(clean, "Let me check.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], "Istanbul");
        assert_eq!(calls[1].name, "read_file");
        assert_eq!(calls[1].arguments["path"], "README.md");
    }
}
