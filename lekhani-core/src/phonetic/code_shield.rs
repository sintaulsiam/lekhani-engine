//! Developer & Power-User Code-Mixing Shield
//!
//! Automatically detects programming identifiers, CLI flags, camelCase,
//! snake_case, URLs, and common developer tokens so they stay in raw English
//! without requiring manual F12 layout toggles.

/// Common programming keywords and developer tokens that should never be transliterated (alphabetically sorted for binary search)
pub const CODE_KEYWORDS: &[&str] = &[
    "apt", "async", "auto", "await", "awk", "bash", "bench", "bool", "branch", "break",
    "brew", "build", "bun", "byte", "cargo", "case", "cat", "catch", "char", "checkout",
    "cherry-pick", "clang", "class", "clone", "cmake", "code", "commit", "const", "continue",
    "cpp", "crate", "curl", "debug", "def", "deno", "deploy", "diff", "dnf", "docker",
    "double", "elif", "else", "enum", "except", "export", "extern", "false", "fetch",
    "finally", "find", "float", "fn", "for", "from", "func", "function", "gcc", "git",
    "github", "gitlab", "goto", "grep", "if", "impl", "import", "include", "init", "install",
    "instanceof", "int", "interface", "let", "long", "loop", "make", "match", "merge",
    "module", "namespace", "nano", "nil", "node", "none", "npm", "null", "nvim", "package",
    "pacman", "pip", "pnpm", "private", "protected", "pub", "public", "pull", "push",
    "python", "raise", "rebase", "release", "remote", "require", "reset", "return", "run",
    "rustc", "scp", "sed", "self", "short", "signed", "sizeof", "ssh", "stash", "status",
    "str", "string", "struct", "sudo", "super", "switch", "test", "this", "throw", "tmux",
    "trait", "true", "try", "type", "typeof", "undefined", "unsigned", "use", "val", "var",
    "vim", "void", "wget", "while", "yarn", "yield", "zsh",
];

/// Check if a Latin token is a developer code token, CLI flag, URL, or structured identifier
pub fn is_code_token(token: &str) -> bool {
    let trimmed = token.trim();
    if trimmed.len() < 2 || trimmed.contains('`') {
        return false;
    }

    // 1. CLI Flags: starts with "--" or "-" followed by ascii letters (e.g. "--help", "-rf", "-v")
    if trimmed.starts_with("--") && trimmed.len() >= 3 && trimmed.chars().skip(2).all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return true;
    }
    if trimmed.starts_with('-') && !trimmed.starts_with("--") && trimmed.len() >= 2 && trimmed.chars().skip(1).all(|c| c.is_ascii_alphabetic()) {
        return true;
    }

    // 2. URLs, Schemas & Web Addresses (e.g. "https://...", "http://...", "localhost:3000", "api.github.com")
    if trimmed.contains("://") || trimmed.starts_with("www.") || trimmed.starts_with("localhost:") {
        return true;
    }
    if (trimmed.ends_with(".com") || trimmed.ends_with(".org") || trimmed.ends_with(".net")
        || trimmed.ends_with(".io") || trimmed.ends_with(".dev") || trimmed.ends_with(".app")
        || trimmed.ends_with(".edu") || trimmed.ends_with(".gov") || trimmed.ends_with(".bd"))
        && trimmed.contains('.')
    {
        return true;
    }

    // 3. Email addresses
    if trimmed.contains('@') && trimmed.contains('.') && !trimmed.starts_with('@') && !trimmed.ends_with('@') {
        return true;
    }

    // 4. File paths and extensions (e.g. "/etc/hosts", "./scripts/run.sh", "main.rs", "Cargo.toml")
    if (trimmed.starts_with("./") || trimmed.starts_with("../") || trimmed.starts_with('/'))
        && trimmed.chars().all(|c| c.is_ascii_alphanumeric() || c == '/' || c == '.' || c == '_' || c == '-')
    {
        return true;
    }
    if let Some(dot_pos) = trimmed.rfind('.') {
        if dot_pos > 0 && dot_pos + 1 < trimmed.len() {
            let ext = &trimmed[dot_pos + 1..];
            if matches!(ext, "rs" | "js" | "ts" | "py" | "go" | "c" | "cpp" | "h" | "html" | "css" | "json" | "toml" | "yaml" | "yml" | "sh" | "md" | "txt" | "png" | "jpg" | "svg") {
                return true;
            }
        }
    }

    // 5. snake_case with ASCII letters/numbers (e.g. "user_id", "get_status", "api_key_v2")
    if trimmed.contains('_') && !trimmed.starts_with('_') && !trimmed.ends_with('_') {
        if trimmed.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return true;
        }
    }

    // 6. kebab-case with ASCII letters (e.g. "user-profile", "btn-primary")
    if trimmed.contains('-') && !trimmed.starts_with('-') && !trimmed.ends_with('-') {
        let parts: Vec<&str> = trimmed.split('-').collect();
        if parts.len() >= 2 && parts.iter().all(|p| p.len() >= 2 && p.chars().all(|c| c.is_ascii_alphanumeric())) {
            return true;
        }
    }

    // 7. camelCase or PascalCase with ASCII letters (e.g. "getUserProfile", "onClick", "HttpClient")
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() >= 4 && chars.iter().all(|c| c.is_ascii_alphanumeric()) {
        // Non-Avro uppercase characters (C, F, P, Q, V, W, X, B, G, J, K, L, M) immediately signal code identifiers
        const NON_AVRO_UPPER: &[char] = &[
            'C', 'F', 'P', 'Q', 'V', 'W', 'X', 'B', 'G', 'J', 'K', 'L', 'M',
        ];
        let has_non_avro_upper = chars.iter().skip(1).any(|c| NON_AVRO_UPPER.contains(c));
        if has_non_avro_upper && chars.iter().any(|c| c.is_ascii_lowercase()) {
            return true;
        }

        // For Avro uppercase letters (e.g. U in getUser), require camelCase word structure:
        // Internal uppercase followed by at least 2 lowercase letters, total length >= 6 (e.g. "getUser", "setTimeout")
        let has_lower = chars.iter().any(|c| c.is_ascii_lowercase());
        if chars.len() >= 6 && has_lower {
            for i in 1..(chars.len() - 2) {
                if chars[i].is_ascii_uppercase()
                    && chars[i + 1].is_ascii_lowercase()
                    && chars[i + 2].is_ascii_lowercase()
                {
                    return true;
                }
            }
        }
    }

    // 8. Reserved programming keywords (binary search)
    let lower = trimmed.to_ascii_lowercase();
    if CODE_KEYWORDS.binary_search(&lower.as_str()).is_ok() {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_keywords_sorted() {
        for i in 1..CODE_KEYWORDS.len() {
            assert!(
                CODE_KEYWORDS[i - 1] < CODE_KEYWORDS[i],
                "CODE_KEYWORDS not sorted: {} >= {}",
                CODE_KEYWORDS[i - 1],
                CODE_KEYWORDS[i]
            );
        }
    }

    #[test]
    fn test_code_token_detection() {
        // CLI flags
        assert!(is_code_token("--help"));
        assert!(is_code_token("--version"));
        assert!(is_code_token("-rf"));

        // Identifiers
        assert!(is_code_token("onClick"));
        assert!(is_code_token("getUserProfile"));
        assert!(is_code_token("user_id"));
        assert!(is_code_token("api_key"));
        assert!(is_code_token("btn-primary"));

        // Keywords
        assert!(is_code_token("const"));
        assert!(is_code_token("return"));
        assert!(is_code_token("docker"));
        assert!(is_code_token("cargo"));
        assert!(is_code_token("async"));

        // URLs & Emails
        assert!(is_code_token("https://github.com"));
        assert!(is_code_token("user@gmail.com"));
        assert!(is_code_token("google.com"));
        assert!(is_code_token("localhost:3000"));

        // Normal Bengali phonetic inputs should NOT be flagged as code
        assert!(!is_code_token("bhalo"));
        assert!(!is_code_token("amar"));
        assert!(!is_code_token("bangla"));
        assert!(!is_code_token("kemon"));
        assert!(!is_code_token("kortesi"));
    }
}
