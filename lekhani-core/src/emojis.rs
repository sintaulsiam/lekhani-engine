//! Emoji and Bengali Symbol Shortcode Mapping

use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct EmojiMap {
    map: HashMap<String, String>,
}

impl Default for EmojiMap {
    fn default() -> Self {
        Self::new()
    }
}

impl EmojiMap {
    pub fn new() -> Self {
        let mut map = HashMap::new();

        // Bengali Symbols & Punctuation
        map.insert("*taka*".to_string(), "৳".to_string());
        map.insert("*dari*".to_string(), "।".to_string());
        map.insert("*ddari*".to_string(), "॥".to_string());
        map.insert("*hasant*".to_string(), "্".to_string());
        map.insert("*khandatta*".to_string(), "ৎ".to_string());
        map.insert("*anusvara*".to_string(), "ং".to_string());
        map.insert("*visarga*".to_string(), "ঃ".to_string());
        map.insert("*candrabindu*".to_string(), "ঁ".to_string());
        map.insert("*avagraha*".to_string(), "ঽ".to_string());
        map.insert("*isshar*".to_string(), "৺".to_string());

        // Common Bengali word emojis
        map.insert(":bhalobasha:".to_string(), "❤️".to_string());
        map.insert(":pranam:".to_string(), "🙏".to_string());
        map.insert(":shurjo:".to_string(), "☀️".to_string());
        map.insert(":chad:".to_string(), "🌙".to_string());
        map.insert(":ful:".to_string(), "🌸".to_string());
        map.insert(":golap:".to_string(), "🌹".to_string());
        map.insert(":aguni:".to_string(), "🔥".to_string());
        map.insert(":hasukhi:".to_string(), "😊".to_string());
        map.insert(":kanna:".to_string(), "😭".to_string());
        map.insert(":thik:".to_string(), "✅".to_string());
        map.insert(":bhul:".to_string(), "❌".to_string());
        map.insert(":shabash:".to_string(), "👍".to_string());

        // Standard English shortcodes
        map.insert(":smile:".to_string(), "😊".to_string());
        map.insert(":grin:".to_string(), "😁".to_string());
        map.insert(":joy:".to_string(), "😂".to_string());
        map.insert(":heart:".to_string(), "❤️".to_string());
        map.insert(":fire:".to_string(), "🔥".to_string());
        map.insert(":thumbsup:".to_string(), "👍".to_string());
        map.insert(":pray:".to_string(), "🙏".to_string());
        map.insert(":sparkles:".to_string(), "✨".to_string());
        map.insert(":check:".to_string(), "✔️".to_string());
        map.insert(":star:".to_string(), "⭐".to_string());
        map.insert(":rocket:".to_string(), "🚀".to_string());
        map.insert(":tada:".to_string(), "🎉".to_string());

        // Emoticons
        map.insert(":)".to_string(), "😊".to_string());
        map.insert(":-)".to_string(), "😊".to_string());
        map.insert(";)".to_string(), "😉".to_string());
        map.insert(";-)".to_string(), "😉".to_string());
        map.insert(":D".to_string(), "😃".to_string());
        map.insert(":-D".to_string(), "😃".to_string());
        map.insert(":P".to_string(), "😋".to_string());
        map.insert(":-P".to_string(), "😋".to_string());
        map.insert("<3".to_string(), "❤️".to_string());

        Self { map }
    }

    pub fn lookup(&self, code: &str) -> Option<&String> {
        self.map.get(code)
    }
}
