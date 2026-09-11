//! Emoji and Bengali Symbol Shortcode Mapping Engine

use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct EmojiMap {
    map: HashMap<String, String>,
    sorted_entries: Vec<(String, String)>,
    keywords: HashMap<String, Vec<String>>,
}

impl Default for EmojiMap {
    fn default() -> Self {
        Self::new()
    }
}

impl EmojiMap {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        let mut keywords: HashMap<String, Vec<String>> = HashMap::new();

        macro_rules! add_emoji {
            ($code:expr, $emoji:expr, [$($kw:expr),*]) => {
                map.insert($code.to_string(), $emoji.to_string());
                $(
                    keywords.entry($kw.to_lowercase()).or_default().push($emoji.to_string());
                )*
            };
            ($code:expr, $emoji:expr) => {
                map.insert($code.to_string(), $emoji.to_string());
            };
        }

        // 1. Bengali Symbols & Punctuation
        add_emoji!("*taka*", "৳", ["taka", "tk", "poisha", "টাকা"]);
        add_emoji!("*tk*", "৳", ["taka", "tk", "টাকা"]);
        add_emoji!("$taka", "৳");
        add_emoji!("$tk", "৳");
        add_emoji!("$$", "৳");
        add_emoji!("*dari*", "।", ["dari", "darhi", "দাড়ি", "দাড়ি"]);
        add_emoji!("*ddari*", "॥");
        add_emoji!("*hasant*", "্", ["hasant", "hasanta", "hosonto", "hoshonto", "হসন্ত"]);
        add_emoji!("*hosonto*", "্");
        add_emoji!("*hoshonto*", "্");
        add_emoji!("*hasanta*", "্");
        add_emoji!("*halant*", "্");
        add_emoji!("*virama*", "্");
        add_emoji!("*khandatta*", "ৎ", ["khandatta", "khandata", "খন্ডত"]);
        add_emoji!("*khandata*", "ৎ");
        add_emoji!("*khanda_ta*", "ৎ");
        add_emoji!("*khandota*", "ৎ");
        add_emoji!("*anusvara*", "ং", ["anusvara", "anushbar", "onushbar", "অনুস্বার"]);
        add_emoji!("*anusvar*", "ং");
        add_emoji!("*anushbar*", "ং");
        add_emoji!("*onushbar*", "ং");
        add_emoji!("*onushor*", "ং");
        add_emoji!("*anushar*", "ং");
        add_emoji!("*visarga*", "ঃ", ["visarga", "bishorgo", "bisharga", "বিসর্গ"]);
        add_emoji!("*visarg*", "ঃ");
        add_emoji!("*bishorgo*", "ঃ");
        add_emoji!("*bisharga*", "ঃ");
        add_emoji!("*bisarga*", "ঃ");
        add_emoji!("*candrabindu*", "ঁ", ["candrabindu", "chandrabindu", "chondrobindu", "চন্দ্রবিন্দু"]);
        add_emoji!("*chandrabindu*", "ঁ");
        add_emoji!("*chondrobindu*", "ঁ");
        add_emoji!("*chandra*", "ঁ");
        add_emoji!("*candra*", "ঁ");
        add_emoji!("*avagraha*", "ঽ");
        add_emoji!("*isshar*", "৺", ["isshar", "ishwar", "iswar", "ঈশ্বর"]);
        add_emoji!("*ishwar*", "৺");
        add_emoji!("*iswar*", "৺");
        add_emoji!("*ganda*", "𑜰");

        // 2. Bengali Named Words & Cultural Expressions
        add_emoji!(":bhalobasha:", "❤️", ["bhalobasha", "valobasha", "bhalobasa", "ভালোবাসা", "ভালবাসা", "love", "heart"]);
        add_emoji!(":valobasha:", "❤️");
        add_emoji!(":prem:", "💖", ["prem", "প্রেম", "crush"]);
        add_emoji!(":pranam:", "🙏", ["pranam", "প্রণাম", "adab", "আদাব", "namaskar", "নমস্কার", "pray"]);
        add_emoji!(":salam:", "🤝", ["salam", "সালাম", "handshake"]);
        add_emoji!(":adab:", "🙏");
        add_emoji!(":shurjo:", "☀️", ["shurjo", "surjo", "সূর্য", "sun", "sunny"]);
        add_emoji!(":surjo:", "☀️");
        add_emoji!(":chad:", "🌙", ["chad", "chand", "চাঁদ", "moon"]);
        add_emoji!(":tara:", "⭐", ["tara", "তারা", "star"]);
        add_emoji!(":ful:", "🌸", ["ful", "phul", "ফুল", "flower", "blossom"]);
        add_emoji!(":golap:", "🌹", ["golap", "গোলাপ", "rose"]);
        add_emoji!(":padma:", "🪷", ["padma", "পদ্ম", "lotus"]);
        add_emoji!(":shapla:", "🪷", ["shapla", "শাপলা"]);
        add_emoji!(":aguni:", "🔥", ["agun", "আগুন", "fire", "hot", "lit"]);
        add_emoji!(":agun:", "🔥");
        add_emoji!(":hasukhi:", "😊", ["hasukhi", "khushi", "খুশি", "happy"]);
        add_emoji!(":hashi:", "😄", ["hashi", "hasi", "হাসি", "smile", "laugh"]);
        add_emoji!(":kanna:", "😭", ["kanna", "কান্না", "dukkho", "দুঃখ", "cry", "sob"]);
        add_emoji!(":thik:", "✅", ["thik", "ঠিক", "yes", "correct", "check"]);
        add_emoji!(":bhul:", "❌", ["bhul", "ভুল", "no", "wrong", "cross"]);
        add_emoji!(":shabash:", "👍", ["shabash", "sabash", "সাবাশ", "bhalo", "like", "good", "thumbsup"]);
        add_emoji!(":durbar:", "⚡", ["durbar", "duronto", "দুর্বার", "bijli", "বিজলী", "zap", "flash"]);
        add_emoji!(":bijli:", "⚡");
        add_emoji!(":chokh:", "👀", ["chokh", "চোখ", "eyes", "look"]);
        add_emoji!(":machh:", "🐟", ["mach", "machh", "মাছ", "fish"]);
        add_emoji!(":ilish:", "🐟", ["ilish", "ইলিশ"]);
        add_emoji!(":pakhi:", "🐦", ["pakhi", "পাখি", "bird"]);
        add_emoji!(":doyel:", "🐦", ["doyel", "দোয়েল"]);
        add_emoji!(":bagh:", "🐅", ["bagh", "বাঘ", "tiger"]);
        add_emoji!(":singho:", "🦁", ["singho", "সিংহ", "lion"]);
        add_emoji!(":shap:", "🐍", ["shap", "সাপ", "snake"]);
        add_emoji!(":gach:", "🌳", ["gach", "গাছ", "tree"]);
        add_emoji!(":brishti:", "🌧️", ["brishti", "বৃষ্টি", "rain", "rainy"]);
        add_emoji!(":megh:", "☁️", ["megh", "মেঘ", "cloud"]);
        add_emoji!(":cha:", "☕", ["cha", "চা", "tea", "coffee"]);
        add_emoji!(":pani:", "💧", ["pani", "পানি", "jol", "জল", "water"]);
        add_emoji!(":vath:", "🍚", ["bhat", "ভাত", "rice"]);
        add_emoji!(":dudh:", "🥛", ["dudh", "দুধ", "milk"]);
        add_emoji!(":mishti:", "🍬", ["mishti", "মিষ্টি", "sweet", "candy"]);
        add_emoji!(":boishakh:", "🎭", ["boishakh", "বৈশাখ", "mela", "মেলা"]);
        add_emoji!(":potaka:", "🇧🇩", ["potaka", "পতাকা", "bangladesh", "বাংলাদেশ", "bd", "flag_bd"]);
        add_emoji!(":bangladesh:", "🇧🇩");
        add_emoji!(":bd:", "🇧🇩");
        add_emoji!(":biral:", "🐱", ["biral", "বিড়াল", "বিড়াল", "cat", "kitty"]);
        add_emoji!(":kukur:", "🐶", ["kukur", "কুকুর", "dog", "puppy"]);
        add_emoji!(":gari:", "🚗", ["gari", "গাড়ি", "গাড়ি", "car"]);
        add_emoji!(":boi:", "📖", ["boi", "বই", "book", "read"]);
        add_emoji!(":kolom:", "🖊️", ["kolom", "কলম", "pen", "write"]);
        add_emoji!(":bari:", "🏠", ["bari", "বাড়ি", "বাড়ি", "home", "house"]);
        add_emoji!(":alo:", "💡", ["alo", "আলো", "bulb", "idea", "light"]);
        add_emoji!(":ghori:", "⏰", ["ghori", "ঘড়ি", "ঘড়ি", "clock", "time"]);
        add_emoji!(":phone:", "📱", ["phone", "ফোন", "mobile", "মোবাইল"]);
        add_emoji!(":daktar:", "🩺", ["daktar", "ডাক্তার", "doctor", "hospital", "হাসপাতাল"]);
        add_emoji!(":khela:", "⚽", ["khela", "খেলা", "football", "soccer"]);
        add_emoji!(":cricket:", "🏏", ["cricket", "ক্রিকেট"]);
        add_emoji!(":gaan:", "🎵", ["gan", "gaan", "গান", "music", "song"]);

        // 3. Comprehensive Popular Emoji Shortcodes
        // Smileys & Emotion
        add_emoji!(":smile:", "😊", ["smile", "happy"]);
        add_emoji!(":smiley:", "😃", ["smiley"]);
        add_emoji!(":grin:", "😁", ["grin"]);
        add_emoji!(":laugh:", "😆", ["laugh", "lol"]);
        add_emoji!(":joy:", "😂", ["joy", "lmao", "rofl"]);
        add_emoji!(":rofl:", "🤣");
        add_emoji!(":sweat_smile:", "😅");
        add_emoji!(":wink:", "😉", ["wink"]);
        add_emoji!(":heart_eyes:", "😍", ["love", "heart_eyes"]);
        add_emoji!(":kissing_heart:", "😘", ["kiss"]);
        add_emoji!(":yum:", "😋", ["yum", "delicious"]);
        add_emoji!(":sunglasses:", "😎", ["cool", "sunglasses"]);
        add_emoji!(":cool:", "😎");
        add_emoji!(":star_struck:", "🤩");
        add_emoji!(":thinking:", "🤔", ["think", "thinking"]);
        add_emoji!(":shrug:", "🤷", ["shrug"]);
        add_emoji!(":neutral_face:", "😐");
        add_emoji!(":expressionless:", "😑");
        add_emoji!(":rolling_eyes:", "🙄");
        add_emoji!(":grimacing:", "😬");
        add_emoji!(":relieved:", "😌");
        add_emoji!(":pensive:", "😔");
        add_emoji!(":sleepy:", "😪");
        add_emoji!(":sleeping:", "😴", ["sleep"]);
        add_emoji!(":mask:", "😷");
        add_emoji!(":hot_face:", "🥵");
        add_emoji!(":cold_face:", "🥶");
        add_emoji!(":woozy_face:", "🥴");
        add_emoji!(":dizzy_face:", "😵");
        add_emoji!(":exploding_head:", "🤯", ["mindblown"]);
        add_emoji!(":mind_blown:", "🤯");
        add_emoji!(":partying_face:", "🥳", ["party"]);
        add_emoji!(":party:", "🥳");
        add_emoji!(":pleading_face:", "🥺", ["please"]);
        add_emoji!(":sob:", "😭", ["sob"]);
        add_emoji!(":scream:", "😱");
        add_emoji!(":angry:", "😠", ["angry"]);
        add_emoji!(":rage:", "😡", ["rage"]);
        add_emoji!(":skull:", "💀", ["dead", "skull"]);
        add_emoji!(":poop:", "💩");
        add_emoji!(":clown:", "🤡");
        add_emoji!(":ghost:", "👻", ["bhoot", "ghost"]);
        add_emoji!(":alien:", "👽");
        add_emoji!(":robot:", "🤖");

        // Hands & Gestures
        add_emoji!(":handshake:", "🤝");
        add_emoji!(":pray:", "🙏");
        add_emoji!(":clap:", "👏", ["clap", "applaud"]);
        add_emoji!(":raised_hands:", "🙌");
        add_emoji!(":thumbsup:", "👍", ["thumbsup", "like", "yes"]);
        add_emoji!(":thumbsdown:", "👎", ["thumbsdown", "dislike", "no"]);
        add_emoji!(":punch:", "👊");
        add_emoji!(":fist:", "✊");
        add_emoji!(":v:", "✌️", ["peace", "victory"]);
        add_emoji!(":love_you_gesture:", "🤟");
        add_emoji!(":metal:", "🤘");
        add_emoji!(":ok_hand:", "👌", ["ok"]);
        add_emoji!(":pinching_hand:", "🤏");
        add_emoji!(":wave:", "👋", ["hi", "hello", "wave", "bye"]);
        add_emoji!(":muscle:", "💪", ["strong", "muscle", "flex"]);

        // Hearts
        add_emoji!(":heart:", "❤️");
        add_emoji!(":red_heart:", "❤️");
        add_emoji!(":orange_heart:", "🧡");
        add_emoji!(":yellow_heart:", "💛");
        add_emoji!(":green_heart:", "💚");
        add_emoji!(":blue_heart:", "💙");
        add_emoji!(":purple_heart:", "💜");
        add_emoji!(":black_heart:", "🖤");
        add_emoji!(":broken_heart:", "💔");
        add_emoji!(":sparkles:", "✨", ["sparkle", "magic"]);
        add_emoji!(":star:", "⭐");
        add_emoji!(":star2:", "🌟");
        add_emoji!(":fire:", "🔥");
        add_emoji!(":100:", "💯");
        add_emoji!(":collision:", "💥");
        add_emoji!(":boom:", "💥");
        add_emoji!(":bulb:", "💡");
        add_emoji!(":idea:", "💡");
        add_emoji!(":check:", "✔️");
        add_emoji!(":white_check_mark:", "✅");
        add_emoji!(":x:", "❌");
        add_emoji!(":warning:", "⚠️");
        add_emoji!(":rocket:", "🚀", ["rocket", "launch", "fast"]);
        add_emoji!(":tada:", "🎉", ["tada", "congrats"]);
        add_emoji!(":confetti:", "🎊");
        add_emoji!(":balloon:", "🎈");
        add_emoji!(":gift:", "🎁", ["gift", "present"]);
        add_emoji!(":trophy:", "🏆", ["winner", "trophy"]);
        add_emoji!(":medal:", "🏅");
        add_emoji!(":gem:", "💎", ["diamond", "gem"]);
        add_emoji!(":money:", "💰", ["money", "rich"]);
        add_emoji!(":coffee:", "☕");
        add_emoji!(":tea:", "🍵");
        add_emoji!(":pizza:", "🍕");
        add_emoji!(":cake:", "🍰", ["cake", "birthday"]);
        add_emoji!(":burger:", "🍔");
        add_emoji!(":apple:", "🍎");
        add_emoji!(":mango:", "🥭");
        add_emoji!(":sun:", "☀️");
        add_emoji!(":moon:", "🌙");
        add_emoji!(":zap:", "⚡");
        add_emoji!(":lightning:", "⚡");
        add_emoji!(":rain:", "🌧️");
        add_emoji!(":snowflake:", "❄️");
        add_emoji!(":rainbow:", "🌈");
        add_emoji!(":cat:", "🐱");
        add_emoji!(":dog:", "🐶");
        add_emoji!(":bird:", "🐦");
        add_emoji!(":fish:", "🐟");
        add_emoji!(":car:", "🚗");
        add_emoji!(":bus:", "🚌");
        add_emoji!(":train:", "🚆");
        add_emoji!(":plane:", "✈️");
        add_emoji!(":computer:", "💻");
        add_emoji!(":laptop:", "💻");
        add_emoji!(":phone:", "📱");
        add_emoji!(":mobile:", "📱");
        add_emoji!(":code:", "💻");
        add_emoji!(":lock:", "🔒");
        add_emoji!(":key:", "🔑");
        add_emoji!(":book:", "📖");
        add_emoji!(":email:", "📧");
        add_emoji!(":bell:", "🔔");

        // 4. Standard ASCII Emoticons
        add_emoji!(":)", "😊");
        add_emoji!(":-)", "😊");
        add_emoji!(";)", "😉");
        add_emoji!(";-)", "😉");
        add_emoji!(":D", "😃");
        add_emoji!(":-D", "😃");
        add_emoji!(":P", "😋");
        add_emoji!(":-P", "😋");
        add_emoji!("<3", "❤️");
        add_emoji!("</3", "💔");
        add_emoji!(":(", "🙁");
        add_emoji!(":-(", "🙁");
        add_emoji!(":'(", "😢");
        add_emoji!(":-O", "😮");
        add_emoji!(":O", "😮");
        add_emoji!("-_-", "😑");
        add_emoji!("O:)", "😇");
        add_emoji!("B-)", "😎");

        let mut sorted_entries: Vec<(String, String)> = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        sorted_entries.sort_by(|a, b| a.0.cmp(&b.0));

        Self {
            map,
            sorted_entries,
            keywords,
        }
    }

    /// Look up exact match or normalized shortcode match
    pub fn lookup(&self, code: &str) -> Option<&String> {
        if let Some(val) = self.map.get(code) {
            return Some(val);
        }

        // Only wrap with colons if the input explicitly starts with ':' (e.g. ":smile" -> ":smile:")
        if code.starts_with(':') && !code.ends_with(':') {
            let with_end_colon = format!("{}:", code);
            if let Some(val) = self.map.get(&with_end_colon) {
                return Some(val);
            }
        }

        if code.starts_with('*') && !code.ends_with('*') {
            let with_end_star = format!("{}*", code);
            if let Some(val) = self.map.get(&with_end_star) {
                return Some(val);
            }
        }

        None
    }

    /// Lookup emojis associated with a word keyword (e.g. "bhalobasha" -> ["❤️"], "cha" -> ["☕"])
    pub fn lookup_by_keyword(&self, word: &str) -> Vec<String> {
        let clean = word.to_lowercase();
        if let Some(list) = self.keywords.get(&clean) {
            return list.clone();
        }
        Vec::new()
    }

    /// Prefix search for live typing autocompletion (e.g. ":sm" -> "😊", "*t" -> "৳")
    pub fn search_prefix(&self, query: &str, limit: usize) -> Vec<String> {
        let q = query.to_lowercase();
        let mut results = Vec::new();

        // 1. Direct prefix matches
        for (key, val) in &self.sorted_entries {
            if key.starts_with(&q) {
                if !results.contains(val) {
                    results.push(val.clone());
                    if results.len() >= limit {
                        return results;
                    }
                }
            }
        }

        // 2. Shortcode stem matches if query starts with ':' or '*'
        if q.len() >= 2 && (q.starts_with(':') || q.starts_with('*')) {
            let stem = &q[1..];
            for (key, val) in &self.sorted_entries {
                let key_inner = key.trim_matches(|c| c == ':' || c == '*');
                if key_inner.starts_with(stem) {
                    if !results.contains(val) {
                        results.push(val.clone());
                        if results.len() >= limit {
                            return results;
                        }
                    }
                }
            }
        }

        // 3. Keyword matches if not starting with special characters
        if !q.starts_with(':') && !q.starts_with('*') && q.len() >= 2 {
            for (kw, emoji_list) in &self.keywords {
                if kw.starts_with(&q) {
                    for em in emoji_list {
                        if !results.contains(em) {
                            results.push(em.clone());
                            if results.len() >= limit {
                                return results;
                            }
                        }
                    }
                }
            }
        }

        results
    }
}
