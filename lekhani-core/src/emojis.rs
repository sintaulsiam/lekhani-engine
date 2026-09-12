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
            ($code:expr, $emoji:expr, [$($kw:expr),* $(,)?]) => {
                map.insert($code.to_string(), $emoji.to_string());
                $(
                    let kw_norm = $kw.to_lowercase();
                    let list = keywords.entry(kw_norm).or_default();
                    if !list.contains(&$emoji.to_string()) {
                        list.push($emoji.to_string());
                    }
                )*
            };
            ($code:expr, $emoji:expr) => {
                map.insert($code.to_string(), $emoji.to_string());
                let clean_code = $code.trim_matches(|c| c == ':' || c == '*').to_lowercase();
                if !clean_code.is_empty() {
                    let list = keywords.entry(clean_code).or_default();
                    if !list.contains(&$emoji.to_string()) {
                        list.push($emoji.to_string());
                    }
                }
            };
        }

        // ==========================================
        // 1. Bengali Symbols & Native Punctuation
        // ==========================================
        add_emoji!("*taka*", "৳", ["taka", "tk", "poisha", "টাকা", "টাকায়", "পয়সা", "পয়সা"]);
        add_emoji!("*tk*", "৳", ["taka", "tk", "টাকা"]);
        add_emoji!("$taka", "৳");
        add_emoji!("$tk", "৳");
        add_emoji!("$$", "৳");
        add_emoji!("*dari*", "।", ["dari", "darhi", "দাড়ি", "দাড়ি"]);
        add_emoji!("*ddari*", "॥", ["ddari", "ডাবলদাড়ি", "দ্বিদাড়ি"]);
        add_emoji!(
            "*hasant*",
            "্",
            ["hasant", "hasanta", "hosonto", "hoshonto", "halant", "virama", "হসন্ত"]
        );
        add_emoji!("*hosonto*", "্");
        add_emoji!("*hoshonto*", "্");
        add_emoji!("*hasanta*", "্");
        add_emoji!("*halant*", "্");
        add_emoji!("*virama*", "্");
        add_emoji!("*khandatta*", "ৎ", ["khandatta", "khandata", "খন্ডত", "খণ্ডত"]);
        add_emoji!("*khandata*", "ৎ");
        add_emoji!("*khanda_ta*", "ৎ");
        add_emoji!("*khandota*", "ৎ");
        add_emoji!(
            "*anusvara*",
            "ং",
            ["anusvara", "anushbar", "onushbar", "onushor", "anushar", "অনুস্বার"]
        );
        add_emoji!("*anusvar*", "ং");
        add_emoji!("*anushbar*", "ং");
        add_emoji!("*onushbar*", "ং");
        add_emoji!("*onushor*", "ং");
        add_emoji!("*anushar*", "ং");
        add_emoji!(
            "*visarga*",
            "ঃ",
            ["visarga", "bishorgo", "bisharga", "bisarga", "বিসর্গ"]
        );
        add_emoji!("*visarg*", "ঃ");
        add_emoji!("*bishorgo*", "ঃ");
        add_emoji!("*bisharga*", "ঃ");
        add_emoji!("*bisarga*", "ঃ");
        add_emoji!(
            "*candrabindu*",
            "ঁ",
            ["candrabindu", "chandrabindu", "chondrobindu", "চন্দ্রবিন্দু"]
        );
        add_emoji!("*chandrabindu*", "ঁ");
        add_emoji!("*chondrobindu*", "ঁ");
        add_emoji!("*chandra*", "ঁ");
        add_emoji!("*candra*", "ঁ");
        add_emoji!("*avagraha*", "ঽ", ["avagraha", "অবগ্রহ"]);
        add_emoji!("*isshar*", "৺", ["isshar", "ishwar", "iswar", "ঈশ্বর"]);
        add_emoji!("*ishwar*", "৺");
        add_emoji!("*iswar*", "৺");
        add_emoji!("*ganda*", "𑜰");

        // ==========================================
        // 2. Bengali Named Words & Cultural Expressions
        // ==========================================
        add_emoji!(
            ":bhalobasha:",
            "❤️",
            [
                "bhalobasha", "valobasha", "bhalobasa", "valobasa", "bhalobasi",
                "ভালোবাসা", "ভালবাসা", "ভালোবাসি", "ভালবাসি", "প্রেম",
                "love", "heart", "romance", "affection"
            ]
        );
        add_emoji!(":valobasha:", "❤️");
        add_emoji!(":prem:", "💖", ["prem", "প্রেম", "crush", "হৃদয়", "crushing"]);
        add_emoji!(
            ":pranam:",
            "🙏",
            [
                "pranam", "প্রণাম", "adab", "আদাব", "namaskar", "নমস্কার",
                "doya", "দোয়া", "দোয়া", "munajat", "মোনাজাত", "pray", "prayer", "please"
            ]
        );
        add_emoji!(":salam:", "🤝", ["salam", "সালাম", "handshake", "agreement", "চুক্তি", "হ্যান্ডশেক"]);
        add_emoji!(":adab:", "🙏");
        add_emoji!(":namaskar:", "🙏");
        add_emoji!(":shurjo:", "☀️", ["shurjo", "surjo", "সূর্য", "রোদ", "sun", "sunny", "sunshine"]);
        add_emoji!(":surjo:", "☀️");
        add_emoji!(":chad:", "🌙", ["chad", "chand", "চাঁদ", "জোছনা", "moon", "night"]);
        add_emoji!(":tara:", "⭐", ["tara", "তারা", "নক্ষত্র", "star"]);
        add_emoji!(":ful:", "🌸", ["ful", "phul", "ফুল", "flower", "blossom", "cherry"]);
        add_emoji!(":golap:", "🌹", ["golap", "গোলাপ", "rose"]);
        add_emoji!(":padma:", "🪷", ["padma", "পদ্ম", "lotus", "waterlily"]);
        add_emoji!(":shapla:", "🪷", ["shapla", "শাপলা"]);
        add_emoji!(":agun:", "🔥", ["agun", "আগুন", "fire", "hot", "flame", "lit"]);
        add_emoji!(":aguni:", "🔥");
        add_emoji!(":hasukhi:", "😊", ["hasukhi", "khushi", "খুশি", "আনন্দ", "happy", "blessed"]);
        add_emoji!(":hashi:", "😄", ["hashi", "hasi", "হাসি", "হাস্য", "smile", "laugh", "happy"]);
        add_emoji!(
            ":kanna:",
            "😭",
            ["kanna", "কান্না", "dukkho", "দুঃখ", "কষ্ট", "cry", "sob", "sad", "tears"]
        );
        add_emoji!(":thik:", "✅", ["thik", "ঠিক", "shothik", "সঠিক", "yes", "correct", "check", "done"]);
        add_emoji!(":bhul:", "❌", ["bhul", "ভুল", "অশুদ্ধ", "no", "wrong", "cross", "incorrect"]);
        add_emoji!(
            ":shabash:",
            "👍",
            [
                "shabash", "sabash", "সাবাশ", "bhalo", "ভালো", "সেরা",
                "like", "good", "thumbsup", "agree", "upvote"
            ]
        );
        add_emoji!(
            ":durbar:",
            "⚡",
            [
                "durbar", "duronto", "দুর্বার", "bijli", "বিজলী", "বিদ্যুৎ",
                "zap", "flash", "lightning", "power", "energy"
            ]
        );
        add_emoji!(":bijli:", "⚡");
        add_emoji!(":chokh:", "👀", ["chokh", "চোখ", "নজর", "দৃষ্টি", "eyes", "look", "see", "watch"]);
        add_emoji!(":machh:", "🐟", ["mach", "machh", "মাছ", "fish", "sea"]);
        add_emoji!(":ilish:", "🐟", ["ilish", "ইলিশ", "hilsa"]);
        add_emoji!(":pakhi:", "🐦", ["pakhi", "পাখি", "bird"]);
        add_emoji!(":doyel:", "🐦", ["doyel", "দোয়েল", "দোয়েল", "magpie"]);
        add_emoji!(":bagh:", "🐅", ["bagh", "বাঘ", "tiger", "royal"]);
        add_emoji!(":singho:", "🦁", ["singho", "সিংহ", "lion"]);
        add_emoji!(":shap:", "🐍", ["shap", "সাপ", "snake"]);
        add_emoji!(":gach:", "🌳", ["gach", "গাছ", "বৃক্ষ", "tree", "nature", "forest"]);
        add_emoji!(":brishti:", "🌧️", ["brishti", "বৃষ্টি", "বর্ষা", "rain", "rainy", "shower"]);
        add_emoji!(":megh:", "☁️", ["megh", "মেঘ", "আকাশ", "cloud", "cloudy"]);
        add_emoji!(":jhor:", "🌪️", ["jhor", "ঝড়", "ঝড়", "তুফান", "storm", "cyclone", "tornado"]);
        add_emoji!(":cha:", "☕", ["cha", "চা", "tea", "coffee", "কফি", "hot", "cuppa"]);
        add_emoji!(":pani:", "💧", ["pani", "পানি", "jol", "জল", "water", "drop", "liquid"]);
        add_emoji!(":vath:", "🍚", ["bhat", "vath", "ভাত", "rice", "meal"]);
        add_emoji!(":biryani:", "🍲", ["biryani", "বিরিয়ানি", "বিরিয়ানি", "খাবার", "food", "dish"]);
        add_emoji!(":dudh:", "🥛", ["dudh", "দুধ", "milk", "drink"]);
        add_emoji!(":mishti:", "🍬", ["mishti", "মিষ্টি", "sweet", "candy", "dessert"]);
        add_emoji!(":boishakh:", "🎭", ["boishakh", "বৈশাখ", "mela", "মেলা", "উৎসব", "drama", "theatre"]);
        add_emoji!(
            ":potaka:",
            "🇧🇩",
            ["potaka", "পতাকা", "bangladesh", "বাংলাদেশ", "bd", "flag_bd", "দেশ"]
        );
        add_emoji!(":bangladesh:", "🇧🇩");
        add_emoji!(":bd:", "🇧🇩");
        add_emoji!(":biral:", "🐱", ["biral", "বিড়াল", "বিড়াল", "বেড়াল", "cat", "kitty", "kitten"]);
        add_emoji!(":kukur:", "🐶", ["kukur", "কুকুর", "kutta", "dog", "puppy"]);
        add_emoji!(":gari:", "🚗", ["gari", "গাড়ি", "গাড়ি", "car", "drive", "vehicle"]);
        add_emoji!(":rickshaw:", "🛺", ["rickshaw", "রিকশা", "রিক্সা", "auto", "cng"]);
        add_emoji!(":nouka:", "⛵", ["nouka", "নৌকা", "boat", "sail"]);
        add_emoji!(":boi:", "📖", ["boi", "বই", "কিতাব", "book", "read", "study"]);
        add_emoji!(":kolom:", "🖊️", ["kolom", "কলম", "pen", "write", "author"]);
        add_emoji!(":pencil:", "✏️", ["pencil", "পেন্সিল", "আঁকা"]);
        add_emoji!(":bari:", "🏠", ["bari", "বাড়ি", "বাড়ি", "ঘর", "home", "house"]);
        add_emoji!(":alo:", "💡", ["alo", "আলো", "bulb", "idea", "light", "আইডিয়া"]);
        add_emoji!(":ghori:", "⏰", ["ghori", "ঘড়ি", "ঘড়ি", "সময়", "shomoy", "clock", "time", "alarm"]);
        add_emoji!(":phone:", "📱", ["phone", "ফোন", "mobile", "মোবাইল", "স্মার্টফোন"]);
        add_emoji!(":laptop:", "💻", ["laptop", "computer", "ল্যাপটপ", "কম্পিউটার", "code", "কোড", "প্রোগ্রাম"]);
        add_emoji!(
            ":daktar:",
            "🩺",
            ["daktar", "ডাক্তার", "doctor", "hospital", "হাসপাতাল", "চিকিৎসক", "ঔষধ"]
        );
        add_emoji!(":shikkhok:", "🧑‍🏫", ["shikkhok", "শিক্ষক", "teacher", "স্যার", "ম্যাডাম"]);
        add_emoji!(":khela:", "⚽", ["khela", "খেলা", "football", "soccer", "ফুটবল"]);
        add_emoji!(":cricket:", "🏏", ["cricket", "ক্রিকেট", "ব্যাট"]);
        add_emoji!(":gaan:", "🎵", ["gan", "gaan", "গান", "সঙ্গীত", "music", "song", "audio"]);
        add_emoji!(":nach:", "💃", ["nach", "নাচ", "নৃত্য", "dance"]);
        add_emoji!(":bhoot:", "👻", ["bhoot", "ভুত", "ভূত", "ghost", "scary"]);

        // ==========================================
        // 3. Smileys & Emotions
        // ==========================================
        add_emoji!(":smile:", "😊", ["smile", "happy", "হাসি", "খুশি"]);
        add_emoji!(":smiley:", "😃", ["smiley", "happy", "joy"]);
        add_emoji!(":grin:", "😁", ["grin", "beaming"]);
        add_emoji!(":laugh:", "😆", ["laugh", "lol", "haha"]);
        add_emoji!(":joy:", "😂", ["joy", "lmao", "rofl", "হাসি", "মজা"]);
        add_emoji!(":rofl:", "🤣", ["rofl", "rolling", "haha"]);
        add_emoji!(":sweat_smile:", "😅", ["sweat_smile", "relief"]);
        add_emoji!(":wink:", "😉", ["wink", "flirt", "ইশারা"]);
        add_emoji!(":blush:", "😊", ["blush", "shy", "লজ্জা"]);
        add_emoji!(":heart_eyes:", "😍", ["love", "heart_eyes", "প্রেম", "ক্রাশ"]);
        add_emoji!(":kissing_heart:", "😘", ["kiss", "kissing", "চুমু"]);
        add_emoji!(":yum:", "😋", ["yum", "delicious", "yummy", "স্বাদ", "মজাদার"]);
        add_emoji!(":sunglasses:", "😎", ["cool", "sunglasses", "স্মার্ট", "চশমা"]);
        add_emoji!(":cool:", "😎", ["cool", "boss", "style"]);
        add_emoji!(":star_struck:", "🤩", ["star_struck", "excited", "উত্তেজিত"]);
        add_emoji!(":thinking:", "🤔", ["think", "thinking", "ভাবনা", "চিন্তা"]);
        add_emoji!(":shrug:", "🤷", ["shrug", "dunno", "জানি_না"]);
        add_emoji!(":neutral_face:", "😐", ["neutral", "meh", "নীরব"]);
        add_emoji!(":expressionless:", "😑", ["expressionless"]);
        add_emoji!(":rolling_eyes:", "🙄", ["rolling_eyes", "whatever", "বিরক্ত"]);
        add_emoji!(":grimacing:", "😬", ["grimacing", "awkward"]);
        add_emoji!(":relieved:", "😌", ["relieved", "peaceful", "শান্তি"]);
        add_emoji!(":pensive:", "😔", ["pensive", "sad", "বিষণ্ণ"]);
        add_emoji!(":sleepy:", "😪", ["sleepy", "tired", "ক্লান্ত"]);
        add_emoji!(":sleeping:", "😴", ["sleep", "sleeping", "ঘুম"]);
        add_emoji!(":mask:", "😷", ["mask", "sick", "অসুস্থ"]);
        add_emoji!(":hot_face:", "🥵", ["hot", "hot_face", "গরম"]);
        add_emoji!(":cold_face:", "🥶", ["cold", "cold_face", "ঠান্ডা", "শীত"]);
        add_emoji!(":woozy_face:", "🥴", ["woozy", "tipsy"]);
        add_emoji!(":dizzy_face:", "😵", ["dizzy", "unconscious"]);
        add_emoji!(":exploding_head:", "🤯", ["mindblown", "mind_blown", "বিস্ফোরণ"]);
        add_emoji!(":mind_blown:", "🤯");
        add_emoji!(":partying_face:", "🥳", ["party", "partying", "celebrate", "উৎসব"]);
        add_emoji!(":party:", "🥳");
        add_emoji!(":pleading_face:", "🥺", ["please", "pleading", "মিনতি"]);
        add_emoji!(":sob:", "😭", ["sob", "crying", "কান্না"]);
        add_emoji!(":scream:", "😱", ["scream", "scared", "ভয়", "আতঙ্ক"]);
        add_emoji!(":angry:", "😠", ["angry", "mad", "রাগ"]);
        add_emoji!(":rage:", "😡", ["rage", "furious", "গোস্বা"]);
        add_emoji!(":skull:", "💀", ["dead", "skull", "কঙ্কাল", "মৃত"]);
        add_emoji!(":poop:", "💩", ["poop", "shit"]);
        add_emoji!(":clown:", "🤡", ["clown", "জোকার"]);
        add_emoji!(":ghost:", "👻", ["ghost", "bhoot", "ভূত"]);
        add_emoji!(":alien:", "👽", ["alien", "এলিয়েন"]);
        add_emoji!(":robot:", "🤖", ["robot", "রোবট", "এআই"]);

        // ==========================================
        // 4. Hands & Gestures
        // ==========================================
        add_emoji!(":handshake:", "🤝", ["handshake", "deal", "সালাম", "চুক্তি"]);
        add_emoji!(":pray:", "🙏", ["pray", "please", "namaste", "প্রণাম", "দোয়া"]);
        add_emoji!(":clap:", "👏", ["clap", "applaud", "তালি", "অভিনন্দন"]);
        add_emoji!(":raised_hands:", "🙌", ["raised_hands", "hooray", "জয়"]);
        add_emoji!(":thumbsup:", "👍", ["thumbsup", "like", "yes", "good", "ভালো"]);
        add_emoji!(":thumbsdown:", "👎", ["thumbsdown", "dislike", "no", "bad", "খারাপ"]);
        add_emoji!(":punch:", "👊", ["punch", "fist_bump"]);
        add_emoji!(":fist:", "✊", ["fist", "power", "শক্তি"]);
        add_emoji!(":v:", "✌️", ["peace", "victory", "শান্তি", "জয়"]);
        add_emoji!(":love_you_gesture:", "🤟", ["love_you", "ily"]);
        add_emoji!(":metal:", "🤘", ["metal", "rock"]);
        add_emoji!(":ok_hand:", "👌", ["ok", "perfect", "ঠিক"]);
        add_emoji!(":pinching_hand:", "🤏", ["pinch", "little", "একটু"]);
        add_emoji!(":wave:", "👋", ["wave", "hi", "hello", "bye", "সালাম", "বিদায়"]);
        add_emoji!(":muscle:", "💪", ["muscle", "strong", "flex", "শক্তি", "বলবান"]);

        // ==========================================
        // 5. Hearts & Colors
        // ==========================================
        add_emoji!(":heart:", "❤️", ["heart", "love", "red_heart", "ভালোবাসা"]);
        add_emoji!(":red_heart:", "❤️", ["red_heart"]);
        add_emoji!(":orange_heart:", "🧡", ["orange_heart"]);
        add_emoji!(":yellow_heart:", "💛", ["yellow_heart"]);
        add_emoji!(":green_heart:", "💚", ["green_heart"]);
        add_emoji!(":blue_heart:", "💙", ["blue_heart"]);
        add_emoji!(":purple_heart:", "💜", ["purple_heart"]);
        add_emoji!(":black_heart:", "🖤", ["black_heart"]);
        add_emoji!(":white_heart:", "🤍", ["white_heart"]);
        add_emoji!(":broken_heart:", "💔", ["broken_heart", "heartbreak", "ভাঙাহৃদয়"]);
        add_emoji!(":sparkling_heart:", "💖", ["sparkling_heart", "love"]);
        add_emoji!(":sparkles:", "✨", ["sparkles", "magic", "জ্যোতি", "চকচক"]);
        add_emoji!(":star:", "⭐", ["star", "তারা"]);
        add_emoji!(":star2:", "🌟", ["star2", "glowing_star"]);
        add_emoji!(":fire:", "🔥", ["fire", "flame", "hot", "lit", "আগুন"]);
        add_emoji!(":100:", "💯", ["100", "perfect", "score", "শতভাগ"]);
        add_emoji!(":collision:", "💥", ["boom", "collision", "blast", "বিস্ফোরণ"]);
        add_emoji!(":boom:", "💥");
        add_emoji!(":bulb:", "💡", ["bulb", "idea", "আলো"]);
        add_emoji!(":idea:", "💡");
        add_emoji!(":check:", "✔️", ["check", "done"]);
        add_emoji!(":white_check_mark:", "✅", ["check_mark", "correct", "সঠিক"]);
        add_emoji!(":x:", "❌", ["cross", "wrong", "ভুল"]);
        add_emoji!(":warning:", "⚠️", ["warning", "alert", "সতর্ক"]);
        add_emoji!(":rocket:", "🚀", ["rocket", "launch", "fast", "রকেট", "দ্রুত"]);
        add_emoji!(":tada:", "🎉", ["tada", "congrats", "party", "অভিনন্দন", "উৎসব"]);
        add_emoji!(":confetti:", "🎊", ["confetti"]);
        add_emoji!(":balloon:", "🎈", ["balloon", "বেলুন"]);
        add_emoji!(":gift:", "🎁", ["gift", "present", "উপহার"]);
        add_emoji!(":trophy:", "🏆", ["trophy", "winner", "champion", "ট্রফি", "বিজয়ী"]);
        add_emoji!(":medal:", "🏅", ["medal", "award", "পদক"]);
        add_emoji!(":gem:", "💎", ["gem", "diamond", "হীরা", "রত্ন"]);
        add_emoji!(":money:", "💰", ["money", "rich", "cash", "টাকা", "ধন"]);

        // ==========================================
        // 6. Food & Drink
        // ==========================================
        add_emoji!(":coffee:", "☕", ["coffee", "tea", "কফি", "চা"]);
        add_emoji!(":tea:", "🍵", ["tea", "green_tea", "চা"]);
        add_emoji!(":milk:", "🥛", ["milk", "দুধ"]);
        add_emoji!(":water:", "💧", ["water", "পানি", "জল"]);
        add_emoji!(":rice:", "🍚", ["rice", "ভাত"]);
        add_emoji!(":pizza:", "🍕", ["pizza", "পিজ্জা", "fastfood"]);
        add_emoji!(":burger:", "🍔", ["burger", "hamburger", "বার্গার"]);
        add_emoji!(":fries:", "🍟", ["fries", "chips", "ফ্রেঞ্চফ্রাই"]);
        add_emoji!(":sandwich:", "🥪", ["sandwich", "স্যান্ডউইচ"]);
        add_emoji!(":chicken:", "🍗", ["chicken", "meat", "মুরগি", "মাংস"]);
        add_emoji!(":cake:", "🍰", ["cake", "birthday", "কেক"]);
        add_emoji!(":birthday_cake:", "🎂", ["birthday", "cake", "জন্মদিন"]);
        add_emoji!(":chocolate:", "🍫", ["chocolate", "চকলেট"]);
        add_emoji!(":candy:", "🍬", ["candy", "sweet", "মিষ্টি"]);
        add_emoji!(":ice_cream:", "🍦", ["icecream", "ice_cream", "আইসক্রিম"]);
        add_emoji!(":apple:", "🍎", ["apple", "আপেল", "ফল"]);
        add_emoji!(":mango:", "🥭", ["mango", "আম", "আম্র"]);
        add_emoji!(":banana:", "🍌", ["banana", "কলা"]);
        add_emoji!(":watermelon:", "🍉", ["watermelon", "তরমুজ"]);
        add_emoji!(":grapes:", "🍇", ["grapes", "আঙ্গুর"]);
        add_emoji!(":egg:", "🥚", ["egg", "ডিম"]);
        add_emoji!(":bread:", "🍞", ["bread", "ruti", "রুটি"]);

        // ==========================================
        // 7. Animals & Nature
        // ==========================================
        add_emoji!(":cat:", "🐱", ["cat", "kitty", "বিড়াল", "বিড়াল"]);
        add_emoji!(":dog:", "🐶", ["dog", "puppy", "কুকুর"]);
        add_emoji!(":tiger:", "🐅", ["tiger", "বাঘ"]);
        add_emoji!(":lion:", "🦁", ["lion", "সিংহ"]);
        add_emoji!(":bear:", "🐻", ["bear", "ভাল্লুক"]);
        add_emoji!(":panda:", "🐼", ["panda", "পান্ডা"]);
        add_emoji!(":elephant:", "🐘", ["elephant", "হাতি"]);
        add_emoji!(":monkey:", "🐵", ["monkey", "বানর"]);
        add_emoji!(":horse:", "🐴", ["horse", "ঘোড়া", "ঘোড়া"]);
        add_emoji!(":cow:", "🐮", ["cow", "গরু"]);
        add_emoji!(":sheep:", "🐑", ["sheep", "ভেড়া"]);
        add_emoji!(":goat:", "🐐", ["goat", "ছাগল"]);
        add_emoji!(":bird:", "🐦", ["bird", "পাখি"]);
        add_emoji!(":eagle:", "🦅", ["eagle", "চিল", "ঈগল"]);
        add_emoji!(":duck:", "🦆", ["duck", "হাঁস"]);
        add_emoji!(":owl:", "🦉", ["owl", "পেঁচা"]);
        add_emoji!(":fish:", "🐟", ["fish", "মাছ"]);
        add_emoji!(":shark:", "🦈", ["shark", "হাঙ্গর"]);
        add_emoji!(":snake:", "🐍", ["snake", "সাপ"]);
        add_emoji!(":butterfly:", "🦋", ["butterfly", "প্রজাপতি"]);
        add_emoji!(":bee:", "🐝", ["bee", "মৌমাছি"]);
        add_emoji!(":sun:", "☀️", ["sun", "সূর্য"]);
        add_emoji!(":moon:", "🌙", ["moon", "চাঁদ"]);
        add_emoji!(":tree:", "🌳", ["tree", "গাছ"]);
        add_emoji!(":flower:", "🌸", ["flower", "ফুল"]);
        add_emoji!(":rose:", "🌹", ["rose", "গোলাপ"]);
        add_emoji!(":rain:", "🌧️", ["rain", "বৃষ্টি"]);
        add_emoji!(":cloud:", "☁️", ["cloud", "মেঘ"]);
        add_emoji!(":zap:", "⚡", ["lightning", "electricity", "বিদ্যুৎ"]);
        add_emoji!(":snowflake:", "❄️", ["snow", "winter", "বরফ", "শীত"]);
        add_emoji!(":rainbow:", "🌈", ["rainbow", "রংধনু"]);

        // ==========================================
        // 8. Travel, Places & Vehicles
        // ==========================================
        add_emoji!(":car:", "🚗", ["car", "vehicle", "গাড়ি", "গাড়ি"]);
        add_emoji!(":taxi:", "🚕", ["taxi", "ট্যাক্সি"]);
        add_emoji!(":bus:", "🚌", ["bus", "বাস"]);
        add_emoji!(":train:", "🚆", ["train", "রেল", "ট্রেন"]);
        add_emoji!(":plane:", "✈️", ["plane", "airplane", "বিমান"]);
        add_emoji!(":motorcycle:", "🏍️", ["bike", "motorcycle", "মোটরসাইকেল"]);
        add_emoji!(":bicycle:", "🚲", ["bicycle", "cycle", "সাইকেল"]);
        add_emoji!(":boat:", "⛵", ["boat", "নৌকা"]);
        add_emoji!(":ship:", "🛳️", ["ship", "launch", "জাহাজ", "লঞ্চ"]);
        add_emoji!(":house:", "🏠", ["home", "house", "বাড়ি", "বাড়ি", "ঘর"]);
        add_emoji!(":hospital:", "🏥", ["hospital", "হাসপাতাল"]);
        add_emoji!(":school:", "🏫", ["school", "স্কুল", "বিদ্যালয়"]);
        add_emoji!(":mosque:", "🕌", ["mosque", "মসজিদ"]);
        add_emoji!(":temple:", "🛕", ["temple", "মন্দির"]);

        // ==========================================
        // 9. Tech, Tools, Objects & Flags
        // ==========================================
        add_emoji!(":computer:", "💻", ["computer", "pc", "কম্পিউটার"]);
        add_emoji!(":laptop_pc:", "💻", ["laptop", "ল্যাপটপ"]);
        add_emoji!(":code:", "💻", ["code", "programming", "কোড"]);
        add_emoji!(":lock:", "🔒", ["lock", "security", "তালা", "নিরাপত্তা"]);
        add_emoji!(":key:", "🔑", ["key", "password", "চাবি"]);
        add_emoji!(":book:", "📖", ["book", "read", "বই"]);
        add_emoji!(":email:", "📧", ["email", "mail", "ইমেইল"]);
        add_emoji!(":bell:", "🔔", ["bell", "notification", "ঘন্টা"]);
        add_emoji!(":camera:", "📷", ["camera", "photo", "ক্যামেরা"]);
        add_emoji!(":music:", "🎵", ["music", "song", "গান"]);
        add_emoji!(":guitar:", "🎸", ["guitar", "গিটার"]);
        add_emoji!(":game:", "🎮", ["game", "gaming", "গেম"]);
        add_emoji!(":flag_in:", "🇮🇳", ["india", "bharat", "ভারত"]);
        add_emoji!(":flag_pk:", "🇵🇰", ["pakistan", "পাকিস্তান"]);
        add_emoji!(":flag_sa:", "🇸🇦", ["saudi", "মক্কা", "সৌদি"]);
        add_emoji!(":flag_ps:", "🇵🇸", ["palestine", "ফিলিস্তিন"]);
        add_emoji!(":flag_us:", "🇺🇸", ["usa", "america", "আমেরিকা"]);
        add_emoji!(":flag_uk:", "🇬🇧", ["uk", "england", "ইংল্যান্ড"]);
        add_emoji!(":flag_ca:", "🇨🇦", ["canada", "কানাডা"]);
        add_emoji!(":flag_au:", "🇦🇺", ["australia", "অস্ট্রেলিয়া"]);
        add_emoji!(":flag_ar:", "🇦🇷", ["argentina", "আর্জেন্টিনা"]);
        add_emoji!(":flag_br:", "🇧🇷", ["brazil", "ব্রাজিল"]);
        add_emoji!(":flag_jp:", "🇯🇵", ["japan", "জাপান"]);
        add_emoji!(":flag_de:", "🇩🇪", ["germany", "জার্মানি"]);
        add_emoji!(":flag_fr:", "🇫🇷", ["france", "ফ্রান্স"]);

        // ==========================================
        // 10. Standard ASCII Emoticons
        // ==========================================
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

        let mut sorted_entries: Vec<(String, String)> =
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
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
        let clean = word.trim().to_lowercase();
        if clean.is_empty() {
            return Vec::new();
        }
        if let Some(list) = self.keywords.get(&clean) {
            return list.clone();
        }
        Vec::new()
    }

    /// Prefix search for live typing autocompletion (e.g. ":sm" -> "😊", "*t" -> "৳")
    pub fn search_prefix(&self, query: &str, limit: usize) -> Vec<String> {
        let q = query.to_lowercase();
        let mut results = Vec::new();

        // 1. Direct prefix matches on shortcode / symbol table
        for (key, val) in &self.sorted_entries {
            if key.starts_with(&q) && !results.contains(val) {
                results.push(val.clone());
                if results.len() >= limit {
                    return results;
                }
            }
        }

        // 2. Shortcode stem matches if query starts with ':' or '*'
        if q.len() >= 2 && (q.starts_with(':') || q.starts_with('*')) {
            let stem = &q[1..];
            for (key, val) in &self.sorted_entries {
                let key_inner = key.trim_matches(|c| c == ':' || c == '*');
                if key_inner.starts_with(stem) && !results.contains(val) {
                    results.push(val.clone());
                    if results.len() >= limit {
                        return results;
                    }
                }
            }
            // Also search keyword stems
            for (kw, emoji_list) in &self.keywords {
                if kw.starts_with(stem) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortcode_lookups() {
        let em = EmojiMap::new();
        assert_eq!(em.lookup(":cha:"), Some(&"☕".to_string()));
        assert_eq!(em.lookup(":cha"), Some(&"☕".to_string()));
        assert_eq!(em.lookup(":fire:"), Some(&"🔥".to_string()));
        assert_eq!(em.lookup(":bhalobasha:"), Some(&"❤️".to_string()));
        assert_eq!(em.lookup("*taka*"), Some(&"৳".to_string()));
        assert_eq!(em.lookup("$$"), Some(&"৳".to_string()));
        assert_eq!(em.lookup(":potaka:"), Some(&"🇧🇩".to_string()));
    }

    #[test]
    fn test_bilingual_keyword_lookups() {
        let em = EmojiMap::new();
        // English keywords
        assert!(em.lookup_by_keyword("tea").contains(&"☕".to_string()));
        assert!(em.lookup_by_keyword("coffee").contains(&"☕".to_string()));
        assert!(em.lookup_by_keyword("dog").contains(&"🐶".to_string()));
        assert!(em.lookup_by_keyword("cat").contains(&"🐱".to_string()));
        assert!(em.lookup_by_keyword("pizza").contains(&"🍕".to_string()));
        assert!(em.lookup_by_keyword("love").contains(&"❤️".to_string()));

        // Bengali phonetic keywords
        assert!(em.lookup_by_keyword("bhalobasha").contains(&"❤️".to_string()));
        assert!(em.lookup_by_keyword("cha").contains(&"☕".to_string()));
        assert!(em.lookup_by_keyword("kukur").contains(&"🐶".to_string()));
        assert!(em.lookup_by_keyword("biral").contains(&"🐱".to_string()));
        assert!(em.lookup_by_keyword("brishti").contains(&"🌧️".to_string()));

        // Bengali native script keywords
        assert!(em.lookup_by_keyword("ভালোবাসা").contains(&"❤️".to_string()));
        assert!(em.lookup_by_keyword("চা").contains(&"☕".to_string()));
        assert!(em.lookup_by_keyword("কুকুর").contains(&"🐶".to_string()));
        assert!(em.lookup_by_keyword("বিড়াল").contains(&"🐱".to_string()));
        assert!(em.lookup_by_keyword("বৃষ্টি").contains(&"🌧️".to_string()));
        assert!(em.lookup_by_keyword("টাকা").contains(&"৳".to_string()));
    }

    #[test]
    fn test_prefix_search() {
        let em = EmojiMap::new();
        let res = em.search_prefix(":sm", 5);
        assert!(!res.is_empty());
        assert!(res.contains(&"😊".to_string()));

        let res_bd = em.search_prefix(":pot", 5);
        assert!(res_bd.contains(&"🇧🇩".to_string()));
    }
}
