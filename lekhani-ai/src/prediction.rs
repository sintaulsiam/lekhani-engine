//! Zero-Preedit Next-Word Prediction Engine

use crate::lm::LanguageModel;
use crate::trainer::TrainedLanguageModelData;

#[derive(Debug, Clone, Default)]
pub struct NextWordPredictor {
    lm: LanguageModel,
}

pub const BENGALI_IDIOM_PHRASES: &[(&[&str], &[&str])] = &[
    (&["অনেক", "অনেক"], &["ধন্যবাদ", "শুভেচ্ছা ও অভিনন্দন"]),
    (&["অনেক"], &["ধন্যবাদ", "সুন্দর", "ভালো"]),
    (&["কেমন"], &["আছেন?", "আছো?", "হলো?"]),
    (&["শুভ"], &["সকাল", "সন্ধ্যা", "রাত্রি", "কামনা", "জন্মদিন", "নববর্ষ"]),
    (&["সব", "কিছু"], &["ঠিক আছে", "সুন্দর"]),
    (&["ঠিক"], &["আছে", "আছেন"]),
    (&["ইনশা"], &["আল্লাহ"]),
    (&["আলহামদু"], &["লিল্লাহ"]),
    (&["মাশা"], &["আল্লাহ"]),
    (&["খুব"], &["ভালো", "সুন্দর", "কষ্ট"]),
];

/// High-confidence formal / journalistic / academic / legal register phrase continuations.
///
/// These cover writing patterns heavily used in Prothom Alo editorials, government notices,
/// academic papers, and legal texts — vocabulary that the statistical n-gram LM
/// under-represents because training corpora skew toward conversational text.
///
/// Score: 0.9 (above any LM bigram) — these always lead the strip in matching contexts.
/// Zero runtime cost: static `&[(&[&str], &[&str])]`, no heap allocation.
pub const BENGALI_FORMAL_PHRASES: &[(&[&str], &[&str])] = &[
    // ── Discourse / Academic connectors ──────────────────────────────────────
    (&["প্রসঙ্গত"], &["উল্লেখ্য", "বলা যায়", "জানানো যাচ্ছে"]),
    (&["উল্লেখ্য"], &["যে", "এই", "তবে", "এখানে"]),
    (&["উল্লেখযোগ্য"], &["যে", "বিষয়", "তথ্য"]),
    (&["বিবেচনায়"], &["নেওয়া", "রাখা", "আসলে"]),
    (&["বিবেচনা"], &["করে", "করা", "করলে", "অনুযায়ী"]),
    (&["পরিপ্রেক্ষিতে"], &["বলা", "দেখা", "এই"]),
    (&["প্রেক্ষাপটে"], &["এই", "সেই", "বর্তমান"]),
    (&["সংক্ষেপে"], &["বলা", "বলতে", "এই"]),
    (&["তদুপরি"], &["বলা", "এটি", "এই"]),
    (&["অন্যদিকে"], &["বলা", "দেখা", "এই", "সরকার"]),
    (&["এতদসত্ত্বেও"], &["বলা", "এই"]),
    (&["পাশাপাশি"], &["এই", "তিনি", "সরকার"]),
    (&["তবে"], &["সরকার", "এই", "বলা", "জানানো"]),
    (&["যদিও"], &["এই", "তিনি", "বলা"]),
    (&["তথাপি"], &["এই", "বলা", "সরকার"]),
    (&["অর্থাৎ"], &["এই", "বলা", "তিনি"]),
    (&["যেমন"], &["এই", "তিনি", "বলা"]),
    (&["বিশেষত"], &["এই", "তিনি", "বলা"]),
    (&["মূলত"], &["এই", "তিনি", "বলা", "সরকার"]),
    (&["মূলতঃ"], &["এই", "তিনি", "বলা"]),
    (&["সামগ্রিকভাবে"], &["এই", "দেখা", "বলা"]),
    (&["সর্বোপরি"], &["এই", "বলা", "তিনি"]),
    (&["ফলে"], &["এই", "তিনি", "সরকার", "দেশ"]),
    (&["ফলস্বরূপ"], &["এই", "তিনি", "সরকার"]),
    (&["পরিণতিতে"], &["এই", "তিনি"]),
    // ── Government / Politics ─────────────────────────────────────────────
    (&["সরকার"], &["কর্তৃক", "পক্ষ থেকে", "ঘোষণা", "জানিয়েছে", "সিদ্ধান্ত"]),
    (&["সরকারি"], &["সিদ্ধান্ত", "নির্দেশ", "প্রতিষ্ঠান", "কর্মকর্তা"]),
    (&["মন্ত্রণালয়"], &["সূত্রে", "জানিয়েছে", "থেকে", "কর্তৃক"]),
    (&["মন্ত্রী"], &["বলেন", "জানান", "সভাপতিত্বে", "কর্তৃক"]),
    (&["প্রধানমন্ত্রী"], &["বলেন", "জানান", "নির্দেশ", "কার্যালয়"]),
    (&["রাষ্ট্রপতি"], &["বলেন", "জানান", "অনুমোদন", "স্বাক্ষর"]),
    (&["সংসদ"], &["অধিবেশন", "সদস্য", "পাস", "সভা", "কমিটি"]),
    (&["জাতীয়", "সংসদ"], &["অধিবেশন", "নির্বাচন", "সদস্য"]),
    (&["নির্বাচন"], &["কমিশন", "আয়োগ", "অনুষ্ঠিত", "পরিচালনা"]),
    (&["নির্বাচন", "কমিশন"], &["জানিয়েছে", "সিদ্ধান্ত", "ঘোষণা"]),
    (&["সরকারের"], &["পক্ষ", "কর্তৃপক্ষ", "সিদ্ধান্ত", "ঘোষণা"]),
    (&["কর্তৃপক্ষ"], &["জানিয়েছে", "সিদ্ধান্ত", "নির্দেশ", "বলেছে"]),
    (&["প্রশাসন"], &["জানিয়েছে", "সিদ্ধান্ত", "কর্তৃপক্ষ"]),
    // ── Judiciary / Legal ────────────────────────────────────────────────
    (&["আদালত"], &["রায়", "নির্দেশ", "আদেশ", "সূত্রে", "জানিয়েছে"]),
    (&["হাইকোর্ট"], &["রায়", "নির্দেশ", "আদেশ", "বেঞ্চ"]),
    (&["সুপ্রিম", "কোর্ট"], &["রায়", "নির্দেশ", "আদেশ"]),
    (&["রায়"], &["ঘোষণা", "দিয়েছেন", "বহাল", "পর্যবেক্ষণ"]),
    (&["আইন"], &["অনুযায়ী", "প্রণয়ন", "সংশোধন", "মন্ত্রণালয়"]),
    (&["মামলা"], &["দায়ের", "পরিচালনা", "নিষ্পত্তি", "খারিজ"]),
    (&["অভিযুক্ত"], &["ব্যক্তি", "আসামি", "পক্ষ"]),
    (&["বিচারক"], &["বলেন", "পর্যবেক্ষণ", "রায়"]),
    // ── Economy / Finance ────────────────────────────────────────────────
    (&["অর্থনীতি"], &["বিশেষজ্ঞ", "মন্দা", "প্রবৃদ্ধি", "উন্নয়ন"]),
    (&["বাজেট"], &["ঘোষণা", "প্রস্তাব", "অনুমোদন", "বরাদ্দ"]),
    (&["মূল্যস্ফীতি"], &["হ্রাস", "বৃদ্ধি", "নিয়ন্ত্রণ", "নির্ধারণ"]),
    (&["রপ্তানি"], &["আয়", "বৃদ্ধি", "হ্রাস", "লক্ষ্যমাত্রা"]),
    (&["আমদানি"], &["ব্যয়", "বৃদ্ধি", "হ্রাস", "নিয়ন্ত্রণ"]),
    (&["বিনিয়োগ"], &["বৃদ্ধি", "আকৃষ্ট", "পরিবেশ", "প্রস্তাব"]),
    (&["প্রবৃদ্ধি"], &["হার", "অর্জন", "লক্ষ্যমাত্রা"]),
    (&["জিডিপি"], &["প্রবৃদ্ধি", "হার", "অর্জন"]),
    (&["ব্যাংক"], &["ঋণ", "সুদ", "হার", "খেলাপি"]),
    (&["কেন্দ্রীয়", "ব্যাংক"], &["সুদ", "নির্দেশ", "হার"]),
    (&["বাংলাদেশ", "ব্যাংক"], &["সার্কুলার", "নির্দেশ", "রিজার্ভ"]),
    // ── Education / Academia ─────────────────────────────────────────────
    (&["বিশ্ববিদ্যালয়"], &["কর্তৃপক্ষ", "শিক্ষার্থী", "ক্যাম্পাস", "ভর্তি"]),
    (&["শিক্ষা"], &["মন্ত্রণালয়", "বোর্ড", "প্রতিষ্ঠান", "ব্যবস্থা"]),
    (&["পরীক্ষা"], &["ফলাফল", "কার্যক্রম", "পদ্ধতি", "নিয়ন্ত্রণ"]),
    (&["গবেষণা"], &["ফলাফল", "কার্যক্রম", "প্রতিষ্ঠান", "পদ্ধতি"]),
    (&["শিক্ষার্থী"], &["সংগঠন", "প্রতিবাদ", "আন্দোলন", "দাবি"]),
    (&["ভর্তি"], &["পরীক্ষা", "প্রক্রিয়া", "বিজ্ঞপ্তি", "কার্যক্রম"]),
    // ── Health / Medicine ────────────────────────────────────────────────
    (&["স্বাস্থ্য"], &["মন্ত্রণালয়", "অধিদপ্তর", "সেবা", "বিভাগ"]),
    (&["হাসপাতাল"], &["কর্তৃপক্ষ", "সূত্রে", "বলেছে", "বিভাগ"]),
    (&["রোগী"], &["সংখ্যা", "মৃত্যু", "সুস্থ", "ভর্তি"]),
    (&["চিকিৎসা"], &["সেবা", "পদ্ধতি", "ব্যবস্থা", "প্রদান"]),
    (&["ভ্যাকসিন"], &["কার্যক্রম", "প্রদান", "সরবরাহ"]),
    // ── Security / Military ──────────────────────────────────────────────
    (&["নিরাপত্তা"], &["বাহিনী", "ব্যবস্থা", "পরিষদ", "হুমকি"]),
    (&["সেনাবাহিনী"], &["অভিযান", "মোতায়েন", "সূত্রে"]),
    (&["পুলিশ"], &["অভিযান", "সূত্রে", "জানিয়েছে", "গ্রেপ্তার"]),
    (&["র্যাব"], &["অভিযান", "সূত্রে", "গ্রেপ্তার"]),
    (&["গ্রেপ্তার"], &["করা", "হয়েছে", "অভিযান", "ব্যক্তি"]),
    // ── Environment / Climate ────────────────────────────────────────────
    (&["পরিবেশ"], &["দূষণ", "সংরক্ষণ", "মন্ত্রণালয়", "অধিদপ্তর"]),
    (&["জলবায়ু"], &["পরিবর্তন", "সংকট", "চুক্তি", "প্রভাব"]),
    (&["বন্যা"], &["পরিস্থিতি", "ক্ষয়ক্ষতি", "ত্রাণ", "নিয়ন্ত্রণ"]),
    (&["ঘূর্ণিঝড়"], &["পূর্বাভাস", "আঘাত", "প্রস্তুতি"]),
    (&["দুর্যোগ"], &["ব্যবস্থাপনা", "পরিস্থিতি", "ত্রাণ"]),
    // ── International / Diplomacy ────────────────────────────────────────
    (&["আন্তর্জাতিক"], &["সম্প্রদায়", "মহল", "সংস্থা", "চাপ"]),
    (&["জাতিসংঘ"], &["সনদ", "নিরাপত্তা", "পরিষদ", "মহাসচিব"]),
    (&["বাংলাদেশ", "সরকার"], &["জানিয়েছে", "সিদ্ধান্ত", "বলেছে"]),
    (&["ভারত"], &["সরকার", "সম্পর্ক", "সীমান্ত", "পানি"]),
    (&["চীন"], &["সরকার", "বিনিয়োগ", "বাণিজ্য"]),
    (&["যুক্তরাষ্ট্র"], &["সরকার", "পররাষ্ট্র", "বিবৃতি"]),
    (&["পররাষ্ট্র"], &["মন্ত্রণালয়", "নীতি", "বিষয়ক"]),
    // ── Development / Infrastructure ─────────────────────────────────────
    (&["উন্নয়ন"], &["প্রকল্প", "কাজ", "লক্ষ্যমাত্রা", "পরিকল্পনা"]),
    (&["প্রকল্প"], &["বাস্তবায়ন", "কাজ", "ব্যয়", "অনুমোদন"]),
    (&["অবকাঠামো"], &["উন্নয়ন", "নির্মাণ", "ব্যয়"]),
    (&["যোগাযোগ"], &["ব্যবস্থা", "মন্ত্রণালয়", "নেটওয়ার্ক"]),
    // ── Formal reporting verbs / attribution ─────────────────────────────
    (&["তিনি"], &["বলেন", "জানান", "উল্লেখ", "যোগ করেন"]),
    (&["সূত্রে"], &["জানা", "বলা", "গেছে"]),
    (&["জানা"], &["গেছে", "গিয়েছে", "যায়"]),
    (&["বলেন", "তিনি"], &["আরও", "এই", "সেই"]),
    (&["জানান", "তিনি"], &["আরও", "এই", "সেই"]),
    (&["সংশ্লিষ্ট"], &["সূত্র", "কর্তৃপক্ষ", "বিভাগ"]),
    (&["নির্ভরযোগ্য"], &["সূত্র", "তথ্য"]),
    (&["বিশ্বস্ত"], &["সূত্র", "তথ্য"]),
    // ── Rights / Civil society ───────────────────────────────────────────
    (&["মানবাধিকার"], &["সংস্থা", "লঙ্ঘন", "কমিশন", "পরিস্থিতি"]),
    (&["নাগরিক"], &["সমাজ", "অধিকার", "সংগঠন"]),
    (&["আন্দোলন"], &["চলছে", "শুরু", "দাবি", "কর্মসূচি"]),
    (&["দাবি"], &["জানানো", "করা", "মানা", "পূরণ"]),
    // ── Technology / Digital ─────────────────────────────────────────────
    (&["ডিজিটাল"], &["বাংলাদেশ", "নিরাপত্তা", "অর্থনীতি", "রূপান্তর"]),
    (&["তথ্যপ্রযুক্তি"], &["খাত", "মন্ত্রণালয়", "উন্নয়ন"]),
    (&["সাইবার"], &["নিরাপত্তা", "হামলা", "অপরাধ"]),
    (&["কৃত্রিম", "বুদ্ধিমত্তা"], &["প্রযুক্তি", "ব্যবহার", "উন্নয়ন"]),
    // ── Miscellaneous formal triggers ────────────────────────────────────
    (&["ক্ষমতা"], &["গ্রহণ", "হস্তান্তর", "অপব্যবহার", "সংক্রান্ত"]),
    (&["দায়িত্ব"], &["গ্রহণ", "পালন", "অর্পণ"]),
    (&["অভিযোগ"], &["উঠেছে", "করা", "করেছেন", "অস্বীকার"]),
    (&["তদন্ত"], &["কমিটি", "রিপোর্ট", "শুরু", "চলছে"]),
    (&["ঘটনা"], &["ঘটেছে", "তদন্ত", "সম্পর্কে", "প্রসঙ্গে"]),
    (&["পরিস্থিতি"], &["নিয়ন্ত্রণ", "উন্নতি", "অবনতি", "পর্যবেক্ষণ"]),
    (&["চুক্তি"], &["স্বাক্ষর", "বাস্তবায়ন", "লঙ্ঘন", "সমঝোতা"]),
    (&["সমঝোতা"], &["স্মারক", "চুক্তি", "হয়েছে"]),
    (&["ঘোষণা"], &["করা", "দেওয়া", "পত্র", "অনুযায়ী"]),
    (&["নির্দেশ"], &["দেওয়া", "পাঠানো", "মেনে", "অনুযায়ী"]),
    (&["অনুমোদন"], &["দেওয়া", "পাওয়া", "মিলেছে"]),
    (&["বাস্তবায়ন"], &["হচ্ছে", "করা", "সম্ভব"]),
    (&["পর্যবেক্ষণ"], &["করা", "রাখা", "জানানো"]),
    (&["বিশেষজ্ঞ"], &["মতে", "বলেন", "মনে", "দল"]),
    (&["বিশ্লেষণ"], &["করা", "বলছে", "অনুযায়ী"]),
    (&["প্রতিবেদন"], &["অনুযায়ী", "জানিয়েছে", "বলছে"]),
    (&["সমীক্ষা"], &["অনুযায়ী", "বলছে", "রিপোর্ট"]),
    (&["জরিপ"], &["অনুযায়ী", "বলছে", "ফলাফল"]),
    (&["তথ্য"], &["অনুযায়ী", "মতে", "বলছে", "প্রকাশ"]),
    (&["তথ্য", "অনুযায়ী"], &["এই", "বলা", "দেখা"]),
    (&["এই", "প্রসঙ্গে"], &["বলা", "তিনি", "জানানো"]),
    (&["এই", "বিষয়ে"], &["বলা", "তিনি", "জানানো"]),
];

impl NextWordPredictor {
    pub fn new() -> Self {
        Self {
            lm: LanguageModel::new(),
        }
    }

    pub fn with_language_model(lm: LanguageModel) -> Self {
        Self { lm }
    }

    pub fn lm(&self) -> &LanguageModel {
        &self.lm
    }

    pub fn lm_mut(&mut self) -> &mut LanguageModel {
        &mut self.lm
    }

    /// Ingest raw text and train the underlying language model
    pub fn train_text(&mut self, text: &str) {
        self.lm.train_text(text);
    }

    /// Load pre-compiled dataset into the underlying language model
    pub fn load_trained_data(&mut self, data: &TrainedLanguageModelData) {
        self.lm.load_trained_data(data);
    }

    /// Predict the top-K probable next words given preceding sentence context
    pub fn predict_next(&self, context: &[&str], limit: usize) -> Vec<String> {
        self.predict_next_with_options(context, limit, true)
    }

    /// Predict the top-K probable next words with explicit scores given preceding sentence context
    pub fn predict_next_scored(&self, context: &[&str], limit: usize) -> Vec<(String, f32)> {
        self.predict_next_scored_with_options(context, limit, true)
    }

    /// Predict the top-K probable next words with scores given preceding sentence context with optional idiom phrases
    pub fn predict_next_scored_with_options(
        &self,
        context: &[&str],
        limit: usize,
        enable_idiom_phrases: bool,
    ) -> Vec<(String, f32)> {
        let context = crate::context::truncate_at_sentence_boundary(context);
        if context.is_empty() {
            let defaults = [
                ("আমি", -1.8),
                ("আপনি", -2.3),
                ("তুমি", -2.1),
                ("আমরা", -2.4),
                ("ধন্যবাদ", -2.5),
            ];
            return defaults
                .iter()
                .take(limit)
                .map(|(w, s)| (w.to_string(), *s))
                .collect();
        }

        // 0. Match high-confidence conversational idioms and phrases
        let mut idiom_matches: Vec<(String, f32)> = Vec::new();
        let mut seen: hashbrown::HashSet<String> = hashbrown::HashSet::new();

        if enable_idiom_phrases {
            for &(pattern, continuations) in BENGALI_IDIOM_PHRASES {
                if context.len() >= pattern.len() {
                    let tail = &context[context.len() - pattern.len()..];
                    if tail == pattern {
                        for (idx, &cont) in continuations.iter().enumerate() {
                            if seen.insert(cont.to_string()) {
                                idiom_matches.push((cont.to_string(), 0.5 - 0.1 * (idx as f32)));
                            }
                        }
                    }
                }
            }

            // 0b. Match formal / journalistic / academic phrase continuations.
            // Score: 0.45 — below conversational idioms (0.5) but well above any LM bigram.
            // These patterns activate formal writing register predictions.
            for &(pattern, continuations) in BENGALI_FORMAL_PHRASES {
                if context.len() >= pattern.len() {
                    let tail = &context[context.len() - pattern.len()..];
                    if tail == pattern {
                        for (idx, &cont) in continuations.iter().enumerate() {
                            if seen.insert(cont.to_string()) {
                                idiom_matches.push((cont.to_string(), 0.45 - 0.05 * (idx as f32)));
                            }
                        }
                    }
                }
            }
        }


        let pool_size = (limit * 3).max(12);
        let mut raw_candidates: Vec<String> = Vec::with_capacity(pool_size);

        let (prev2, prev1) = if context.len() >= 2 {
            (Some(context[context.len() - 2]), Some(context[context.len() - 1]))
        } else {
            (None, Some(context[context.len() - 1]))
        };

        if let (Some(p2), Some(p1)) = (prev2, prev1) {
            // 1. Trigram continuations
            for w in self.lm.get_next_words_trigram(p2, p1, pool_size) {
                if seen.insert(w.clone()) {
                    raw_candidates.push(w);
                }
            }
        }

        if let Some(p1) = prev1 {
            // 2. Bigram continuations
            for w in self.lm.get_next_words(p1, pool_size) {
                if seen.insert(w.clone()) {
                    raw_candidates.push(w);
                }
            }
        }

        // Score candidates
        let mut scored_candidates: Vec<(String, f32)> = raw_candidates
            .into_iter()
            .map(|cand| {
                let score = self.lm.score_candidate(prev2, prev1, &cand);
                (cand, score)
            })
            .collect();

        scored_candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Fill remaining slots with high-frequency contextual fallbacks
        if idiom_matches.len() + scored_candidates.len() < limit {
            let fallbacks = [
                "হবে", "আছে", "করব", "যাব", "ভালো",
                "চাই", "কথা", "দেখা", "ছিল", "করছি",
            ];
            for fb in fallbacks {
                if idiom_matches.len() + scored_candidates.len() >= limit {
                    break;
                }
                if seen.insert(fb.to_string()) && !context.contains(&fb) {
                    let score = self.lm.score_candidate(prev2, prev1, fb);
                    scored_candidates.push((fb.to_string(), score));
                }
            }
        }

        let mut final_result = idiom_matches;
        final_result.extend(scored_candidates);
        final_result.truncate(limit);
        final_result
    }

    /// Predict the top-K probable next words given preceding sentence context with optional idiom phrases
    pub fn predict_next_with_options(
        &self,
        context: &[&str],
        limit: usize,
        enable_idiom_phrases: bool,
    ) -> Vec<String> {
        self.predict_next_scored_with_options(context, limit, enable_idiom_phrases)
            .into_iter()
            .map(|(w, _)| w)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_continuations() {
        let predictor = NextWordPredictor::new();

        let preds_ami = predictor.predict_next(&["আমি"], 5);
        assert!(!preds_ami.is_empty());
        assert!(
            preds_ami.contains(&"মনে".to_string())
                || preds_ami.contains(&"জানি".to_string())
                || preds_ami.contains(&"আমার".to_string())
                || preds_ami.contains(&"ভালো".to_string())
                || preds_ami.contains(&"যাচ্ছি".to_string())
        );

        let preds_rice = predictor.predict_next(&["আমি", "ভাত"], 3);
        assert!(!preds_rice.is_empty());
        assert!(preds_rice.contains(&"খাচ্ছি".to_string()) || preds_rice.contains(&"খাব".to_string()));

        let preds_bd = predictor.predict_next(&["বাংলাদেশ", "একটি"], 3);
        assert!(!preds_bd.is_empty());
        assert!(preds_bd.contains(&"সুন্দর".to_string()) || preds_bd.contains(&"স্বাধীন".to_string()));
    }

    #[test]
    fn test_dynamic_corpus_training_prediction() {
        let mut predictor = NextWordPredictor::new();
        predictor.train_text("বাংলা আমার অহংকার। বাংলা আমার মাতৃভাষা।");

        let preds = predictor.predict_next(&["বাংলা", "আমার"], 3);
        assert!(!preds.is_empty());
        assert!(preds.contains(&"অহংকার".to_string()) || preds.contains(&"মাতৃভাষা".to_string()));
    }
}
