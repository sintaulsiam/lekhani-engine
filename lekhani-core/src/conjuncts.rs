//! Comprehensive Bengali Conjunct ("যুক্তবর্ণ") Catalog & Search Engine
//! Provides breakdown (e.g. ক + ্ + ষ), phonetic key sequences (e.g. kkh / kSh),
//! and sample vocabulary for all major Bengali conjuncts.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConjunctInfo {
    pub conjunct: String,
    pub breakdown: String,
    pub phonetic: String,
    pub examples: String,
}

impl ConjunctInfo {
    pub fn new(conjunct: &str, breakdown: &str, phonetic: &str, examples: &str) -> Self {
        Self {
            conjunct: conjunct.to_string(),
            breakdown: breakdown.to_string(),
            phonetic: phonetic.to_string(),
            examples: examples.to_string(),
        }
    }
}

pub struct ConjunctCatalog;

impl ConjunctCatalog {
    pub fn all() -> Vec<ConjunctInfo> {
        vec![
            // K-series
            ConjunctInfo::new("ক্ষ", "ক + ্ + ষ", "kkh / kSh", "শিক্ষা, ক্ষমা, পরীক্ষা, ক্ষতি"),
            ConjunctInfo::new("ক্ক", "ক + ্ + ক", "kk", "মক্কা, চক্কর, এক্কা"),
            ConjunctInfo::new("ক্ট", "ক + ্ + ট", "kT", "অক্টোবর, ডক্টর"),
            ConjunctInfo::new("ক্ত", "ক + ্ + ত", "kt", "রক্ত, ভক্ত, শক্ত, মুক্তি"),
            ConjunctInfo::new("ক্ল", "ক + ্ + ল", "kl", "ক্লান্ত, ক্লেশ, ক্লাস"),
            ConjunctInfo::new("ক্ব", "ক + ্ + ব", "kw", "পক্ব, ক্বচিৎ"),
            ConjunctInfo::new("ক্র", "ক + ্ + র", "kr", "চক্র, বক্র, বিক্রয়"),
            ConjunctInfo::new("ক্স", "ক + ্ + স", "ks", "বাক্স, রিকশা, ট্যাক্স"),
            // G/Gh-series
            ConjunctInfo::new("গ্ধ", "গ + ্ + ধ", "gdh", "মুগ্ধ, দগ্ধ, স্নিগ্ধ"),
            ConjunctInfo::new("গ্ন", "গ + ্ + ন", "gn", "ভগ্ন, অগ্নিকাণ্ড, মগ্ন"),
            ConjunctInfo::new("গ্র", "গ + ্ + র", "gr", "গ্রাম, গ্রহ, অগ্র"),
            ConjunctInfo::new("গ্ল", "গ + ্ + ল", "gl", "গ্লানি, গ্লুকোজ"),
            // Ng-series
            ConjunctInfo::new("ঙ্ক", "ঙ + ্ + ক", "ngk", "অঙ্ক, আতঙ্ক, শঙ্কা, কলঙ্ক"),
            ConjunctInfo::new("ঙ্ক্ষ", "ঙ + ্ + ক + ্ + ষ", "ngkkh", "আকাঙ্ক্ষা"),
            ConjunctInfo::new("ঙ্গ", "ঙ + ্ + গ", "ngg", "সঙ্গ, অঙ্গ, বঙ্গ, গঙ্গা"),
            ConjunctInfo::new("ঙ্ঘ", "ঙ + ্ + ঘ", "nggh", "সঙ্ঘ, জঙ্ঘা, সঙ্ঘর্ষ"),
            ConjunctInfo::new("ঙ্ম", "ঙ + ্ + ম", "ngm", "বাঙ্ময়"),
            // C/Ch-series
            ConjunctInfo::new("চ্চ", "চ + ্ + চ", "cc", "উচ্চ, বাচ্চা, সাচ্চা"),
            ConjunctInfo::new("চ্ছ", "চ + ্ + ছ", "cch", "ইচ্ছা, স্বচ্ছ, গুচ্ছ, তুচ্ছ"),
            ConjunctInfo::new("চ্ছ্ব", "চ + ্ + ছ + ্ + ব", "cchw", "উচ্ছ্বাস, উচ্ছ্বল"),
            ConjunctInfo::new("চ্ঞ", "চ + ্ + ঞ", "cn", "যাচঞা"),
            // J-series
            ConjunctInfo::new("জ্ঞ", "জ + ্ + ঞ", "ggy / jn", "জ্ঞান, বিজ্ঞান, অজ্ঞ, প্রজ্ঞা"),
            ConjunctInfo::new("জ্জ", "জ + ্ + জ", "jj", "লজ্জা, সজ্জা, মজ্জা"),
            ConjunctInfo::new("জ্ঝ", "জ + ্ + ঝ", "jjh", "ঝঞ্ঝা, উজ্ঝট"),
            ConjunctInfo::new("জ্ব", "জ + ্ + ব", "jw", "জ্বর, জ্বালা, উজ্জ্বল"),
            ConjunctInfo::new("জর", "জ + ্ + র", "jr", "বজ্র, অগ্রজ"),
            // Ny-series
            ConjunctInfo::new("ঞ্চ", "ঞ + ্ + চ", "nc", "অঞ্চল, পঞ্চম, কাঞ্চন, মঞ্চ"),
            ConjunctInfo::new("ঞ্ছ", "ঞ + ্ + ছ", "nch", "বাঞ্ছা, বাঞ্ছনীয়"),
            ConjunctInfo::new("ঞ্জ", "ঞ + ্ + জ", "nj", "কুঞ্জ, গুঞ্জন, ব্যঞ্জন, রঞ্জন"),
            ConjunctInfo::new("ঞ্ঝ", "ঞ + ্ + ঝ", "njh", "ঝঞ্ঝা"),
            // T/Th (Retroflex)-series
            ConjunctInfo::new("ট্ট", "ট + ্ + ট", "TT", "চট্টগ্রাম, ভুট্টা, পট্টি"),
            ConjunctInfo::new("ট্ব", "ট + ্ + ব", "Tw", "টুইটার"),
            ConjunctInfo::new("ট্র", "ট + ্ + র", "Tr", "ট্রেন, ট্রাক, পেট্রোল"),
            ConjunctInfo::new("ড্ড", "ড + ্ + ড", "DD", "উড্ডয়ন, হাড্ডি, আড্ডা"),
            ConjunctInfo::new("ড্র", "ড + ্ + র", "Dr", "ড্রাইভার, ড্রাম"),
            // N (Murdhanya)-series
            ConjunctInfo::new("ণ্ট", "ণ + ্ + ট", "NT", "ঘণ্টা, বণ্টন, লণ্ঠন"),
            ConjunctInfo::new("ণ্ঠ", "ণ + ্ + ঠ", "NTh", "কণ্ঠ, উৎকণ্ঠা, কুণ্ঠা"),
            ConjunctInfo::new("ণ্ড", "ণ + ্ + ড", "ND", "পণ্ড, দণ্ড, কাণ্ড, খণ্ড"),
            ConjunctInfo::new("ণ্ঢ", "ণ + ্ + ঢ", "NDh", "ঢুণ্ঢন"),
            ConjunctInfo::new("ণ্ণ", "ণ + ্ + ণ", "NN", "বিষণ্ণ, লাবণ্য"),
            // T/Th (Dental)-series
            ConjunctInfo::new("ত্ত", "ত + ্ + ত", "tt", "উত্তর, বৃত্ত, পিত্ত, সত্য"),
            ConjunctInfo::new("ত্ত্ব", "ত + ্ + ত + ্ + ব", "ttw", "তত্ত্ব, গুরুত্ব, মহত্ত্ব"),
            ConjunctInfo::new("ত্থ", "ত + ্ + থ", "tth", "উত্থান, পথ্য"),
            ConjunctInfo::new("ত্ন", "ত + ্ + ন", "tn", "রত্ন, যত্ন, পত্নী"),
            ConjunctInfo::new("ত্ম", "ত + ্ + ম", "tm", "আত্মা, মহাত্মা, আত্মীয়"),
            ConjunctInfo::new("ত্র", "ত + ্ + র", "tr", "ছাত্র, পত্র, রাত্রি, চিত্র"),
            ConjunctInfo::new("ত্ব", "ত + ্ + ব", "tw", "ত্বক, দূরত্ব, ব্যক্তিত্ব"),
            // D/Dh-series
            ConjunctInfo::new("দ্দ", "দ + ্ + দ", "dd", "উদ্দেশ্য, রদ্দি, বদ্দর"),
            ConjunctInfo::new("দ্ধ", "দ + ্ + ধ", "ddh", "যুদ্ধ, বুদ্ধি, শুদ্ধ, সমৃদ্ধ"),
            ConjunctInfo::new("দ্ব", "দ + ্ + ব", "dw", "দ্বীপ, বিদ্বান, দ্বিতীয়, দ্বন্দ্ব"),
            ConjunctInfo::new("দ্ভ", "দ + ্ + ভ", "dbh", "উদ্ভব, অদ্ভুত, সদ্ভাব"),
            ConjunctInfo::new("দ্ম", "দ + ্ + ম", "dm", "পদ্ম, ছদ্মবেশ"),
            ConjunctInfo::new("দ্র", "দ + ্ + র", "dr", "ভদ্র, নিদ্রা, রুদ্র"),
            // N (Dantya)-series
            ConjunctInfo::new("ন্ত", "ন + ্ + ত", "nt", "শান্ত, অনন্ত, দিগন্ত, অন্ত"),
            ConjunctInfo::new("ন্ত্র", "ন + ্ + ত + ্ + র", "ntr", "যন্ত্র, মন্ত্র, তন্ত্র, নিয়ন্ত্রণ"),
            ConjunctInfo::new("ন্ত্ব", "ন + ্ + ত + ্ + ব", "ntw", "সান্ত্বনা"),
            ConjunctInfo::new("ন্থ", "ন + ্ + থ", "nth", "গ্রন্থ, পন্থা, পান্থ"),
            ConjunctInfo::new("ন্দ", "ন + ্ + দ", "nd", "সুন্দর, আনন্দ, ছন্দ, বন্দনা"),
            ConjunctInfo::new("ন্দ্র", "ন + ্ + দ + ্ + র", "ndr", "চন্দ্র, ইন্দ্র, কেন্দ্র"),
            ConjunctInfo::new("ন্দ্ব", "ন + ্ + দ + ্ + ব", "ndw", "দ্বন্দ্ব"),
            ConjunctInfo::new("ন্ধ", "ন + ্ + ধ", "ndh", "অন্ধ, বন্ধ, গন্ধ, রন্ধন"),
            ConjunctInfo::new("ন্ন", "ন + ্ + ন", "nn", "অন্ন, ভিন্ন, পান্না, কান্না"),
            ConjunctInfo::new("ন্ম", "ন + ্ + ম", "nm", "জন্ম, উন্মাদ, তন্ময়"),
            // P/Ph-series
            ConjunctInfo::new("প্ত", "প + ্ + ত", "pt", "সুপ্ত, প্রাপ্ত, সমাপ্ত"),
            ConjunctInfo::new("প্ন", "প + ্ + ন", "pn", "স্বপ্ন"),
            ConjunctInfo::new("প্প", "প + ্ + প", "pp", "বাপ্পা, থাপ্পড়"),
            ConjunctInfo::new("প্র", "প + ্ + র", "pr", "প্রেম, প্রাণ, প্রথম, প্রশ্ন"),
            ConjunctInfo::new("প্ল", "প + ্ + ল", "pl", "প্লাবন, বিপ্লব, প্লাস্টিক"),
            ConjunctInfo::new("প্স", "প + ্ + স", "ps", "লিপ্সা"),
            // B/Bh-series
            ConjunctInfo::new("ব্জ", "ব + ্ + জ", "bj", "অব্জ, কুব্জ"),
            ConjunctInfo::new("ব্দ", "ব + ্ + দ", "bd", "শব্দ, শতাব্দী, জব্দ"),
            ConjunctInfo::new("ব্ধ", "ব + ্ + ধ", "bdh", "লব্ধ, স্তব্ধ, প্রলুব্ধ"),
            ConjunctInfo::new("ব্ব", "ব + ্ + ব", "bb", "আব্বা, ডিব্বা"),
            ConjunctInfo::new("ব্র", "ব + ্ + র", "br", "ব্রাহ্মণ, তীব্র, ব্রত"),
            ConjunctInfo::new("ব্ল", "ব + ্ + ল", "bl", "ব্লগ, ব্লক, টেবিল"),
            // M-series
            ConjunctInfo::new("ম্প", "ম + ্ + প", "mp", "সম্পদ, চম্পা, কম্পন"),
            ConjunctInfo::new("ম্ফ", "ম + ্ + ফ", "mf / mph", "লম্ফ, গুল্ফ"),
            ConjunctInfo::new("ম্ব", "ম + ্ + ব", "mb", "কম্বল, সম্বল, অম্বর"),
            ConjunctInfo::new("ম্ভ", "ম + ্ + ভ", "mbh", "সম্ভাবনা, আরম্ভ, গম্ভীর"),
            ConjunctInfo::new("ম্ম", "ম + ্ + ম", "mm", "সম্মান, সম্মতি, সম্মেলন"),
            ConjunctInfo::new("ম্র", "ম + ্ + র", "mr", "নম্র, আম্র"),
            ConjunctInfo::new("ম্ল", "ম + ্ + ল", "ml", "ম্লান, অম্ল"),
            // L-series
            ConjunctInfo::new("ল্ক", "ল + ্ + ক", "lk", "বল্কল"),
            ConjunctInfo::new("ল্গ", "ল + ্ + গ", "lg", "বল্গা"),
            ConjunctInfo::new("ল্ট", "ল + ্ + ট", "lT", "উল্টো, বল্টু"),
            ConjunctInfo::new("ল্ড", "ল + ্ + ড", "lD", "হোল্ডিং, ফিল্ড"),
            ConjunctInfo::new("ল্প", "ল + ্ + প", "lp", "গল্প, অল্প, কল্পনা, শিল্প"),
            ConjunctInfo::new("ল্ব", "ল + ্ + ব", "lw", "বিল্ব"),
            ConjunctInfo::new("ল্ল", "ল + ্ + ল", "ll", "পল্লী, উল্লাস, কেল্লা"),
            // Sh (Talobya)-series
            ConjunctInfo::new("শ্চ", "শ + ্ + চ", "shc", "নিশ্চয়, আশ্চর্য, পশ্চাৎ"),
            ConjunctInfo::new("শ্ছ", "শ + ্ + ছ", "shch", "শিরশ্ছেদ"),
            ConjunctInfo::new("শ্ন", "শ + ্ + ন", "shn", "প্রশ্ন"),
            ConjunctInfo::new("শ্ব", "শ + ্ + ব", "shw", "অশ্ব, বিশ্বাস, ঈশ্বর, শ্বেত"),
            ConjunctInfo::new("শ্ম", "শ + ্ + ম", "shm", "শ্মশান, কাশ্মীর, রশ্নি"),
            ConjunctInfo::new("শ্র", "শ + ্ + র", "shr", "পরিশ্রম, বিশ্রাম, শ্রদ্ধা, আশ্রয়"),
            ConjunctInfo::new("শ্ল", "শ + ্ + ল", "shl", "শ্লোক, অশ্লীল"),
            // Sh (Murdhanya)-series
            ConjunctInfo::new("ষ্ট", "ষ + ্ + ট", "ShT", "কষ্ট, নষ্ট, সৃষ্টি, বৃষ্টি, স্পষ্ট"),
            ConjunctInfo::new("ষ্ঠ", "ষ + ্ + ঠ", "ShTh", "শ্রেষ্ঠ, জ্যেষ্ঠ, অনুষ্ঠান, ষষ্ঠ"),
            ConjunctInfo::new("ষ্ণ", "ষ + ্ + ণ", "ShN", "কৃষ্ণ, উষ্ণ, তৃষ্ণা, বৈষ্ণব"),
            ConjunctInfo::new("ষ্প", "ষ + ্ + প", "Shp", "পুষ্প, নিষ্পাপ, বাষ্প"),
            ConjunctInfo::new("ষ্ফ", "ষ + ্ + ফ", "Shf", "নিষ্ফল"),
            ConjunctInfo::new("ষ্ম", "ষ + ্ + ম", "Shm", "গ্রীষ্ম, উষ্ম"),
            // S (Dantya)-series
            ConjunctInfo::new("স্ক", "স + ্ + ক", "sk", "স্কুল, পুরস্কার, ভাস্কর"),
            ConjunctInfo::new("স্খ", "স + ্ + খ", "skh", "স্খলন"),
            ConjunctInfo::new("স্ত", "স + ্ + ত", "st", "রাস্তা, ব্যস্ত, সমস্ত, সস্তা"),
            ConjunctInfo::new("স্ত্র", "স + ্ + ত + ্ + র", "str", "স্ত্রী, অস্ত্র, শাস্ত্র"),
            ConjunctInfo::new("স্থ", "স + ্ + থ", "sth", "স্থান, স্বাস্থ্য, প্রস্থান, অবস্থা"),
            ConjunctInfo::new("স্ন", "স + ্ + ন", "sn", "স্নান, স্নেহ, স্নানাগার"),
            ConjunctInfo::new("স্প", "স + ্ + প", "sp", "স্পর্শ, স্পষ্ট, স্পন্দন"),
            ConjunctInfo::new("স্ফ", "স + ্ + ফ", "sf / sph", "স্ফীতি, স্ফুলিঙ্গ, আস্ফালন"),
            ConjunctInfo::new("স্ব", "স + ্ + ব", "sw", "স্বাধীনতা, স্বপ্ন, স্বাদ, স্বদেশ"),
            ConjunctInfo::new("স্ম", "স + ্ + ম", "sm", "স্মৃতি, স্মরণ, বিস্ময়, ভস্ম"),
            ConjunctInfo::new("স্র", "স + ্ + র", "sr", "স্রোত, সহস্র, অজস্র"),
            ConjunctInfo::new("স্ল", "স + ্ + ল", "sl", "স্লোগান, স্লিপ"),
            // H-series
            ConjunctInfo::new("হ্ন", "হ + ্ + ন", "hn", "বহ্নি, মধ্যাহ্ন"),
            ConjunctInfo::new("হ্ণ", "হ + ্ + ণ", "hN", "অপরাহ্ণ, সায়াহ্ন"),
            ConjunctInfo::new("হ্ব", "হ + ্ + ব", "hw", "আহ্বান, জিহ্বা, বিহ্বল"),
            ConjunctInfo::new("হ্ম", "হ + ্ + ম", "hm", "ব্রাহ্মণ, ব্রহ্মপুত্র, ব্রহ্ম"),
            ConjunctInfo::new("হ্য", "হ + ্ + য", "hy", "সহ্য, ধার্য, বাহ্য"),
            ConjunctInfo::new("হ্র", "হ + ্ + র", "hr", "হ্রদ, হ্রাস"),
            ConjunctInfo::new("হ্ল", "হ + ্ + ল", "hl", "আহ্লাদ, প্রহ্লাদ"),
            ConjunctInfo::new("হৃ", "হ + ৃ", "hri", "হৃদয়, হৃৎপিণ্ড"),
        ]
    }

    /// Live case-insensitive search filtering across conjunct glyph, breakdown, phonetic keys, and examples
    pub fn search(query: &str) -> Vec<ConjunctInfo> {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return Self::all();
        }

        Self::all()
            .into_iter()
            .filter(|info| {
                info.conjunct.contains(&q)
                    || info.breakdown.to_lowercase().contains(&q)
                    || info.phonetic.to_lowercase().contains(&q)
                    || info.examples.to_lowercase().contains(&q)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conjunct_catalog_completeness_and_search() {
        let all = ConjunctCatalog::all();
        assert!(all.len() >= 80);

        // Search by glyph
        let res_kkh = ConjunctCatalog::search("ক্ষ");
        assert!(!res_kkh.is_empty());
        assert_eq!(res_kkh[0].conjunct, "ক্ষ");

        // Search by phonetic key
        let res_phonetic = ConjunctCatalog::search("kkh");
        assert!(!res_phonetic.is_empty());

        // Search by breakdown
        let res_breakdown = ConjunctCatalog::search("হ + ্ + ম");
        assert!(!res_breakdown.is_empty());
        assert_eq!(res_breakdown[0].conjunct, "হ্ম");
    }
}
