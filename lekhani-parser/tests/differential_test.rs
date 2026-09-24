use lekhani_parser::default_avro_parser;

#[test]
fn test_golden_avro_raw_equivalence() {
    let lekhani = default_avro_parser();

    let golden_cases = [
        ("ami", "আমি"),
        ("tumi", "তুমি"),
        ("se", "সে"),
        ("amra", "আম্রা"),
        ("tomra", "তম্রা"),
        ("tOmra", "তোম্রা"),
        ("tara", "তারা"),
        ("bangla", "বাংলা"),
        ("bangladesh", "বাংলাদেশ"),
        ("dhaka", "ধাকা"),
        ("Dhaka", "ঢাকা"),
        ("buddhu", "বুদ্ধু"),
        ("biggan", "বিজ্ঞান"),
        ("biggani", "বিজ্ঞানি"),
        ("bigganI", "বিজ্ঞানী"),
        ("shikha", "শিখা"),
        ("shikkhok", "শিক্ষক"),
        ("porikkha", "পরিক্ষা"),
        ("porIkkha", "পরীক্ষা"),
        ("kkh", "ক্ষ"),
        ("shch", "শ্ছ"),
        ("Ngk", "ঙ্ক"),
        ("ngk", "ংক"),
        ("NgkSh", "ঙ্ক্ষ"),
        ("ShTh", "ষ্ঠ"),
        ("cch", "চ্ছ"),
        ("prothom", "প্রথম"),
        ("sundor", "সুন্দর"),
        ("bhalo", "ভাল"),
        ("bhalO", "ভালো"),
        ("kharap", "খারাপ"),
        ("notun", "নতুন"),
        ("puraton", "পুরাতন"),
        ("shanti", "শান্তি"),
        ("santi", "সান্তি"),
        ("rasta", "রাস্তা"),
        ("ghor", "ঘর"),
        ("bari", "বারি"),
        ("gari", "গারি"),
        ("nodI", "নদী"),
        ("pahaR", "পাহাড়"),
        ("akash", "আকাশ"),
        ("batas", "বাতাস"),
        ("brriShTi", "বৃষ্টি"),
        ("bidyut```", "বিদ্যুৎ"),

        ("megh", "মেঘ"),
        ("sUrrzo", "সূর্য"),
        ("ca^d", "চাঁদ"),
        ("ha^s", "হাঁস"),
        ("pa^c", "পাঁচ"),
        ("t``", "ৎ"),
        ("hoThat``", "হঠাৎ"),
        ("ut``sob", "উৎসব"),
        ("du:kho", "দুঃখ"),
        ("du:somoy", "দুঃসময়"),
        ("ongsho", "অংশ"),
        ("bongsho", "বংশ"),
        ("songsod", "সংসদ"),
        ("rong", "রং"),
        ("bostu", "বস্তু"),
        ("sroShTa", "স্রষ্টা"),
        ("kanna", "কান্না"),
        ("hasonahena", "হাসনাহেনা"),
        ("porashUna", "পরাশূনা"),
        ("lekha", "লেখা"),
        ("pora", "পরা"),
        ("khaoya", "খাওয়া"),
        ("daoya", "দাওয়া"),
        ("ghuma", "ঘুমা"),
        ("jagorito", "জাগরিত"),
        ("obostha", "অবস্থা"),
        ("byabostha", "ব্যাবস্থা"),
        ("byakti", "ব্যাক্তি"),
        ("bidyan", "বিদ্যান"),
        ("bidyaloy", "বিদ্যালয়"),
        ("shikkhaloy", "শিক্ষালয়"),
        ("bishwobidyaloy", "বিশ্ববিদ্যালয়"),
    ];

    let mut failures = Vec::new();
    for (input, expected) in golden_cases {
        let output = lekhani.convert(input);
        if output != expected {
            failures.push(format!("'{}': got '{}', expected '{}'", input, output, expected));
        }
    }
    assert!(failures.is_empty(), "Mismatches:\n{}", failures.join("\n"));
}
