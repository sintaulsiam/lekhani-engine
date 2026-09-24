use lekhani_parser::default_avro_parser;

#[test]
fn test_parser_basic_conversions() {
    let parser = default_avro_parser();

    assert_eq!(parser.convert("ami"), "আমি");
    assert_eq!(parser.convert("tumi"), "তুমি");
    assert_eq!(parser.convert("se"), "সে");
    assert_eq!(parser.convert("bangla"), "বাংলা");
    assert_eq!(parser.convert("buddhu"), "বুদ্ধু");
    assert_eq!(parser.convert("biggani"), "বিজ্ঞানি");
    assert_eq!(parser.convert("bigganI"), "বিজ্ঞানী");
}

#[test]
fn test_case_sensitivity_rules() {
    let parser = default_avro_parser();

    // Uppercase non-case-sensitive characters normalize to lowercase
    assert_eq!(parser.convert("B"), "ব");
    assert_eq!(parser.convert("K"), "ক");
    assert_eq!(parser.convert("M"), "ম");
    assert_eq!(parser.convert("P"), "প");

    // Case-sensitive characters distinguish retroflex / dental / sibilants
    assert_eq!(parser.convert("t"), "ত");
    assert_eq!(parser.convert("T"), "ট");
    assert_eq!(parser.convert("d"), "দ");
    assert_eq!(parser.convert("D"), "ড");
    assert_eq!(parser.convert("s"), "স");
    assert_eq!(parser.convert("S"), "শ");
    assert_eq!(parser.convert("sh"), "শ");
    assert_eq!(parser.convert("Sh"), "ষ");
    assert_eq!(parser.convert("r"), "র");
    assert_eq!(parser.convert("R"), "ড়");
    assert_eq!(parser.convert("Rh"), "ঢ়");
    assert_eq!(parser.convert("n"), "ন");
    assert_eq!(parser.convert("N"), "ণ");
}

#[test]
fn test_complex_conjuncts() {
    let parser = default_avro_parser();

    assert_eq!(parser.convert("kkh"), "ক্ষ");
    assert_eq!(parser.convert("shch"), "শ্ছ");
    assert_eq!(parser.convert("Ngk"), "ঙ্ক");
    assert_eq!(parser.convert("ngk"), "ংক");
    assert_eq!(parser.convert("NgkSh"), "ঙ্ক্ষ");
    assert_eq!(parser.convert("ShTh"), "ষ্ঠ");
    assert_eq!(parser.convert("cch"), "চ্ছ");

}

#[test]
fn test_chandra_bindu_and_modifiers() {
    let parser = default_avro_parser();

    assert_eq!(parser.convert("ca^d"), "চাঁদ");
    assert_eq!(parser.convert("ha^s"), "হাঁস");
    assert_eq!(parser.convert("ba^dh"), "বাঁধ");
    assert_eq!(parser.convert("pa^c"), "পাঁচ");
    assert_eq!(parser.convert("t``"), "ৎ");
    assert_eq!(parser.convert("hoThat``"), "হঠাৎ");
    assert_eq!(parser.convert("du:kho"), "দুঃখ");
}

#[test]
fn test_zero_alloc_convert_into() {
    let parser = default_avro_parser();
    let mut buf = String::with_capacity(64);

    parser.convert_into("buddhu", &mut buf);
    assert_eq!(buf, "বুদ্ধু");

    buf.clear();
    parser.convert_into("bangladesh", &mut buf);
    assert_eq!(buf, "বাংলাদেশ");
}
