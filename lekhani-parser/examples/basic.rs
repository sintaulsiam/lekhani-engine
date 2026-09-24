use lekhani_parser::default_avro_parser;

fn main() {
    let parser = default_avro_parser();

    let tests = [
        "ami banglay gan gai",
        "amra tomra bangladesh",
        "shikkhok porikkha biggan",
        "brriShTi sUrrzo",
    ];

    println!("=== Lekhani Bengali Phonetic Parser ===");
    for input in tests {
        let output = parser.convert(input);
        println!("{:25} -> {}", input, output);
    }
}
