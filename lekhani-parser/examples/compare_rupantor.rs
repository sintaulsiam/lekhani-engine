use std::time::Instant;
use lekhani_parser::default_avro_parser;
use rupantor::parser::PhoneticParser;
use serde_json::Value;

fn main() {
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("   🔬 Benchmark: LekhaniParser vs Rupantor (100,000 Iterations / Test)");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    let lekhani = default_avro_parser();

    let json_bytes = include_str!("../data/avrophonetic.json");
    let val: Value = serde_json::from_str(json_bytes).unwrap();
    let layout_obj = if let Some(l) = val.get("layout") { l } else { &val };
    let rupantor = PhoneticParser::new(layout_obj);

    let test_suites = [
        ("Short word (\"ami\")", "ami"),
        ("Medium word (\"bangla\")", "bangla"),
        ("Complex conjunct (\"shikkhok\")", "shikkhok"),
        ("Heavy conjuncts (\"brriShTi\")", "brriShTi"),
        ("Standard sentence (\"amader bangladesh\")", "amader bangladesh"),
        ("Long sentence (\"ami banglay gan gai ami banglar gan gai\")", "ami banglay gan gai ami banglar gan gai"),
    ];

    const ITERATIONS: u32 = 100_000;

    println!("{:<32} | {:<16} | {:<16} | {:<10}", "Workload", "Rupantor", "Lekhani (alloc)", "Speedup");
    println!("{:-<32}-+-{:-<16}-+-{:-<16}-+-{:-<10}", "", "", "", "");

    for (desc, input) in &test_suites {
        // Warm up
        for _ in 0..10_000 {
            let _ = rupantor.convert(input);
            let _ = lekhani.convert(input);
        }

        // Benchmark Rupantor
        let start_rup = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = rupantor.convert(input);
        }
        let dur_rup = start_rup.elapsed();
        let ns_rup = dur_rup.as_nanos() as f64 / ITERATIONS as f64;

        // Benchmark Lekhani (with String allocation)
        let start_lek = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = lekhani.convert(input);
        }
        let dur_lek = start_lek.elapsed();
        let ns_lek = dur_lek.as_nanos() as f64 / ITERATIONS as f64;

        let speedup = ns_rup / ns_lek;

        println!(
            "{:<32} | {:>10.2} ns/op | {:>10.2} ns/op | {:>7.2}x",
            desc, ns_rup, ns_lek, speedup
        );
    }

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("   🚀 Zero-Allocation Hot-Path (convert_into reusing caller scratch buffer)");
    println!("══════════════════════════════════════════════════════════════════════════════\n");

    println!("{:<32} | {:<16} | {:<16} | {:<10}", "Workload", "Rupantor (alloc)", "Lekhani (0-alloc)", "Speedup");
    println!("{:-<32}-+-{:-<16}-+-{:-<16}-+-{:-<10}", "", "", "", "");

    let mut buf = String::with_capacity(128);

    for (desc, input) in &test_suites {
        // Benchmark Rupantor
        let start_rup = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = rupantor.convert(input);
        }
        let dur_rup = start_rup.elapsed();
        let ns_rup = dur_rup.as_nanos() as f64 / ITERATIONS as f64;

        // Benchmark Lekhani Zero-Alloc
        let start_lek = Instant::now();
        for _ in 0..ITERATIONS {
            buf.clear();
            lekhani.convert_into(input, &mut buf);
        }
        let dur_lek = start_lek.elapsed();
        let ns_lek = dur_lek.as_nanos() as f64 / ITERATIONS as f64;

        let speedup = ns_rup / ns_lek;

        println!(
            "{:<32} | {:>10.2} ns/op | {:>10.2} ns/op | {:>7.2}x",
            desc, ns_rup, ns_lek, speedup
        );
    }

    println!("\n══════════════════════════════════════════════════════════════════════════════\n");
}
