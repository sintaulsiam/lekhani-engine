use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let json_candidates = [
        Path::new("/mnt/data/lekhani-engine/data/dictionaries/phonetic_overrides.json"),
        Path::new("data/dictionaries/phonetic_overrides.json"),
        Path::new("/mnt/data/lekhani-android/data/dictionaries/phonetic_overrides.json"),
    ];
    let json_path = json_candidates
        .iter()
        .find(|p| p.exists())
        .copied()
        .ok_or("Cannot find phonetic_overrides.json")?;
    let file = File::open(json_path)?;
    let map: HashMap<String, Vec<(String, f32)>> = serde_json::from_reader(BufReader::new(file))?;
    println!("Loaded {} overrides from {:?}", map.len(), json_path);

    let bin_bytes = lekhani_core::phonetic::overrides::ZeroCopyOverrides::compile(&map);
    println!(
        "Compiled to {} bytes ({:.1} KB)",
        bin_bytes.len(),
        bin_bytes.len() as f32 / 1024.0
    );

    let targets = [
        "/mnt/data/lekhani-android/data/dictionaries/phonetic_overrides.bin",
        "/mnt/data/lekhani-android/android/app/src/main/assets/dictionaries/phonetic_overrides.bin",
        "/mnt/data/lekhani/data/dictionaries/phonetic_overrides.bin",
        "/mnt/data/lekhani-engine/data/dictionaries/phonetic_overrides.bin",
    ];

    for target in targets {
        let p = Path::new(target);
        if let Some(parent) = p.parent() {
            if parent.exists() {
                let mut f = File::create(p)?;
                f.write_all(&bin_bytes)?;
                println!("Saved to {:?}", p);
            }
        }
    }

    Ok(())
}
