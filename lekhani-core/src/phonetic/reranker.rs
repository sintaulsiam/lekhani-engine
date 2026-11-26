//! Intelligent Candidate Reranker
//!
//! Fixes direct Avro transliteration bias and unigram frequency anomalies:
//! 1. Initial `a-` vowel bias (আ- vs অ-)
//! 2. Inherent vowel `a` vs explicit `া` suppression (অক্ষারকে -> অক্ষরকে)
//! 3. Clitic particle promotion (-o -> -ও, -i -> -ই)
//! 4. Tatsama retroflex nasal conjuncts (nd -> ণ্ড)
//! 5. English loanword digraphs (auxford -> oxford -> অক্সফোর্ড)
//! 6. De-gemination of double consonant typing noise (oppohoron -> অপহরণ)
//! 7. Terminal inflectional suffixes (-ey -> -ে, independent এর -> ের, -er -> -ের)
//! 8. Sanskrit / Indic 'v' / 'w' -> 'ব' (avatar -> abatar -> অবতার, vidya -> বিদ্যা)
//! 9. Participial adjective suffixes (-ito/-ita -> -িত, e.g. atyacharita -> অত্যাচারিত)
//! 10. Modern Standard Orthography (হ্রস্ব ু vs দীর্ঘ ূ normalization)
//! 11. Dictionary word preference over non-dictionary candidates (agradut -> অগ্রদূত)

use super::suggestion::PhoneticSuggestion;

pub struct CandidateReranker;

impl CandidateReranker {
    /// Applies intelligent contextual reranking to the candidate list.
    pub fn rerank(input: &str, candidates: &mut Vec<String>, suggestion: &mut PhoneticSuggestion) {
        if candidates.is_empty() {
            return;
        }

        let lower_owned;
        let lower: &str = if input.bytes().any(|b| b.is_ascii_uppercase()) {
            lower_owned = input.to_ascii_lowercase();
            &lower_owned
        } else {
            input
        };

        // Supervised phonetic override has absolute authority; never demote or disrupt it
        if suggestion.database.lookup_override(lower).is_some()
            || suggestion.database.lookup_override(input).is_some()
            || lekhani_parser::has_explicit_avro_intent(input)
        {
            return;
        }

        // 1. Initial Vowel Reranker (আ- vs অ-)
        Self::rerank_initial_vowel(lower, candidates, suggestion);

        // 2. Medial Inherent Vowel Reranker (অক্ষারকে -> অক্ষরকে)
        Self::rerank_medial_inherent_vowel(lower, candidates, suggestion);

        // 3. Clitic Particle Reranker (অঞ্চল + ও -> অঞ্চলও)
        Self::rerank_clitic_particles(lower, candidates, suggestion);

        // 4. Tatsama Retroflex Nasal Conjuncts (ন্দ -> ণ্ড)
        Self::rerank_retroflex_nasal(lower, candidates, suggestion);

        // 5. English Loanword Initial 'au' -> 'অ' (auxford -> অক্সফোর্ড)
        Self::rerank_loanword_au(lower, candidates, suggestion);

        // 6. De-gemination of Double Consonant Typing Noise (oppohoron -> অপহরন / অপহরণ)
        Self::rerank_degemination(lower, candidates, suggestion);

        // 7. Terminal Inflectional Suffixes (-ey -> -ে, independent এর -> ের)
        Self::rerank_terminal_inflections(lower, candidates, suggestion);

        // 8. Sanskrit/Indic 'v' / 'w' -> 'ব' (avatar -> abatar -> অবতার, vidya -> বিদ্যা)
        Self::rerank_indic_v_w(lower, candidates, suggestion);

        // 9. Participial Adjectives (-ito/-ita -> -িত, e.g. atyacharita -> অত্যাচারিত, adhito -> অধীত)
        Self::rerank_participial_it(lower, candidates, suggestion);

        // 10. Modern Standard Orthography (হ্রস্ব ু vs দীর্ঘ ূ normalization)
        Self::rerank_standard_orthography(lower, candidates, suggestion);

        // 11. Dictionary Word Preference over Non-Dictionary Candidates
        Self::rerank_dictionary_preference(lower, candidates, suggestion);
    }

    /// Fix 1: Initial Vowel Reranker
    /// When input starts with 'a', Avro deterministically generates 'আ-'.
    /// If an 'অ-' variant exists in the dictionary and the 'আ-' candidate is either
    /// not in the dictionary or has significantly lower frequency, promote 'অ-'.
    fn rerank_initial_vowel(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &PhoneticSuggestion,
    ) {
        if !input.starts_with('a') && !input.starts_with('o') {
            return;
        }

        let first = &candidates[0];
        if first.starts_with('আ') {
            // Check if there is an corresponding 'অ-' form
            let o_variant = format!("অ{}", &first['আ'.len_utf8()..]);

            // Check if o_variant is in candidates or dictionary
            let o_in_dict = suggestion.database.trie.contains_exact(&o_variant)
                || suggestion.database.is_exact_dictionary_word(&o_variant);
            let aa_in_dict = suggestion.database.trie.contains_exact(first)
                && suggestion.database.get_frequency(first) > 500;

            if o_in_dict && !aa_in_dict {
                // If o_variant is already in candidates, promote it to slot 0
                if let Some(pos) = candidates.iter().position(|c| c == &o_variant) {
                    let item = candidates.remove(pos);
                    candidates.insert(0, item);
                } else {
                    candidates.insert(0, o_variant);
                }
            } else if o_in_dict {
                let freq_o = suggestion.database.get_frequency(&o_variant);
                let freq_aa = suggestion.database.get_frequency(first);
                if freq_o >= freq_aa * 3 {
                    if let Some(pos) = candidates.iter().position(|c| c == &o_variant) {
                        let item = candidates.remove(pos);
                        candidates.insert(0, item);
                    } else {
                        candidates.insert(0, o_variant);
                    }
                }
            }
        }
    }

    /// Fix 2: Medial Inherent Vowel Reranker
    /// Banglish 'a' in non-initial syllables is often the inherent vowel /ɔ/.
    /// If removing an 'া' yields a stem with drastically higher dictionary frequency
    /// (e.g. অক্ষরকে vs অক্ষারকে), promote the higher frequency inherent vowel candidate.
    fn rerank_medial_inherent_vowel(
        _input: &str,
        candidates: &mut Vec<String>,
        suggestion: &PhoneticSuggestion,
    ) {
        if candidates[0].contains('া') {
            let first = candidates[0].clone();
            for (idx, _) in first.match_indices('া') {
                // Only consider medial positions, never terminal aa-kar
                if idx >= first.len() - 'া'.len_utf8() {
                    continue;
                }
                let mut alt = first.clone();
                alt.remove(idx);
                let freq_alt = suggestion.database.get_frequency(&alt);
                let freq_orig = suggestion.database.get_frequency(&first);
                if freq_alt > (freq_orig.max(50) * 5)
                    && !suggestion.database.is_exact_dictionary_word(&first)
                {
                    candidates.retain(|c| c != &alt);
                    candidates.insert(0, alt);
                    break;
                }
            }
        }
    }

    /// Fix 3: Clitic Particles (-o -> -ও, -i -> -ই)
    /// Only promote clitic candidate if the stem ends in a vowel (e.g. 'amio' -> 'আমিও', 'tumio' -> 'তুমিও', 'atiteo' -> 'অতীতেও')
    /// and the clitic candidate already exists in the candidate strip.
    /// Note: Do NOT trigger on general consonant + 'o' (e.g. 'kormo', 'shobdo') because 'o' is the inherent vowel /ɔ/.
    fn rerank_clitic_particles(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &PhoneticSuggestion,
    ) {
        if candidates.is_empty() {
            return;
        }

        if (input.ends_with("io") || input.ends_with("eo") || input.ends_with("ao"))
            && input.len() >= 4
        {
            let first = candidates[0].clone();
            if !first.ends_with('ও') {
                let target_clitic = format!("{}ও", first);
                if let Some(pos) = candidates.iter().position(|c| c == &target_clitic) {
                    let item = candidates.remove(pos);
                    candidates.insert(0, item);
                } else if let Some(pos) = candidates.iter().position(|c| {
                    c.ends_with('ও')
                        && (suggestion.database.is_exact_dictionary_word(c)
                            || suggestion.database.trie.contains_exact(c))
                }) {
                    if pos < 4 {
                        let item = candidates.remove(pos);
                        candidates.insert(0, item);
                    }
                }
            }
        }
    }

    /// Fix 4: Tatsama Retroflex Nasal Conjuncts (ন্দ -> ণ্ড)
    /// Typists type 'nd' for both 'ন্দ' and 'ণ্ড' (e.g. akhandata -> অখণ্ডতা, thanda -> ঠাণ্ডা, dando -> দণ্ড)
    fn rerank_retroflex_nasal(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &PhoneticSuggestion,
    ) {
        if !input.contains("nd") {
            return;
        }

        let first = candidates[0].clone();
        if first.contains("ন্দ") {
            let alt = first.replace("ন্দ", "ণ্ড");
            if suggestion.database.is_exact_dictionary_word(&alt)
                || suggestion.database.trie.contains_exact(&alt)
            {
                candidates.retain(|c| c != &alt);
                candidates.insert(0, alt);
                return;
            }
            // Also test dropping non-initial aa-kar with retroflex (e.g. আখান্দাতা -> অখণ্ডতা)
            if alt.contains('া') {
                let alt2 = alt.replace('া', "");
                if suggestion.database.is_exact_dictionary_word(&alt2)
                    || suggestion.database.trie.contains_exact(&alt2)
                {
                    candidates.retain(|c| c != &alt2);
                    candidates.insert(0, alt2);
                }
            }
        }
    }

    /// Fix 5: English Loanword Initial 'au' -> 'o' (auxford -> oxford -> অক্সফোর্ড, augustin -> অগাস্টিন)
    fn rerank_loanword_au(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &PhoneticSuggestion,
    ) {
        if input.starts_with("au") && input.len() >= 4 {
            let o_form = input.replacen("au", "o", 1);
            let o_conv = suggestion.convert_phonetic(&o_form);
            if !o_conv.is_empty()
                && (suggestion.database.is_exact_dictionary_word(&o_conv)
                    || suggestion.database.trie.contains_exact(&o_conv))
            {
                candidates.retain(|c| c != &o_conv);
                candidates.insert(0, o_conv);
            }
        }
    }

    /// Fix 6: De-gemination of Double Consonant Typing Noise (e.g. oppohoron -> অপহরন / অপহরণ)
    fn rerank_degemination(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &mut PhoneticSuggestion,
    ) {
        if candidates.is_empty() || suggestion.database.lookup_override(input).is_some() {
            return;
        }
        let first_orig = &candidates[0];
        if suggestion.database.is_exact_dictionary_word(first_orig)
            || suggestion.database.trie.contains_exact(first_orig)
        {
            return;
        }

        let double_consonants = ["pp", "bb", "kk", "tt", "dd", "mm", "ll", "ss", "gg"];
        for dc in double_consonants {
            if input.contains(dc) {
                let single = &dc[..1];
                let degeminated_input = input.replace(dc, single);
                let empty_mem = hashbrown::HashMap::new();
                let (alt_cands, _) =
                    suggestion.suggest(&degeminated_input, false, true, &empty_mem);
                if let Some(first_alt) = alt_cands.first() {
                    let first_orig = &candidates[0];
                    let freq_alt = suggestion.database.get_frequency(first_alt);
                    let freq_orig = suggestion.database.get_frequency(first_orig);
                    if freq_alt > freq_orig * 2
                        && suggestion.database.is_exact_dictionary_word(first_alt)
                    {
                        candidates.retain(|c| c != first_alt);
                        candidates.insert(0, first_alt.clone());
                        break;
                    }
                }
            }
        }
    }

    /// Fix 7: Terminal Inflectional Suffixes (-ey -> -ে, independent এর -> ের)
    /// e.g. oxfordey -> অক্সফোর্ডে, otitkaley -> অতীতকালে, ottacharer -> অত্যাচারের
    fn rerank_terminal_inflections(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &mut PhoneticSuggestion,
    ) {
        // A. Terminal -ey -> locative case -ে (e.g. oxfordey -> oxforde -> অক্সফোর্ডে)
        if input.ends_with("ey") && input.len() >= 4 {
            // If top candidate already ends with 'ে' and is valid, keep it!
            if candidates[0].ends_with('ে')
                && (suggestion.database.is_exact_dictionary_word(&candidates[0])
                    || suggestion.database.trie.contains_exact(&candidates[0]))
            {
                // Already optimal
            } else if let Some(pos) = candidates.iter().position(|c| {
                c.ends_with('ে')
                    && (suggestion.database.is_exact_dictionary_word(c)
                        || suggestion.database.trie.contains_exact(c))
            }) {
                // If an inflected candidate ending in 'ে' exists elsewhere in the candidate strip, promote it
                let item = candidates.remove(pos);
                candidates.insert(0, item);
            } else {
                // Otherwise try suggesting for stem + 'e'
                let stem = &input[..input.len() - 2];
                let e_input = format!("{}e", stem);
                let empty_mem = hashbrown::HashMap::new();
                let (alt_cands, _) = suggestion.suggest(&e_input, false, true, &empty_mem);
                if let Some(first_alt) = alt_cands.iter().find(|c| {
                    c.ends_with('ে')
                        && (suggestion.database.is_exact_dictionary_word(c)
                            || suggestion.database.trie.contains_exact(c))
                }) {
                    candidates.retain(|c| c != first_alt);
                    candidates.insert(0, first_alt.clone());
                }
            }
        } else if candidates.len() >= 2
            && input.ends_with('e')
            && !input.ends_with("ee")
            && !input.ends_with("ie")
            && input.len() >= 4
            && !candidates[0].ends_with('ে')
        {
            // A2. Terminal -e locative preservation (e.g. abosare -> অবসর vs অবসরে)
            let stem = candidates[0].as_str();
            if let Some(rel_pos) = candidates[1..].iter().position(|c| {
                c.strip_suffix('ে') == Some(stem)
                    && (suggestion.database.is_exact_dictionary_word(c)
                        || suggestion.database.trie.contains_exact(c))
            }) {
                let item = candidates.remove(rel_pos + 1);
                candidates.insert(0, item);
            }
        }

        // A3. Negation prefix 'অন-' preservation when 'an-'/'on-' input truncated to 'অ-' (e.g. anadhikar -> অধিকার vs অনধিকার)
        if candidates.len() >= 2
            && (input.starts_with("an") || input.starts_with("on"))
            && input.len() >= 6
            && !candidates[0].starts_with("অন")
        {
            if let Some(tail) = candidates[0].strip_prefix('অ') {
                if let Some(rel_pos) = candidates[1..].iter().position(|c| {
                    c.strip_prefix("অন") == Some(tail)
                        && (suggestion.database.is_exact_dictionary_word(c)
                            || suggestion.database.trie.contains_exact(c))
                }) {
                    let item = candidates.remove(rel_pos + 1);
                    candidates.insert(0, item);
                }
            }
        }

        // B. Dependent Genitive 'ের' vs Independent 'এর'
        // In Avro, consonant + er often yields '...এর' (e.g. 'অত্যাচারএর'). Fix to 'অত্যাচারের'.
        if let Some(stem) = candidates[0].strip_suffix("এর") {
            if stem.chars().count() >= 1 {
                let alt = format!("{}ের", stem);
                if suggestion.database.is_exact_dictionary_word(&alt)
                    || suggestion.database.trie.contains_exact(&alt)
                {
                    candidates.retain(|c| c != &alt);
                    candidates.insert(0, alt);
                    return;
                }
            }
        }

        // C. Terminal Genitive -er -> promote candidates ending in 'ের' / 'র'
        // Prevents locative -e (e.g. অবদমনে) from hijacking genitive -er (e.g. অবদমনের)
        if input.ends_with("er") && input.len() >= 4 && !candidates[0].ends_with('র') {
            if let Some(pos) = candidates.iter().position(|c| {
                (c.ends_with("ের") || c.ends_with('র'))
                    && (suggestion.database.is_exact_dictionary_word(c)
                        || suggestion.database.trie.contains_exact(c))
            }) {
                let item = candidates.remove(pos);
                candidates.insert(0, item);
            }
        }
    }

    /// Fix 8: Sanskrit / Indic 'v' / 'w' -> 'ব' (e.g. avatar -> অবতার, vidyalay -> বিদ্যালয়)
    fn rerank_indic_v_w(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &mut PhoneticSuggestion,
    ) {
        if input.len() < 3 {
            return;
        }

        // Sub-rule A: Typist typed 'b' (not 'bh'), but fuzzy layout chose 'ভ' (e.g. abatar -> আভাটার instead of অবতার)
        if input.contains('b') && !input.contains("bh") && candidates[0].contains('ভ') {
            if let Some(pos) = candidates.iter().position(|c| {
                c.contains('ব')
                    && (suggestion.database.is_exact_dictionary_word(c)
                        || suggestion.database.trie.contains_exact(c))
            }) {
                let item = candidates.remove(pos);
                candidates.insert(0, item);
            }
        }

        if !input.contains('v') && !input.contains('w') {
            return;
        }

        let first = &candidates[0];
        let freq_first = suggestion.database.get_frequency(first);

        let b_inputs = [
            input.replace(['v', 'w'], "b"),
            if let Some(stripped) = input.strip_prefix('a') {
                format!("o{}", stripped.replace(['v', 'w'], "b"))
            } else {
                String::new()
            },
        ];
        for b_input in &b_inputs {
            if b_input.is_empty() {
                continue;
            }
            let empty_mem = hashbrown::HashMap::new();
            let (alt_cands, _) = suggestion.suggest(b_input, false, true, &empty_mem);
            if let Some(valid_alt) = alt_cands.iter().find(|c| {
                // Reject autocomplete prefix extensions that add extra words (e.g. অদ্ভুতভাবে from advut)
                if c.chars().count() > input.chars().count() + 1 {
                    return false;
                }
                c.contains('ব')
                    && (suggestion.database.is_exact_dictionary_word(c)
                        || suggestion.database.trie.contains_exact(c))
            }) {
                let freq_alt = suggestion.database.get_frequency(valid_alt);
                if freq_alt > freq_first
                    || (!suggestion.database.is_exact_dictionary_word(first)
                        && freq_alt >= freq_first)
                {
                    candidates.retain(|c| c != valid_alt);
                    candidates.insert(0, valid_alt.clone());
                    return;
                }
            }
        }
    }

    /// Fix 9: Participial Adjective Suffixes (-ito / -ita -> -িত, e.g. atyacharita -> অত্যাচারিত, adhito -> অধীত)
    fn rerank_participial_it(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &mut PhoneticSuggestion,
    ) {
        if candidates.is_empty() {
            return;
        }

        // Sanskrit/Tatsama participial stems have length >= 6 and never end with dipthongs "oi"/"ai"
        if (input.ends_with("ito")
            || (input.ends_with("ita") && !input.ends_with("oita") && !input.ends_with("aita")))
            && input.len() >= 6
        {
            // First, if an exact participial candidate ending in -িত is already in candidates, promote it!
            if let Some(pos) = candidates.iter().position(|c| {
                c.ends_with("িত")
                    && (suggestion.database.is_exact_dictionary_word(c)
                        || suggestion.database.trie.contains_exact(c))
            }) {
                let item = candidates.remove(pos);
                candidates.insert(0, item);
            }

            let stem = &input[..input.len() - 3];
            let it_input = format!("{}it", stem);
            let empty_mem = hashbrown::HashMap::new();
            let (alt_cands, _) = suggestion.suggest(&it_input, false, true, &empty_mem);
            if let Some(first_alt) = alt_cands.first() {
                if suggestion.database.is_exact_dictionary_word(first_alt)
                    || suggestion.database.trie.contains_exact(first_alt)
                {
                    candidates.retain(|c| c != first_alt);
                    candidates.insert(0, first_alt.clone());
                }
            }
        }
    }

    /// Fix 10: Modern Standard Orthography (হ্রস্ব ু vs দীর্ঘ ূ normalization)
    /// Modern Bengali orthography (Bangla Academy standard) deprecates archaic dirgho-u 'ূ'
    /// in words like 'অন্তর্ভুক্ত' (vs archaic 'অন্তর্ভূক্ত') when the user typed single 'u'.
    fn rerank_standard_orthography(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &PhoneticSuggestion,
    ) {
        if candidates.is_empty() {
            return;
        }

        let first = candidates[0].clone();
        // Specifically fix archaic 'ভূক্ত' -> 'ভুক্ত' (e.g. অন্তর্ভূক্ত -> অন্তর্ভুক্ত)
        if first.contains("ভূক্ত") && !input.contains("oo") && !input.contains("uu") {
            let modern_variant = first.replace("ভূক্ত", "ভুক্ত");
            if suggestion
                .database
                .is_exact_dictionary_word(&modern_variant)
                || suggestion.database.trie.contains_exact(&modern_variant)
            {
                candidates.retain(|c| c != &modern_variant);
                candidates.insert(0, modern_variant);
                return;
            }
        }

        // Also check modern dirgho-i 'ী' vs hrosso-i 'ি' in standard words (e.g. অতীত vs অতিতেই)
        // when input has double 'ee' or 'ii' (e.g. otiiteo -> অতীতেও)
        if (input.contains("ee") || input.contains("ii")) && first.contains('ি') {
            let modern_i = first.replace('ি', "ী");
            if suggestion.database.is_exact_dictionary_word(&modern_i)
                || suggestion.database.trie.contains_exact(&modern_i)
            {
                candidates.retain(|c| c != &modern_i);
                candidates.insert(0, modern_i);
            }
        }
    }

    /// Fix 11: Dictionary Word Preference over Non-Dictionary Candidates
    /// If the current top candidate is NOT in the dictionary, but an exact dictionary
    /// candidate exists within the strip of similar length, promote it!
    fn rerank_dictionary_preference(
        input: &str,
        candidates: &mut Vec<String>,
        suggestion: &PhoneticSuggestion,
    ) {
        if candidates.is_empty() {
            return;
        }

        let first = &candidates[0];
        // Do not demote intentional clitics or inflectional suffixes that match user input
        if first.ends_with('ও') && input.ends_with('o') {
            return;
        }
        if first.ends_with('ে') && (input.ends_with('e') || input.ends_with("ey")) {
            return;
        }
        if first.ends_with('র') && (input.ends_with('r') || input.ends_with("er")) {
            return;
        }

        let first_is_dict = suggestion.database.is_exact_dictionary_word(first)
            || suggestion.database.trie.contains_exact(first);

        if !first_is_dict {
            let first_len = first.chars().count() as i32;
            if let Some(pos) = candidates.iter().position(|c| {
                let c_len = c.chars().count() as i32;
                (c_len - first_len).abs() <= 2
                    && suggestion.database.is_exact_dictionary_word(c)
                    && suggestion.database.get_frequency(c) >= 50
            }) {
                let item = candidates.remove(pos);
                candidates.insert(0, item);
            }
        }
    }
}
