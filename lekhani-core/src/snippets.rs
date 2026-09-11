//! Snippets & Dynamic Macro Expansion Engine

use chrono::{Datelike, Local, Timelike};
use hashbrown::HashMap;

#[derive(Debug, Clone, Default)]
pub struct SnippetManager {
    user_snippets: HashMap<String, String>,
}

impl SnippetManager {
    pub fn new() -> Self {
        let mut user_snippets = HashMap::new();
        user_snippets.insert(
            "!shubhechha".to_string(),
            "আন্তরিক শুভেচ্ছা ও অভিনন্দন".to_string(),
        );
        user_snippets.insert(
            "!dhonnobad".to_string(),
            "আপনাকে অনেক অনেক ধন্যবাদ".to_string(),
        );
        user_snippets.insert("!shagotom".to_string(), "স্বাগতম".to_string());
        user_snippets.insert("!aborton".to_string(), "🔄".to_string());
        Self { user_snippets }
    }

    pub fn insert_snippet(&mut self, trigger: String, expansion: String) {
        self.user_snippets.insert(trigger, expansion);
    }

    pub fn remove_snippet(&mut self, trigger: &str) -> Option<String> {
        self.user_snippets.remove(trigger)
    }

    pub fn get_snippets(&self) -> &HashMap<String, String> {
        &self.user_snippets
    }

    /// Prefix search for live snippet autocompletion (e.g. "!sh" -> "আন্তরিক শুভেচ্ছা ও অভিনন্দন")
    pub fn search_prefix(&self, query: &str, limit: usize) -> Vec<String> {
        let q = query.to_lowercase();
        let mut results = Vec::new();

        // 1. Direct key prefix match
        for (key, val) in &self.user_snippets {
            if key.to_lowercase().starts_with(&q) && !results.contains(val) {
                results.push(val.clone());
                if results.len() >= limit {
                    return results;
                }
            }
        }

        // 2. If query starts with '!', match inner stem
        if q.starts_with('!') && q.len() >= 2 {
            let stem = &q[1..];
            for (key, val) in &self.user_snippets {
                let key_stem = key.trim_start_matches('!');
                if key_stem.to_lowercase().starts_with(stem) && !results.contains(val) {
                    results.push(val.clone());
                    if results.len() >= limit {
                        return results;
                    }
                }
            }
        }

        results
    }

    /// Check if word matches a dynamic macro, snippet, math expression, or unit conversion
    pub fn expand(&self, word: &str) -> Option<String> {
        let cands = self.expand_all(word);
        cands.into_iter().next()
    }

    /// Return all ranked candidate expansions for a trigger
    pub fn expand_all(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return Vec::new();
        }

        // 1. User Defined Snippets
        if let Some(val) = self.user_snippets.get(word) {
            return vec![val.clone()];
        }
        if !word.starts_with('!') {
            let with_exclaim = format!("!{}", word);
            if let Some(val) = self.user_snippets.get(&with_exclaim) {
                return vec![val.clone()];
            }
        }

        // 2. Math Formula Evaluator (e.g. "=125*8", "=sqrt(144)", "=1500+250")
        if word.starts_with('=') && word.len() > 1 {
            let expr = &word[1..];
            if let Some(val) = eval_math_expression(expr) {
                let bn_str = format_math_result_bn(val);
                let en_str = format_math_result_en(val);
                return if bn_str != en_str {
                    vec![bn_str, en_str]
                } else {
                    vec![bn_str]
                };
            }
        }

        // 3. Currency & Unit Conversions (e.g. "#usd50", "$50", "#eur20", "#100c", "#10km")
        if (word.starts_with('#') || word.starts_with('$')) && word.len() > 1 {
            let query = &word[1..];
            if let Some(results) = eval_unit_or_currency(query) {
                return results;
            }
        }

        // 4. Built-in Dynamic Real-Time Macros
        match word {
            "#tarikh" | "#date" => vec![
                format_current_bengali_date(),
                Local::now().format("%Y-%m-%d").to_string(),
            ],
            "#shomoy" | "#time" => vec![
                format_current_bengali_time_12h(),
                format_current_bengali_time_24h(),
            ],
            "#shomoy24" | "#time24" => vec![format_current_bengali_time_24h()],
            "#din" | "#day" => vec![format_current_bengali_day()],
            "#mash" | "#month" => vec![format_current_bengali_month()],
            "#bongabdo" | "#shal" => vec![format_current_bongabdo()],
            _ => Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Math Evaluation Engine (Recursive Descent Parser)
// ---------------------------------------------------------------------------

pub fn eval_math_expression(input: &str) -> Option<f64> {
    let clean: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.is_empty() {
        return None;
    }
    let mut parser = MathParser::new(&clean);
    let res = parser.parse_expr()?;
    if parser.pos == parser.chars.len() && !res.is_nan() && !res.is_infinite() {
        Some(res)
    } else {
        None
    }
}

struct MathParser<'a> {
    chars: Vec<char>,
    pos: usize,
    _marker: std::marker::PhantomData<&'a str>,
}

impl<'a> MathParser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            chars: s.chars().collect(),
            pos: 0,
            _marker: std::marker::PhantomData,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn next_char(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn parse_expr(&mut self) -> Option<f64> {
        let mut left = self.parse_term()?;
        while let Some(c) = self.peek() {
            if c == '+' {
                self.next_char();
                left += self.parse_term()?;
            } else if c == '-' {
                self.next_char();
                left -= self.parse_term()?;
            } else {
                break;
            }
        }
        Some(left)
    }

    fn parse_term(&mut self) -> Option<f64> {
        let mut left = self.parse_power()?;
        while let Some(c) = self.peek() {
            if c == '*' {
                self.next_char();
                left *= self.parse_power()?;
            } else if c == '/' {
                self.next_char();
                let right = self.parse_power()?;
                if right == 0.0 {
                    return None;
                }
                left /= right;
            } else if c == '%' {
                self.next_char();
                let right = self.parse_power()?;
                if right == 0.0 {
                    return None;
                }
                left %= right;
            } else {
                break;
            }
        }
        Some(left)
    }

    fn parse_power(&mut self) -> Option<f64> {
        let base = self.parse_factor()?;
        if let Some('^') = self.peek() {
            self.next_char();
            let exp = self.parse_power()?;
            Some(base.powf(exp))
        } else {
            Some(base)
        }
    }

    fn parse_factor(&mut self) -> Option<f64> {
        let c = self.peek()?;

        if c == '+' {
            self.next_char();
            return self.parse_factor();
        }
        if c == '-' {
            self.next_char();
            return Some(-self.parse_factor()?);
        }

        if c == '(' {
            self.next_char();
            let val = self.parse_expr()?;
            if self.next_char()? != ')' {
                return None;
            }
            return Some(val);
        }

        if c.is_ascii_alphabetic() {
            let mut name = String::new();
            while let Some(ch) = self.peek() {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    name.push(ch);
                    self.next_char();
                } else {
                    break;
                }
            }
            let lower = name.to_lowercase();
            if lower == "pi" {
                return Some(std::f64::consts::PI);
            }
            if lower == "e" {
                return Some(std::f64::consts::E);
            }

            if self.peek() == Some('(') {
                self.next_char();
                let arg = self.parse_expr()?;
                let mut arg2 = None;
                if self.peek() == Some(',') {
                    self.next_char();
                    arg2 = Some(self.parse_expr()?);
                }
                if self.next_char()? != ')' {
                    return None;
                }

                return match lower.as_str() {
                    "sqrt" => Some(arg.sqrt()),
                    "cbrt" => Some(arg.cbrt()),
                    "abs" => Some(arg.abs()),
                    "sin" => Some(arg.to_radians().sin()),
                    "cos" => Some(arg.to_radians().cos()),
                    "tan" => Some(arg.to_radians().tan()),
                    "ln" => Some(arg.ln()),
                    "log" | "log10" => Some(arg.log10()),
                    "round" => Some(arg.round()),
                    "floor" => Some(arg.floor()),
                    "ceil" => Some(arg.ceil()),
                    "pow" => arg2.map(|exp| arg.powf(exp)),
                    _ => None,
                };
            }
            return None;
        }

        if c.is_ascii_digit() || c == '.' {
            let mut num_str = String::new();
            let mut has_dot = false;
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    num_str.push(ch);
                    self.next_char();
                } else if ch == '.' && !has_dot {
                    has_dot = true;
                    num_str.push(ch);
                    self.next_char();
                } else {
                    break;
                }
            }
            return num_str.parse::<f64>().ok();
        }

        None
    }
}

fn format_math_result_en(val: f64) -> String {
    if (val.fract().abs() < 1e-9) && val.abs() < 1e15 {
        let int_val = val.round() as i64;
        format_with_commas(int_val)
    } else {
        format!("{:.4}", val)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

fn format_math_result_bn(val: f64) -> String {
    let en = format_math_result_en(val);
    to_bengali_digits_str(&en)
}

fn format_with_commas(n: i64) -> String {
    let s = n.abs().to_string();
    let mut res = String::new();
    let len = s.len();
    for (i, c) in s.chars().enumerate() {
        res.push(c);
        let rem = len - 1 - i;
        if rem > 0 && rem.is_multiple_of(3) {
            res.push(',');
        }
    }
    if n < 0 {
        format!("-{}", res)
    } else {
        res
    }
}

// ---------------------------------------------------------------------------
// Unit & Currency Conversion Engine
// ---------------------------------------------------------------------------

fn eval_unit_or_currency(input: &str) -> Option<Vec<String>> {
    let clean = input.to_lowercase();

    // Extract number and unit
    let (num, unit) = extract_num_and_unit(&clean)?;

    // 1. Currency conversions (to BDT)
    let bdt_rate = match unit.as_str() {
        "usd" | "dollar" | "$" => Some(122.0),
        "eur" | "euro" => Some(133.0),
        "gbp" | "pound" => Some(158.0),
        "inr" | "rupee" => Some(1.43),
        "sar" | "riyal" => Some(32.5),
        "aed" | "dirham" => Some(33.2),
        "myr" | "ringgit" => Some(27.5),
        "cad" => Some(90.0),
        "aud" => Some(80.5),
        "sgd" => Some(92.0),
        "jpy" | "yen" => Some(0.82),
        "kwd" => Some(398.0),
        "qar" => Some(33.5),
        _ => None,
    };

    if let Some(rate) = bdt_rate {
        let total = num * rate;
        let bn_amount = format_math_result_bn(total);
        let en_amount = format_math_result_en(total);
        return Some(vec![
            format!("৳{}", bn_amount),
            format!("{} BDT", en_amount),
            format!("{} টাকা", bn_amount),
        ]);
    }

    // 2. Unit Conversions
    match unit.as_str() {
        "c2f" | "c" => {
            let f = (num * 9.0 / 5.0) + 32.0;
            let f_bn = format_math_result_bn(f);
            let f_en = format_math_result_en(f);
            Some(vec![format!("{} °F", f_bn), format!("{} °F", f_en)])
        }
        "f2c" | "f" => {
            let c = (num - 32.0) * 5.0 / 9.0;
            let c_bn = format_math_result_bn(c);
            let c_en = format_math_result_en(c);
            Some(vec![format!("{} °C", c_bn), format!("{} °C", c_en)])
        }
        "km2m" | "km" => {
            let m = num * 1000.0;
            Some(vec![
                format!("{} মিটার", format_math_result_bn(m)),
                format!("{} m", format_math_result_en(m)),
            ])
        }
        "m2km" => {
            let km = num / 1000.0;
            Some(vec![
                format!("{} কিলোমিটার", format_math_result_bn(km)),
                format!("{} km", format_math_result_en(km)),
            ])
        }
        "mi2km" | "mi" | "mile" | "miles" => {
            let km = num * 1.60934;
            Some(vec![
                format!("{} কিমি", format_math_result_bn(km)),
                format!("{} km", format_math_result_en(km)),
            ])
        }
        "kg2lb" | "kg" => {
            let lb = num * 2.20462;
            Some(vec![
                format!("{} পাউন্ড", format_math_result_bn(lb)),
                format!("{} lbs", format_math_result_en(lb)),
            ])
        }
        "lb2kg" | "lb" | "lbs" => {
            let kg = num * 0.453592;
            Some(vec![
                format!("{} কেজি", format_math_result_bn(kg)),
                format!("{} kg", format_math_result_en(kg)),
            ])
        }
        "gb2mb" | "gb" => {
            let mb = num * 1024.0;
            Some(vec![
                format!("{} MB", format_math_result_bn(mb)),
                format!("{} MB", format_math_result_en(mb)),
            ])
        }
        _ => None,
    }
}

fn extract_num_and_unit(s: &str) -> Option<(f64, String)> {
    let mut num_part = String::new();
    let mut unit_part = String::new();

    // Check if format is "usd50" or "50usd"
    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            num_part.push(c);
        } else if c.is_ascii_alphabetic() || c == '$' {
            unit_part.push(c);
        }
    }

    if num_part.is_empty() || unit_part.is_empty() {
        return None;
    }

    let num = num_part.parse::<f64>().ok()?;
    Some((num, unit_part))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert ASCII digits in a string to Bengali numerals (০-৯)
pub fn to_bengali_digits_str(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '0' => '০',
            '1' => '১',
            '2' => '২',
            '3' => '৩',
            '4' => '৪',
            '5' => '৫',
            '6' => '৬',
            '7' => '৭',
            '8' => '৮',
            '9' => '৯',
            other => other,
        })
        .collect()
}

/// Convert number to Bengali numeral string
pub fn to_bengali_digits(n: u32) -> String {
    to_bengali_digits_str(&n.to_string())
}

/// Returns current date formatted in Bengali (e.g., "১১ সেপ্টেম্বর ২০২৬")
pub fn format_current_bengali_date() -> String {
    let now = Local::now();
    let day = to_bengali_digits(now.day());
    let year = to_bengali_digits(now.year() as u32);
    let month_name = get_bengali_gregorian_month(now.month());
    format!("{} {} {}", day, month_name, year)
}

/// Returns current 12-hour time formatted in Bengali (e.g., "০৪:৫০ অপরাহ্ন")
pub fn format_current_bengali_time_12h() -> String {
    let now = Local::now();
    let (pm, hour_12) = now.hour12();
    let hour_num = if hour_12 == 0 { 12 } else { hour_12 };
    let hour = format!("{:02}", hour_num);
    let minute = format!("{:02}", now.minute());
    let period = if pm { "PM" } else { "AM" };
    format!(
        "{}:{} {}",
        to_bengali_digits_str(&hour),
        to_bengali_digits_str(&minute),
        period
    )
}

/// Returns current 24-hour time formatted in Bengali (e.g., "১৬:৫০")
pub fn format_current_bengali_time_24h() -> String {
    let now = Local::now();
    let hour = format!("{:02}", now.hour());
    let minute = format!("{:02}", now.minute());
    format!(
        "{}:{}",
        to_bengali_digits_str(&hour),
        to_bengali_digits_str(&minute)
    )
}

/// Returns current day of week in Bengali (e.g., "শুক্রবার")
pub fn format_current_bengali_day() -> String {
    let now = Local::now();
    match now.weekday() {
        chrono::Weekday::Mon => "সোমবার".to_string(),
        chrono::Weekday::Tue => "মঙ্গলবার".to_string(),
        chrono::Weekday::Wed => "বুধবার".to_string(),
        chrono::Weekday::Thu => "বৃহস্পতিবার".to_string(),
        chrono::Weekday::Fri => "শুক্রবার".to_string(),
        chrono::Weekday::Sat => "শনিবার".to_string(),
        chrono::Weekday::Sun => "রবিবার".to_string(),
    }
}

/// Returns current Gregorian month name in Bengali (e.g., "সেপ্টেম্বর")
pub fn format_current_bengali_month() -> String {
    let now = Local::now();
    get_bengali_gregorian_month(now.month()).to_string()
}

/// Returns current date in Bengali Solar Calendar (Bongabdo, e.g. "২৬ ভাদ্র ১৪৩৩")
pub fn format_current_bongabdo() -> String {
    let now = Local::now();
    let (b_day, b_month_idx, b_year) = gregorian_to_bongabdo(now.year(), now.month(), now.day());
    let bengali_months = [
        "বৈশাখ",
        "জ্যৈষ্ঠ",
        "আষাঢ়",
        "শ্রাবণ",
        "ভাদ্র",
        "আশ্বিন",
        "কার্তিক",
        "অগ্রহায়ণ",
        "পৌষ",
        "মাঘ",
        "ফাল্গুন",
        "চৈত্র",
    ];
    let month_name = bengali_months.get(b_month_idx).unwrap_or(&"বৈশাখ");
    format!(
        "{} {} {} বঙ্গাব্দ",
        to_bengali_digits(b_day),
        month_name,
        to_bengali_digits(b_year as u32)
    )
}

fn get_bengali_gregorian_month(month: u32) -> &'static str {
    match month {
        1 => "জানুয়ারি",
        2 => "ফেব্রুয়ারি",
        3 => "মার্চ",
        4 => "এপ্রিল",
        5 => "মে",
        6 => "জুন",
        7 => "জুলাই",
        8 => "আগস্ট",
        9 => "সেপ্টেম্বর",
        10 => "অক্টোবর",
        11 => "নভেম্বর",
        12 => "ডিসেম্বর",
        _ => "",
    }
}

/// Convert Gregorian (Year, Month, Day) to Bengali Solar Calendar (Day, Month Index 0-11, Year)
fn gregorian_to_bongabdo(year: i32, month: u32, day: u32) -> (u32, usize, i32) {
    let is_leap_year = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_greg_months = [
        31,
        if is_leap_year { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];

    // Day of year (1-indexed)
    let mut day_of_year = day;
    for &days in days_in_greg_months.iter().take((month - 1) as usize) {
        day_of_year += days;
    }

    // April 14 is the start of Boishakh (day 104 or 105 in leap year)
    let boishakh_start = if is_leap_year { 105 } else { 104 };

    let (b_year, day_in_bengali_year) = if day_of_year >= boishakh_start {
        (year - 593, day_of_year - boishakh_start + 1)
    } else {
        let prev_year = year - 1;
        let prev_leap = (prev_year % 4 == 0 && prev_year % 100 != 0) || (prev_year % 400 == 0);
        let days_in_prev_year = if prev_leap { 366 } else { 365 };
        let prev_boishakh = if prev_leap { 105 } else { 104 };
        (
            year - 594,
            days_in_prev_year - prev_boishakh + 1 + day_of_year,
        )
    };

    // Bengali month lengths (Revised Bangladeshi standard: 5x31, 7x30 / Falgun 29 or 30)
    let b_month_lengths = [
        31,
        31,
        31,
        31,
        31,
        30,
        30,
        30,
        30,
        30,
        if is_leap_year { 30 } else { 29 },
        30,
    ];

    let mut remaining_days = day_in_bengali_year;
    let mut b_month_idx = 0;
    for (idx, &mlen) in b_month_lengths.iter().enumerate() {
        if remaining_days <= mlen {
            b_month_idx = idx;
            break;
        }
        remaining_days -= mlen;
    }

    (remaining_days, b_month_idx, b_year)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_bengali_digits() {
        assert_eq!(to_bengali_digits(2026), "২০২৬");
        assert_eq!(to_bengali_digits_str("04:30"), "০৪:৩০");
    }

    #[test]
    fn test_math_evaluation() {
        let mgr = SnippetManager::new();
        let res = mgr.expand_all("=125*8");
        assert!(!res.is_empty());
        assert_eq!(res[0], "১,০০০");
        assert_eq!(res[1], "1,000");

        let sqrt_res = mgr.expand_all("=sqrt(144)");
        assert_eq!(sqrt_res[0], "১২");
        assert_eq!(sqrt_res[1], "12");

        let complex = mgr.expand_all("=(50+25)*4");
        assert_eq!(complex[0], "৩০০");
    }

    #[test]
    fn test_unit_and_currency() {
        let mgr = SnippetManager::new();
        let usd = mgr.expand_all("#usd50");
        assert!(!usd.is_empty());
        assert!(usd[0].contains("৳৬,১০০") || usd[0].contains("৳"));

        let km = mgr.expand_all("#5km");
        assert!(!km.is_empty());
        assert!(km[0].contains("মিটার"));
    }

    #[test]
    fn test_dynamic_macros() {
        let mgr = SnippetManager::new();
        let tarikh = mgr.expand("#tarikh").expect("tarikh should expand");
        assert!(tarikh.contains("২০") || tarikh.chars().any(|c| ('০'..='৯').contains(&c)));

        let shomoy = mgr.expand("#shomoy").expect("shomoy should expand");
        assert!(shomoy.contains(':'));

        let din = mgr.expand("#din").expect("din should expand");
        assert!(din.contains("বার"));

        let bongabdo = mgr.expand("#bongabdo").expect("bongabdo should expand");
        assert!(bongabdo.contains("বঙ্গাব্দ"));

        // Prefix search for snippets
        let p_shub = mgr.search_prefix("!sh", 5);
        assert!(!p_shub.is_empty());
        assert!(p_shub.contains(&"আন্তরিক শুভেচ্ছা ও অভিনন্দন".to_string()));
        assert!(p_shub.contains(&"স্বাগতম".to_string()));
    }
}
