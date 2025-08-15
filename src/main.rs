use comment_parser::CommentParser;
use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <file.py>", args[0]);
        process::exit(1);
    }
    let filepath = &args[1];
    let python_code = fs::read_to_string(filepath).unwrap_or_else(|_| {
        eprintln!("Error: Could not read file {}", filepath);
        process::exit(1);
    });

    let rules = comment_parser::get_syntax("python").unwrap();
    let parser = CommentParser::new(&python_code, rules);

    println!(" ##############################");
    println!(" Line # \t Extracted Comments:");
    println!(" ##############################");
    for comment in parser {
        match comment {
            comment_parser::Event::LineComment(span, text) => {
                if is_probable_code(&span) && !text.contains("# type: ignore") {
                    println!(
                        "{:#?}\t {:#?}",
                        python_code[..python_code.find(span).unwrap()]
                            .lines()
                            .count()
                            + 1,
                        span
                    );
                }
            }

            _ => {}
        }
    }
}
/// Heuristic function to detect code in a comment
fn is_probable_code(s: &str) -> bool {
    // Looks for Python keywords, operators, or function calls
    let keywords = [
        "def ", "return", "import ", "for ", "while ", "if ", "=", "print(", "(",
    ];
    let wrong_words = ["the"];
    keywords.iter().any(|kw| s.contains(kw)) && !wrong_words.iter().any(|w| s.contains(w))
}
