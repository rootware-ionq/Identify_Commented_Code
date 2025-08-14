use comment_parser::CommentParser;
use std::{env, fs, process};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <file.ris>", args[0]);
        process::exit(1);
    }
    let filepath = &args[1];
    let python_code = fs::read_to_string(filepath).unwrap_or_else(|_| {
        eprintln!("Error: Could not read file {}", filepath);
        process::exit(1);
    });

    let rules = comment_parser::get_syntax("python").unwrap();
    let parser = CommentParser::new(&python_code, rules);

    for comment in parser {
        if matches!(comment.clone(), LineComment) {
            println!("{:#?}", comment);
        }
    }
}
