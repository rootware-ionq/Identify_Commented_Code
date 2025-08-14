use rustpython_parser::{Parse, Tok, ast, lexer::lex};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python_source = fs::read_to_string(
        "/Users/shahsaad.alam/projects/coupled_noise_models/system_performance/pulse_sim/ionq/pulse_sim/CoupledModel/CoupledModel.py",
    )?;

    for token_result in lex(&python_source, rustpython_parser::Mode::Module) {
        if let Ok((Tok::String { value: comment, .. }, range)) = token_result {
            let stripped = comment.trim_start_matches('#').trim();

            // Skip empty comments
            // if stripped.is_empty() {
            //     continue;
            // }

            // Check if the comment looks like Python code
            if is_probable_code(stripped) {
                // Calculate the line number
                let line_number = python_source[..range.start().into()].matches('\n').count() + 1;
                println!(
                    "Line {} may contain commented-out code: {}",
                    line_number, stripped
                );
            }
        }
    }

    Ok(())
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
