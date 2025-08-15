use rustpython_parser::{Parse, ast};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python_source = fs::read_to_string(
        "/Users/shahsaad.alam/projects/fernandos_PR/system_performance/pulse_sim/ionq/pulse_sim/Simulator/MagnusModel.py",
    )?;

    // Parse as a full Python program (Suite)
    let python_ast = ast::Suite::parse(&python_source, "<embedded>")?;
    // The second argument "<embedded>" is the source name,
    // useful for error reporting.

    // Now you have the AST in `python_ast` and can traverse it
    // or extract information as needed.
    println!("{:#?}", python_ast); // Print the AST for inspection

    Ok(())
}


lex(&python_source, rustpython_parser::Mode::Module).for_each(|token| {
    if let Ok(t) = token {
        println!("{:?}", t);
    } else {
        eprintln!("Error lexing token: {:?}", token);
    }
    total = total + 1;
    if total > 10 {
        return;
    }
});
