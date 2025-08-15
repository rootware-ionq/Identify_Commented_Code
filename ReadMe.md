## Identifying Commented Code in Python

In a lot of my PRs, I often accidentally leave commented out Python Code. Wrote this small Rust snippet to make a CLI tool that reads in a Python file, identifies comments in the code, checks to see if the comments seem like unused code or human comments, and prints the comments with the corresponding line numbers. For example, with `sample.py`, we get the output

```python
 ##############################
 Line #          Extracted Comments:
 ##############################
6        "# import matplotlib.pyplot as plt"
```

To run the CLI executable, you can first add the directory where it's located to your PATH variable, and then simply call

`% bring_out_your_dead sample.py`