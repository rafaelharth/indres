"""
Contains the logic for label formulas (e.g., silver-screen CLOSE TO wall).
The existence of CLOSE-TO operators has made this much more complex than it used to be in the standard algorithm.

One of the new functionalities is the to_code method in the formula interface. This returns a unique integer for a formula. (It's unique
  in the sense that two formulas that are logically different are guaranteed to have different codes. Code duplicates for syntactically
  different but logically equivalent formulas such as "(A or B)" and "(B or A)" is possible and considered a feature.)

The code that achieves this makes use of prime numbers, hence the primes.txt and primes.memmap files.
"""