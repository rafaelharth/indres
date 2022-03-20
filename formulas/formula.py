"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

Contains the code for formula classes.

In general, formulas have sufficient structure to reconstruct complex masks. E.g., if there is a formula object corresponding to the string

                         ((building CLOSE TO shop window-X) OR pantry)

then the formula interface should 'figure out' what mask this formula encodes, provided it is given access to masks for the primitive
  formulas 'building' and 'shop window-X' and 'pantry'. This class provides a data structure that makes this possible; the code that
  implements mask construction on that basis is in utils.py.

This class is the only part of the program were we I've made extensive use of object orientation. The inheritance graph is as follows:

F ----- Leaf
  -
  ----- UnaryNode ---- Expand
  -
  ---- BinaryNode ---- AndNot
  -               -
  -               ---- With
  -               -
  -               ---- Close
  -
  ----- NNaryNode ---- Or
  -               -
  -               ---- And
  -
  ---- CloseTo

We often have references directly to classes and treat them as operators. I.e., if op is a pointer to a class that inherits from
  BinaryNode or NNaryNode, then one can write op(node1, node2) to create a new formula.
"""

import copy
import os
import numpy as np

import S
import settings
import util.util

primes = []
label_namer = lambda x : x

def init():
    """ Ensures access to the prime numbers as a .memmap file. (They are needed for the to_code() method of F.) """
    global primes

    if os.path.exists('formulas/primes.memmap'):
        primes = np.memmap("formulas/primes.memmap", dtype=int, mode="r", shape=(50000,))
    else:
        primes = np.memmap("formulas/primes.memmap", dtype=int, mode="w+", shape=(50000,))

        with open("formulas/primes.txt", 'r') as f:
            primes_list = [int(prime) for prime in (f.read().split())]
            primes[0:50000] = primes_list[0:50000]

class F:
    """ The root class of formulas. This functions both as an interface (making the compiler know that all classes that inherit have
implemented __len__ and to_str and to_list), and as the implementation of to_code and to_short_code, which are the same for all formulas."""

    def to_code(self, ignore_expansion=False, neuron_i=None):
        # ---------------------------------------------------------------------------------------------------------------------------------
        # - First, note that coding just labels is insufficient: (A OR B) AND C   and   (A OR C) AND B are different formulas. The same   -
        # -   example shows that also coding operators is still not enough. Rather, we need to code (elem, position) pairs, where elem is -
        # -   either a label or an operator. This requires a mapping all possible elements to unique integers. The mapping is as follows: -
        # -------------------------------                                                                                                 -
        # - EXPAND   <-> 0              -                                                                                                 -
        # - WITH     <-> S.n_labels + 1 -                                                                                                 -
        # - AND NOT  <-> S.n_labels + 2 -                                                                                                 -
        # - CLOSE    <-> S.n_labels + 3 -                                                                                                 -
        # - OR       <-> S.n_labels + 4 -                                                                                                 -
        # - AND      <-> S.n_labels + 5 -                                                                                                 -
        # - CLOSE_TO <-> S.n_labels + 6 -                                                                                                 -
        # - label #n <-> n              -                                                                                                 -
        # -------------------------------                                                                                                 -
        # - Let f be the function implementing the above. With that, we obtain a coding of a formula by first unpacking it into a list    -
        # -   [f(e0), f(e1), f(e2), ...], where the ei are the elements. (Whenever order matters, the original order is preserved;        -
        # -   whenever it does not, the elements are sorted in ascending order.) The code of such a list is then given by                 -
        # -   PROD_{i = 0}^n primes[k * shift + f(e_i)], where shift = S.n_labels + 7.                                                    -
        # ---------------------------------------------------------------------------------------------------------------------------------

        if not settings.EASY_MODE:
            neuron_i = None

        code = int(1)
        shift = S.n_labels + 7

        for (i, f_of_elem) in enumerate(self.to_list(ignore_expansion)):
            code *= int(primes[i * shift + f_of_elem])

        if neuron_i is not None:
            code = f"{neuron_i}_{code}"

        return code

    def to_short_code(self):
        # Alternate implementation disregarding order. WIll NOT be injective but may work ok in practice.
        code = 1

        for (i, elem) in enumerate(self.to_list()):
            code *= int(primes[elem])

        return code

    def __len__(self):
        return 0

    def to_str(self):
        return ""

    def to_list(self, ignore_expansion=False):
        return []

class Leaf(F):
    """ To ensure that all formulas are built out of formulas, we differentiate between formula-that-only-contains-a-label-and-nothing-else
(which would be a member of this class) and the label itself (which is just an integer). All formulas bottom out at Leaf objects. To
access the label contained in a Leaf object, type {leaf_object_name}.val. """

    def __init__(self, val):
        self.val = val

    def to_list(self, ignore_expansion=False):
        return [self.val]

    def to_str(self, namer=(lambda name: 'NONE' if name is None else label_namer(name))):
        return namer(self.val)

    def __len__(self):
        return 1

class UnaryNode(F):
    op = None

    def __init__(self, val):
        self.val = val

    def to_list(self, ignore_expansion=False):
        if ignore_expansion:
            return self.val.to_list(ignore_expansion)
        else:
            return [0] + self.val.to_list(ignore_expansion)

    def to_str(self, namer=(lambda name: 'NONE' if name is None else label_namer(name))):
        return f"{self.val.to_str(namer)}-X"

    def __len__(self):
        return len(self.val)

class BinaryNode(F):
    op = None

    def __init__(self, left, right):
        # left = A and not B
        # # left.left = A
        # # left.right = B
        # # right = C
        # right = C
        # standard res = (A and not B) and not C
        # better res = A and not (B or C)
        # super().__init__()
        if isinstance(self, AndNot) and isinstance(left, AndNot):
            self.left = left.left
            self.right = Or(left.right, right)
        else:
            self.left = left
            self.right = right

    def to_str(self, namer=(lambda name: 'NONE' if name is None else label_namer(name))):
        left_name = self.left.to_str(namer)
        right_name = self.right.to_str(namer)

        return f"({left_name} {self.op} {right_name})"

    def to_list(self, ignore_expansion=False):
        if isinstance(self, With):
            operator_prime_index = S.n_labels + 1
        elif isinstance(self, AndNot):
            operator_prime_index = S.n_labels + 2
        elif isinstance(self, Close):
            operator_prime_index = S.n_labels + 3
        else:
            raise ValueError(f"Type {self.__class__} not recognized (BinaryNode)")

        # Since Close is a symmetrical operator, make it such that A CLOSE B and B CLOSE A yield the same code.
        if isinstance(self, Close) and self.right.to_code(ignore_expansion) < self.left.to_code(ignore_expansion):
            return self.right.to_list(ignore_expansion) + [operator_prime_index] + self.left.to_list(ignore_expansion)
        else:
            return self.left.to_list(ignore_expansion) + [operator_prime_index] + self.right.to_list(ignore_expansion)

    def __len__(self):
        return len(self.left) + len(self.right)

class NNaryNode(F):
    op = None

    def __init__(self, one, two):
        self.formulas = [one, two]

        if self.__class__ == one.__class__:
            new_formulas = copy.deepcopy(one.formulas)
            new_formulas.append(two)
            self.formulas = new_formulas

    def to_str(self, namer=(lambda name: 'NONE' if name is None else label_namer(name))):
        res = "("

        for (i, formula) in enumerate(self.formulas):
            res += formula.to_str(namer)
            if i < len(self.formulas) - 1:
                res += f" {self.op} "

        return res + ")"

    def to_list(self, ignore_expansion=False):
        if isinstance(self, Or):
            operator_prime_index = S.n_labels + 4
        elif isinstance(self, And):
            operator_prime_index = S.n_labels + 5
        else:
            raise ValueError(f"Type {self.__class__} not recognized (NNaryNode)")

        formula_list = [formula.to_list(ignore_expansion) for formula in self.formulas]
        formula_list.sort()

        return [operator_prime_index] + util.util.flatten(formula_list)

    def __len__(self):
        return sum([len(formula) for formula in self.formulas])

    def add(self, formula):
        self.formulas.append(formula)

class CloseTo(F):
    op = "CLOSE TO"

    def __init__(self, one, two):
        self.left = one
        self.right = two if isinstance(two, list) else [two]

    def to_str(self, namer=(lambda name: 'NONE' if name is None else label_namer(name))):
        res = f"({self.left.to_str(namer)} {self.op} "

        for (i, formula) in enumerate(self.right):
            res += formula.to_str(namer)
            if i < len(self.right) - 1:
                res += " & "

        return res + ")"

    def to_list(self, ignore_expansion=False):
        operator_prime_index = S.n_labels + 6

        right_list = [formula.to_list(ignore_expansion) for formula in self.right]
        right_list.sort()

        return [operator_prime_index] + self.left.to_list(ignore_expansion) + util.util.flatten(right_list)

    def add(self, formula):
        self.right.append(formula)

    def __len__(self):
        return len(self.left) + sum([len(R) for R in self.right])

class Expand(UnaryNode):
    op = "X"

class AndNot(BinaryNode):
    op = "AND NOT"

class With(BinaryNode):
    op = "WITH"

class Close(BinaryNode):
    op = "CLOSE"

class Or(NNaryNode):
    op = "OR"

class And(NNaryNode):
    op = "AND"