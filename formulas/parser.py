"""
Contains the code for parsing a formula string (and returning the formula object). An example of a valid string is
                  ((building CLOSE TO shop window-X) OR pantry)
"""

import re

import formulas.formula as F
import loaders.metadata_loader as MetaData

def parse(formula_string):
    def create_parenthesis_bitmask(str_):
        """
        Returns a bitmask that is TRUE at every point not inside of a subexpression.
        I.e., "(ASD)F((WER)T)F" returns an array True at positions 5 and -1 and False everywhere else
        """
        bitmask = [False] * len(str_)

        open_parentheses = 0

        for i, letter in enumerate(str_):
            if letter == '(':
                open_parentheses += 1
            elif letter == ')':
                open_parentheses -= 1
            elif open_parentheses == 0:  # Otherwise, the closing parenthesis is marked as True. Harmless in our case but not logical.
                bitmask[i] = True

        return bitmask

    def indices(str_, substr):
        """
        Returns the starting indices of all occurences of substr in str_.
        For example, ("asdasdasd", "sd") returns [1, 4, 7].
        If substr does not occur in str, returns [].
        """

        return [m.start() for m in re.finditer(substr, str_)]

    if all(op_string not in formula_string for op_string in ['OR', 'AND', 'WITH', 'NEXT TO', 'CLOSE TO', 'CLOSE']):

        # We have a primitive formulas. This should either be '{label_name}' or '(NOT {label_name})'
        if '&' in formula_string:
            # Note that we only allow primitive formulas to be part the right side of a CLOSE TO expression, which means that the , symbol
            #   cannot appear in subformulas
            return [parse(x.strip()) for x in formula_string.split('&')]


        if formula_string[-2:] == '-X':
            formula_string = formula_string[:-2]
            expansion = True
        else:
            expansion = False

        formula = F.Leaf(MetaData.map_label_name_2_label[formula_string])

        if expansion:
            formula = F.Expand(formula)

        return formula

    # We have a composite formula. This one is always surrounded by parentheses, so remove them.
    formula_string = formula_string[1:-1]

    bitmask = create_parenthesis_bitmask(formula_string)

    # --------------------------------------------------------------------------------------------------------------------------------
    # - Find the starting indices of the logical connector on the highest level. Note that there can be only one of {OR, AND, WITH}. -
    # - (If it's WITH, then there is also only one, but we don't make use of this.)                                                  -
    # --------------------------------------------------------------------------------------------------------------------------------
    or_indices = [i for i in indices(formula_string, 'OR') if bitmask[i]]
    and_not_indices = [i for i in indices(formula_string, 'AND NOT') if bitmask[i]]
    and_indices = [i for i in indices(formula_string, 'AND') if bitmask[i]]
    with_indices = [i for i in indices(formula_string, 'WITH') if bitmask[i]]
    closeto_indices = [i for i in indices(formula_string, 'CLOSE TO') if bitmask[i]]
    close_indices = [i for i in indices(formula_string, 'CLOSE') if bitmask[i]]

    # It's unfortunate that CLOSE and CLOSE TO now have a common prefix. However, if we first check for CLOSE TO and the checks are mutually
    # exclusive, it should not cause any errors.
    if or_indices:
        operator_indices = or_indices
        operator = F.Or
        len_of_operator = 2
    elif and_not_indices:
        operator_indices = and_not_indices
        operator = F.AndNot
        len_of_operator = 7
    elif and_indices:
        operator_indices = and_indices
        operator = F.And
        len_of_operator = 3
    elif with_indices:
        operator_indices = with_indices
        operator = F.With
        len_of_operator = 4
    elif closeto_indices:
        operator_indices = closeto_indices
        operator = F.CloseTo
        len_of_operator = 8
    elif close_indices:
        operator_indices = close_indices
        operator = F.Close
        len_of_operator = 5
    else:
        raise ValueError(f"Formula {formula_string} doesn't have AND or OR or WITH or CLOSE TO on the highest level.")


    # ----------------------------------------------------------------------------------------------------------------------
    # - Compute list of (start, end) index pairs out of indices_. Note that there needs to be one more than len(indices_). -
    # ----------------------------------------------------------------------------------------------------------------------
    operator_indices = [-len_of_operator-1] + operator_indices + [len(formula_string) + 1] # chosen to make index_start and index_end work out
    formula_indices = []

    for i in range(1, len(operator_indices)):
        # To verify this, consider 01 OR 67 OR CD
        index_start = operator_indices[i-1] + len_of_operator + 1
        index_end = operator_indices[i] - 1

        formula_indices.append((index_start, index_end))

    subformulas = [parse(formula_string[s:e]) for (s, e) in formula_indices]


    composite_formula = operator(subformulas[0], subformulas[1])
    if len(subformulas) > 2:
        for subformula in subformulas[2:]:
            composite_formula.add(subformula)

    return composite_formula
