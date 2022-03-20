"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

This class implements functionality that is based on the data structure implemented in formula.py. This primarily consists of constructing
  label masks for any given formula. Because computing masks for CloseTo operators is computationally expensive, this class now contains
  a bunch of complicated code implementing various performance optimization measures for Close-To computations. I resent this optimization
  because it pushes the level of complexity above a level that I perceive as reasonable. Unfortunately, it's one of the more effective
  optimization measures so I'm not going to take it out, but I recommend not reading the details unless absolutely necessary.

A separate part of the functionality concerns the way the beam search in score_calculator.py searches through the space of all formulas. In
  the standard algorithm, formulas are only manipulated at the outermost layer -- i.e,. if the current formula is F and the new primitive
  formula is P, only (F and P), (F or P), and (F and not P) are considered. This makes it impossible to build e.g. (A and B) or (C and D).

In the Close algorithm, optimal formulas often have such a form, e.g. (A CLOSE TO (B OR C)) is a common form. Finding such formulas requires
  special code that, given formulas F and P, looks into the structure of F.
"""

import copy
import multiprocessing
import random
import time
import numpy as np

import S
import score_calculator as ScoreCalculator
import settings
import formulas.formula as F

manager = multiprocessing.Manager()

TIMES = manager.list()
ABORTIONS = manager.list()

close_indices = {}
adjacent_indices = {}

def init():
    """ Precomputes, for every location in the 7x7 cell, the set of locations implicated in a CLOSE-TO operation. """

    def minus(num1, num2):
        N = num1 - num2
        return 0 if N < 0 else N

    def plus(num1, num2):
        N = num1 + num2
        return 6 if N > 6 else N

    for y, x in np.ndindex((7,7)):
        C = []
        A = []

        C.append((y,x))
        A.append((y,x))

        C.append((y, minus(x, 1))) # O O O O O
        C.append((minus(y, 1), x)) # O O X O O
        C.append((y, plus(x, 1)))  # O X O X O
        C.append((plus(y, 1), x))  # O O X O O
        A.append((y, minus(x, 1))) # O O O O O
        A.append((minus(y, 1), x))
        A.append((y, plus(x, 1)))
        A.append((plus(y, 1), x))

        C.append((minus(y, 1), minus(x, 1))) # O O O O O
        C.append((minus(y, 1), plus(x, 1)))  # O X O X O
        C.append((plus(y, 1), plus(x, 1)))   # O O O O O
        C.append((plus(y, 1), minus(x, 1)))  # O X O X O
        A.append((minus(y, 1), minus(x, 1))) # O O O O O
        A.append((minus(y, 1), plus(x, 1)))
        A.append((plus(y, 1), plus(x, 1)))
        A.append((plus(y, 1), minus(x, 1)))

        C.append((y, minus(x, 2))) # O O X O O
        C.append((minus(y, 2), x)) # O O O O O
        C.append((y, plus(x, 2)))  # X O O O X
        C.append((plus(y, 2), x))  # O O O O O
                                   # O O X O O

        C.append((minus(y, 1), minus(x, 2))) # O X O X O
        C.append((minus(y, 2), minus(x, 1))) # X O O O X
        C.append((minus(y, 2), plus(x, 1)))  # O O O O O
        C.append((minus(y, 1), plus(x, 2)))  # X O O O X
        C.append((plus(y, 1), plus(x, 2)))   # O X O X O
        C.append((plus(y, 2), plus(x, 1)))
        C.append((plus(y, 2), minus(x, 1)))
        C.append((plus(y, 1), minus(x, 2)))

        close_indices[(y,x)] = list(dict.fromkeys(C))
        adjacent_indices[(y,x)] = list(dict.fromkeys(A))

def list_new_formulas(current, new, op):
    """
    Contains the code that, given an existing formula F, a new primitive formula P, and a formula class op, returns a list of all formulas
      that are considered in the beam search. We differentiate four cases:

    Case 0: the standard case, i.e., none of the below. The only new formula is F op P.

    Case 1: op = OR and F = (A >< B), where >< is binary. There are then three formulas that make sense, namely:
      (1) (A >< B) OR P
      (2) (A OR P) >< B
      (3) A >< (B OR P)
      Formula (2) may be found via Case 0 as well, but only if A OR P makes the beam. (3) can only be found like this.

    Case 2: op = CLOSE and F = A CLOSE TO B. In this case, we only consider A CLOSE TO B & P.

    Case 3: op = CLOSE and F = A CLOSE B. Doing something with this is arguable; nonetheless we currently return an empty list.

    Case 4: op = CLOSE and F.op is not CLOSE or CLOSE TO. There are again three formulas that make sense, namely:
      (1) F CLOSE P
      (2) F CLOSE TO P
      (3) P CLOSE TO F
      Here, the computations of the corresponding masks are correlated. Consequently, they are computed jointly by the
        function compute_close_masks(). This is also why CLOSE TO is not included as an operator in the outer loop.
"""
    F_ = current.__class__

    if F_ in [F.With, F.Close] and op in [F.Or, F.AndNot]: # >> Case 1 <<
        return [op(current, new), F_(op(current.left, new), current.right), F_(current.left, op(current.right, new))]
    elif F_ == F.CloseTo and op in [F.Or, F.AndNot]: # >> Case 1 with potential special case<<
        if len(current.right) == 1:
            # return [op(current_formula, new_p_formula), C(op(current_formula.left, new_p_formula), current_formula.right), C(current_formula.left, op(current_formula.right[0], new_p_formula))]
            return [op(current, new), F_(op(current.left, new), current.right), F_(current.left, op(current.right[0], new))]
        else:
            # Special case: we cannot do A CLOSE TO (B op C) because B is already a list of formulas (Q & P & R)
            return [op(current, new)]
    elif F_ == F.CloseTo and op == F.Close: # >> Case 2  (with restriction to primitive formulas) <<
        if all([isinstance(x, F.Leaf) for x in current.right]):
            new_formula = copy.deepcopy(current)
            new_formula.add(new)
            return [new_formula]
        else:
            return []
    elif F_ == F.Close and op == F.Close: # >> Case 3 <<
        return []
    elif op == F.Close: # >> Case 4 <<
        return [F.Close(current, new), F.CloseTo(current, new), F.CloseTo(new, current)]
    else: ## >> Case 0 <<
        return [op(current, new)]

def redundancy_check(formula, primitive_formula, combined_formula, operator, codes):
    """
    Checks for properties of a (formula, new primitive formula, new operator) triple that makes it evidently not useful (-> redundant) as
      another measure to optimize runtime.

    Note that finding *all* possible redundancies is a genuinely hard problem. This is supposed to be the 0.01/0.9 variant that solves 90%
      of the problem for 1% of the work. We primarily care about (A OR B) vs. (B OR A) which is handled by the to_code functionality,
      and (A OR A) as well as (A OR B) OR B which is handled by explicit checks.
    """

    if combined_formula.to_code() in codes:
        return True, 'code check'

    new_label = primitive_formula.val if isinstance(primitive_formula, F.Leaf) else primitive_formula.val.val # Leaf or Expand(Leaf)

    if isinstance(formula, F.Leaf):
        if formula.val == new_label: # This covers A op A, where op could be any of {AND, OR, WITH, CLOSE}
            return True, 'identical labels'

    if (isinstance(formula, F.Or) and operator == F.Or) or (isinstance(formula, F.And) and operator == F.And):
        labels = [subformula.val for subformula in formula.formulas if isinstance(subformula, F.Leaf)]
        if new_label in labels:
            return True, 'duplicate in AND/OR expression'

    # Found no obvious redundancy, so... *shrugs*
    return False, None

def compute_composite_mask(formula, abort=None, neuron_i=None):
    """
    Returns the mask of an arbitrary formula, or None if the formula has a CLOSE-TO operation at the top level that was aborted early
      (see comments in score_calculator.py).

    The code in this class closely mirrors the data structure in formula.py. E.g., if the formula has an AND operator at the highest level,
      we recursively compute the masks of all components of the AND and use the in-built function np.all to combine those masks. As usual,
      most of the complexity comes in with the code doing performance optimization, in this case for early abortions of CLOSE-TO operators.

    Throughout program execution, a global record of known masks (based on their codes) is maintained in ScoreCalculator.known_masks. A
      needing the mask for a given formula is not required to check if the mask is already known; instead the check happens here.
    """
    # assert settings.EASY_MODE or (neuron_i == None)


    code = formula.to_code(neuron_i=neuron_i)

    if code in ScoreCalculator.known_masks.keys():
        return ScoreCalculator.known_masks[code]

    if isinstance(formula, F.And):
        bitmask_array = np.array([compute_composite_mask(subformula, neuron_i=neuron_i) for subformula in formula.formulas])
        return np.all(bitmask_array, axis=0)
    elif isinstance(formula, F.Or):
        bitmask_array = np.array([compute_composite_mask(subformula, neuron_i=neuron_i) for subformula in formula.formulas])
        return np.any(bitmask_array, axis=0)
    elif isinstance(formula, F.Expand):
        return compute_expanded_mask(compute_composite_mask(formula.val, neuron_i=neuron_i))
    elif isinstance(formula, F.With):
        left = compute_composite_mask(formula.left, neuron_i=neuron_i)
        right = compute_composite_mask(formula.right, neuron_i=neuron_i)
        masks_result = np.zeros((S.n_images, *S.mask_shape))
        for (i, (left_bitmask, right_bitmask)) in enumerate(zip(left, right)):
            if right_bitmask.sum() > 0:
                masks_result[i] = left_bitmask  # I thought this would just put a reference but I've tested it and it seems to copy everything
            # if the sum is 0, then we want there to be all 0s, so no need to do anything given how masks_result was initialized
        return masks_result
    elif isinstance(formula, F.AndNot):
        left = compute_composite_mask(formula.left, neuron_i=neuron_i)
        right = compute_composite_mask(formula.right, neuron_i=neuron_i)
        return np.logical_and(left, np.logical_not(right))
    elif isinstance(formula, F.CloseTo):
        R = cc2m_special_return(compute_composite_mask(formula.left, neuron_i=neuron_i), [compute_composite_mask(X, neuron_i=neuron_i) for X in formula.right], abort=abort)
        return package_potentially_aborted_mask(R, abort)
    elif isinstance(formula, F.Close):
        mask_left = compute_composite_mask(formula.left, neuron_i=neuron_i)
        mask_right = compute_composite_mask(formula.right, neuron_i=neuron_i)

        if abort:
            stop, min_value, neuron_mask = abort

            L, LR = cc2m_special_return(mask_left, mask_right, abort=abort)
            R, RR = cc2m_special_return(mask_right, mask_left, abort=abort)

            if LR or RR: # at least one of the masks was aborted
                if settings.SCORE_FUNCTION_OPTIMIZATION(neuron_mask[:stop], np.logical_or(L[:stop], R[:stop])) < min_value:
                    return None
                else:
                    # the union is good enough, so recompute the masks without aborting and return the full union
                    return np.logical_or(cc2m_special_return(mask_left, mask_right), cc2m_special_return(mask_right, mask_left))
            else:
                # Both masks are by themselves good enough. While it's possible for the union to be worse, there is no benefit to be gained
                #   by checking for that since we already did the work.
                return np.logical_or(L, R)
        else:
            return np.logical_or(cc2m_special_return(mask_left, mask_right), cc2m_special_return(mask_right, mask_left))
    else:
        x = formula.val if isinstance(formula, F.Leaf) else ""
        raise ValueError(f"Must be passed formula, not {type(formula)} with value {x}")

def compute_expanded_mask(mask):
    """
    Deals with the EXPAND operator (written {formula-name}-X. Note that, much like the CLOSE-TO operator, this operator is also
      computationally expensive (because it has to look into individual images), which is why we do not treat it as a regular operator that
      can be applied to arbitrary formulas (even though this is logically coherent and could be done with the implementation given here).
      Instead, we precompute expanded versions of primitive masks only and then treat those as part of the extended set of available labels.
      See comments in loaders/mask_loader.py for details.

    The parameter settings.EXPANSION_THRESHOLD specifies how masks are expanded (i.e., how many neighbors must be activated to activate a
      previously not-activated cell). This parameter becomes part of the name of the results folder, e.g., FL3_iou_X3_close indicates that
      the operator was used (hence the 'X") and that settings.EXPANSION_THRESHOLD = 3.
    """
    expanded_mask = np.copy(mask)

    for image_mask, original_image_mask in zip(expanded_mask, mask):
        if np.any(original_image_mask):
            for y, x in np.ndindex(S.mask_shape):
                if image_mask[y, x]:
                    continue

                count = 0

                for Y, X in adjacent_indices[(y, x)]:
                    if original_image_mask[Y, X]:
                        count += 1
                    if count >= settings.EXPANSION_THRESHOLD:
                        image_mask[y, x] = True
                        break

    return expanded_mask

def package_potentially_aborted_mask(R, abort):
    """
    Takes a result of cc2m_special_return, which is either a mask or (if aborting was considered) a tuple (mask, B) where B
    is a flag indicating whether computation was aborted early or not. Returns None if it was aborted and the mask otehrwise
    """
    if abort and R[1]:  # aborting was considered and executed
        R = None
    elif abort and not R[1]:  # aborting was considered but not executed
        R = R[0]
    return R

def compute_close_masks(old_formula, new_formula, redundancies, abort=None, neuron_i=None):
    """
    Computes the formulas corresponding to an old formula, a new formula, and a CLOSE operator, as well as their corresponding masks
    Saves the masks in the global field and returns a list of the formulas.
    """

    if not settings.EASY_MODE:
        neuron_i = None


    mask_left = compute_composite_mask(old_formula, neuron_i=neuron_i)
    mask_right = compute_composite_mask(new_formula, neuron_i=neuron_i)

    if redundancies == [True, False, True]: # Only A close to B
        L = cc2m_special_return(mask_left, mask_right, abort=abort)

        if abort:
            if L[1]:
                return [None, None, None]
            else:
                L = L[0]

        formula_CT_regular = F.CloseTo(old_formula, new_formula)
        ScoreCalculator.known_masks[formula_CT_regular.to_code(neuron_i=neuron_i)] = L
        return [None, formula_CT_regular, None]  # if aborting was not considered, then R is already a proper mask
    elif redundancies == [True, True, False]: # Only B close to A
        R = cc2m_special_return(mask_right, mask_left, abort=abort)

        if abort:
            if R[1]:
                return [None, None, None]
            else:
                R = R[0]

        formula_CT_reverse = F.CloseTo(new_formula, old_formula)
        ScoreCalculator.known_masks[formula_CT_reverse.to_code(neuron_i=neuron_i)] = R
        return [None, None, formula_CT_reverse]  # if aborting was not considered, then R is already a proper mask
    else: # A close B is required, so we have to compute both A close to B and B close to A anyway
        if abort:
            stop, min_value, neuron_mask = abort
            L, LR = cc2m_special_return(mask_left, mask_right, abort=abort)
            R, RR = cc2m_special_return(mask_right, mask_left, abort=abort)
            sym = settings.SCORE_FUNCTION_OPTIMIZATION(neuron_mask[:stop], np.logical_or(L[:stop], R[:stop])) >= min_value
        else:
            L = cc2m_special_return(mask_left, mask_right)
            R = cc2m_special_return(mask_right, mask_left)

        if abort and LR:
            formula_CT_regular = None
            if sym:
                L = cc2m_special_return(mask_left, mask_right)
        else:
            formula_CT_regular = F.CloseTo(old_formula, new_formula)
            ScoreCalculator.known_masks[formula_CT_regular.to_code(neuron_i=neuron_i)] = L

        if abort and RR:
            formula_CT_reverse = None
            if sym:
                R = cc2m_special_return(mask_right, mask_left)
        else:
            formula_CT_reverse = F.CloseTo(new_formula, old_formula)
            ScoreCalculator.known_masks[formula_CT_reverse.to_code(neuron_i=neuron_i)] = R

        if abort and not sym:
            formula_CT_symmetrical = None
        else:
            formula_CT_symmetrical = F.Close(new_formula, old_formula)
            ScoreCalculator.known_masks[formula_CT_symmetrical.to_code(neuron_i=neuron_i)] = np.logical_or(L, R)

        return [formula_CT_symmetrical, formula_CT_regular, formula_CT_reverse]

def cc2m_special_return(left, right, abort=None):
    """
    if abort is None, returns the resulting mask.
    Otherwise, returns the resulting mask and a boolean indicating whether or not it was aborted early.
    If it was aborted early, the mask will have all False from settings.CUTOFF_SAMPLES + 1 onward.
    """

    t = time.time()
    result = np.zeros_like(left)

    # right can either be a mask of shape (n_images, im_width, im_height), or a list of such masks. Make it so it's always a list.
    if len(np.shape(right)) == 3:
        right = [right]

    # transform right from mask_index -> (image_index -> image_mask) to image_index -> (mask_index -> image_mask)
    right = np.swapaxes(np.array(right), 0, 1)

    if abort:
        stop, min_score, neuron_mask = abort


    # -------------------------------------------------------------------------------------------------------------------------------------
    # - For each image mask, there are two ways it can be computed:                                                                       -
    # - One: go through the right mask, 'expand' each activated position into the area that it 'accepts'; then XOR the resulting mask     -
    # -    with the left image mask to obtain the result                                                                                  -
    # - Two: go through the left mask, for each activated position, search for a proximate activated position in the right mask. If there -
    # -    is one, activate the position in the result mask.                                                                              -
    # - The first approach has runtime approximately 25 times the number of activated positions in the right mask; the second approach 25 -
    # -    times the number of activated positions in the left mask. Thus, we make the choice based on which mask activates more often.   -
    # - Since the second approach often allows terminating the search early, privilege it unless the right mask is at least 3 smaller.    -
    # -------------------------------------------------------------------------------------------------------------------------------------
    for i, (left_image_mask, right_image_masks) in enumerate(zip(left, right)):

        # -- || FIRST WAY || -- #
        if len(right_image_masks) == 1 and right_image_masks[0].sum() < left_image_mask.sum() - 2:
            mask_proximity_to_right_image_mask = np.zeros(shape=S.mask_shape, dtype=bool)
            for y, x in np.ndindex(S.mask_shape):
                if right_image_masks[0, y, x]:
                    for Y, X in close_indices[(y,x)]:
                        mask_proximity_to_right_image_mask[Y, X] = True
            result[i] = np.logical_and(left_image_mask, mask_proximity_to_right_image_mask)

        # -- || SECOND WAY || -- #
        else:
            result_image_mask = np.zeros(S.mask_shape, dtype=bool) # Do this instead of writing into left_image_mask because that changes the original

            for y, x in np.ndindex(S.mask_shape):
                if left_image_mask[y, x]:
                    all_masks_have_close_cells = True # this needs to be true as well for the result to be true

                    for right_image_mask in right_image_masks:
                        found_close_cell_for_mask = False

                        for Y, X in close_indices[(y,x)]:
                            if right_image_mask[Y, X]:
                                found_close_cell_for_mask = True
                                break

                        if not found_close_cell_for_mask:
                            all_masks_have_close_cells = False
                            break

                    if all_masks_have_close_cells:
                        result_image_mask[y,x] = True
            result[i] = result_image_mask

        # ----------------------------------------------------------
        # - Check if the result is sufficiently bad to abort early -
        # ----------------------------------------------------------
        if abort and i == stop:
            score = settings.SCORE_FUNCTION_OPTIMIZATION(neuron_mask[:stop], result[:stop])
            if score < min_score:
                if random.random() < 0.1:
                    ABORTIONS.append(-1)
                TIMES.append(time.time() - t)
                return result, True
            else:
                if random.random() < 0.1:
                    ABORTIONS.append(1)

    # --------------------------------------------------------------------------------------
    # - Check again if abort was not None to know whether to also return the boolean value -
    # --------------------------------------------------------------------------------------
    # Note that, if the parentheses around result, False are removed, `False if abort else result` is evaluated first, which is undesirable
    TIMES.append(time.time() - t)
    return (result, False) if abort else result