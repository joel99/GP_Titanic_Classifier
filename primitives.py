# Some place for primitives

# [bool, float, float] -> float
def if_then_else(input, output1, output2):
    return output1 if input else output2

# [float, float] -> bool
def is_greater(input1, input2):
    return input1 > input2

def is_equal_to(input1, input2):
    return input1 == input2

def relu(input1):
    return input1 if input1 > 0 else 0

# Wrappers
# [float, float] -> float
def safe_division(input1, input2):
    if absolute(input2) < 1: # Prevent explosions
        return input1
    return input1 / input2        

# [float] -> float
def absolute(input1):
    return input1 if input1 > 0 else -1 * input1

# [float] -> float
def cube(input1): # Putting in pow is too much, system can't use safely
    return input1 ** 3

# [float, int] -> float
def safe_pow(input1, input2):
    if input1 <= 0:
        return 0
    return input1 ** min(input2, 4)

# Complex operators
# Forced if_then_else
# NEGATIVE EFFECT
# [bool, bool, float, float] -> float
def equal_conditional(input1, input2, output1, output2):
    return output1 if is_equal_to(input1, input2) else output2

# In range check
# [float, float, float] -> bool
def in_range(input1, input2, input3):
    return is_greater(input1, input2) and is_greater(input3, input1)