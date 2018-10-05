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