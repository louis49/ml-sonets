# Initial value and coefficient
value = 0.005
coefficient = 0.99#618

# Applying the coefficient 0.99 600 times
for _ in range(600):
    value *= coefficient
value2 = 0.000012
print(value)