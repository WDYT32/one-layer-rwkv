import math

vocab_size = 21
d_model = 256
target_params = 1142784 # rough target from earlier

a = 4
b = 4 * d_model + 10 + vocab_size
c = vocab_size * d_model - target_params

# Quadratic formula
H_float = (-b + math.sqrt(b**2 - 4*a*c)) / (2*a)
H = round(H_float)

print(f"Calculated H: {H}")

# verify
P_calc = vocab_size * d_model + 4 * H * d_model + 4 * H**2 + 8 * H + 2 * H + H * vocab_size
print(f"Calculated P: {P_calc}")
print(f"Target P: {target_params}")

