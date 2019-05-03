import random

# quick example

NUMBER_OF_ITERATIONS = 100000
cumsum = 0
values = []

for i in range(NUMBER_OF_ITERATIONS):
    value = random.randint(1,101) * (1/NUMBER_OF_ITERATIONS)
    values.append(value)
    cumsum += value

print(cumsum)
print(max([i * NUMBER_OF_ITERATIONS for i in values]))
print(min([i * NUMBER_OF_ITERATIONS for i in values]))

