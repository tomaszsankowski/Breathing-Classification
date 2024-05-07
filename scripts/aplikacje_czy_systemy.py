import random

random_numbers = [random.random() for _ in range(10)]

for _ in range(10):
    random_num = random.random()
    if random_num > 0.5:
        print("APLIKACJE")
    else:
        print("SYSTEMY")
