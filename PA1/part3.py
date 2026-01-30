import numpy as np

# e^(v_the * c_y) / 
# (e^(v_the * c_dog) + e^(v_the * c_cat) + e^(v_the * c_a) + e^(v_the * c_the))

cy_label = ['dog', 'cat', 'a', 'the']
cy = [np.array((0, 1)), np.array((0, 1)), np.array((1, 0)), np.array((1, 0))]
def skipgram(v, c):
    return np.exp(v @ c) / np.sum([np.exp(v @ c_y) for c_y in cy])

for i in range(1, 10):
    v = np.array((-i, i))
    if np.allclose([skipgram(v, c_y) for c_y in cy], [0.5, 0.5, 0, 0], atol=0.01):
        print("Part 3 Q1 3b:")
        print(f"Found v: {v}")
        for c_y, label in zip(cy, cy_label):
            print(f"P({label} | the) = {skipgram(v, c_y)}")
        break
print()
print("Part 3 Q2 3d:")
print(f"P(the | dog) = {skipgram(np.array((2, -2)), cy[3])}")
print(f"P(a | dog) = {skipgram(np.array((2, -2)), cy[2])}")
print(f"P(the | cat) = {skipgram(np.array((2, -2)), cy[3])}")
print(f"P(a | cat) = {skipgram(np.array((2, -2)), cy[2])}")
print(f"P(dog | the) = {skipgram(np.array((-2, 2)), cy[0])}")
print(f"P(cat | the) = {skipgram(np.array((-2, 2)), cy[1])}")
print(f"P(dog | a) = {skipgram(np.array((-2, 2)), cy[0])}")
print(f"P(cat | a) = {skipgram(np.array((-2, 2)), cy[1])}")
print(f"P(dog | dog) = {skipgram(np.array((2, -2)), cy[0])}")
print(f"P(cat | cat) = {skipgram(np.array((2, -2)), cy[1])}")
print(f"P(dog | cat) = {skipgram(np.array((2, -2)), cy[0])}")
print(f"P(cat | dog) = {skipgram(np.array((2, -2)), cy[1])}")
print(f"P(the | the) = {skipgram(np.array((-2, 2)), cy[3])}")
print(f"P(a | a) = {skipgram(np.array((-2, 2)), cy[2])}")
print(f"P(the | a) = {skipgram(np.array((-2, 2)), cy[3])}")
print(f"P(a | the) = {skipgram(np.array((-2, 2)), cy[2])}")
