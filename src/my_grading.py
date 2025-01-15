# I use this file to print the test I have passed so far
thresholds = [3432807.680391572, 1e8, 1e9, 1e10, 2e10, 5e10, 1e10, 2e10, 5e10]

if __name__ == '__main__':
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    score_1 = lines[0]
    score_2 = lines[1]
    for i in range(6):
        print(score_1 >= thresholds[i])
    for i in range(3):
        print(score_2 >= thresholds[i + 6])