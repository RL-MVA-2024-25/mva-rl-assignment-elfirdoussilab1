

# Tests for each unique threshold in reward_thresholds
def test_1():
    """Test if the one environment performance meets the 3432807.680391572 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    return one_env_performance >= 3432807.680391572

def test_2():
    """Test if the one environment performance meets the 1e8 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    return one_env_performance >= 1e8

def test_3():
    """Test if the one environment performance meets the 1e9 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    return one_env_performance >= 1e9

def test_4():
    """Test if the one environment performance meets the 1e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    return one_env_performance >= 1e10

def test_5():
    """Test if the one environment performance meets the 2e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    return one_env_performance >= 2e10

def test_6():
    """Test if the one environment performance meets the 5e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    one_env_performance = lines[0]
    return one_env_performance >= 5e10

# Tests for each unique threshold in reward_dr_thresholds
def test_7():
    """Test if the DR environment performance meets the 1e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    dr_env_performance = lines[1]
    return dr_env_performance >= 1e10

def test_8():
    """Test if the DR environment performance meets the 2e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    dr_env_performance = lines[1]
    return dr_env_performance >= 2e10

def test_9():
    """Test if the DR environment performance meets the 5e10 threshold."""
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    dr_env_performance = lines[1]
    return dr_env_performance >= 5e10

thresholds = [3432807.680391572, 1e8, 1e9, 1e10, 2e10, 5e10, 1e10, 2e10, 5e10]
if __name__ == '__main__':
    #score = 0
    with open("score.txt") as file:
        lines = [float(line.rstrip()) for line in file]
    score_1 = lines[0]
    score_2 = lines[1]
    for i in range(6):
        print(score_1 >= thresholds[i])
    for i in range(3):
        print(score_2 >= thresholds[i + 6])
    #print(score)