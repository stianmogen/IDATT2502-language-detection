def max_deviation(matrix, limit):
    max_indicies = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j and matrix[i][j] >= limit:
                    max_indicies.append(i) if i not in max_indicies else max_indicies
                    max_indicies.append(j) if j not in max_indicies else max_indicies
    return sorted(max_indicies)