import numpy as np

def main():
    A = get_matrix()
    print(matrix_tutorial(A))

def get_matrix():
    mat = []
    [n, m] = [int(x) for x in input().strip().split(" ")]
    for i in range(n):
        row = [int(x) for x in input().strip().split(" ")]
        mat.append(row)
    return np.array(mat)

def matrix_tutorial(A):

    # 아래 코드를 완성하세요.
    print(A.shape)
    B = A.transpose()
    print("!!!!")
    print(B.shape)
    try:
        C = np.linalg.inv(B)
        number=np.sum(C>0)
        return number
    except:
        return "not invertible"


if __name__ == "__main__":
    main()
