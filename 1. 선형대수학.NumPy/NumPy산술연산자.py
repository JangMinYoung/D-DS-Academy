import numpy as np

def main():
    print(matrix_tutorial())

def matrix_tutorial():
    A = np.array([[1,4,5,8], [2,1,7,3], [5,4,5,9]])

    # 아래 코드를 작성하세요.
    sum_val=A.sum()

    A = A/sum_val

    return A.var()

if __name__ == "__main__":
    main()
