from src.matrix import Matrix

def test_multiply():
    a = Matrix([
        [2,3,],
        [1,1,],
        [0,1,],
        [1,1,],
    ])
    b = Matrix([
        [3,2,1,],
        [1,1,1,],
    ])
    c = a.multiply(b)
    assert c == Matrix([
        [9,7,5,],
        [4,3,2,],
        [1,1,1,],
        [4,3,2,],
    ])

def test_multiply2():
    a = Matrix([
        [1,2,3],
    ])
    b = Matrix([
        [1],
        [2],
        [3],
    ])
    c = a.multiply(b)
    assert c == Matrix([[14]])
