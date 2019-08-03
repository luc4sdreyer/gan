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

def test_multiply_scalar():
    a = Matrix([
        [2,3,],
        [1,1,],
        [0,1,],
        [1,1,],
    ])
    c = a.multiply_scalar(-1)
    assert c == Matrix([
        [-2,-3,],
        [-1,-1,],
        [-0,-1,],
        [-1,-1,],
    ])

    b = Matrix([
        [3,2,1,],
        [1,1,1,],
    ])
    c = b.multiply_scalar(3.0)
    assert c == Matrix([
        [9.0,6.0,3.0,],
        [3.0,3.0,3.0,],
    ])

def test_add():
    a = Matrix([
        [2,3,],
        [1,1,],
    ])
    b = Matrix([
        [0,1,],
        [1,-1,],
    ])
    c = a.add(b)
    assert c == Matrix([
        [2,4,],
        [2,0,],
    ])

def test_transpose():
    assert Matrix([
        [2,1,0],
        [3,1,1],
    ]) == Matrix([
        [2,3,],
        [1,1,],
        [0,1,],
    ]).transpose()

    assert Matrix([
        [1,2,3,],
    ]) == Matrix([
        [1,],
        [2,],
        [3,],
    ]).transpose()
