import json


class Matrix(object):
    def __init__(self, native_nested_list):
        self.dimensions = [len(native_nested_list), len(native_nested_list[0])]
        # self._get_dimensionality(native_nested_list, self.dimensionality)
        self._arr = native_nested_list

    @staticmethod
    def create(dimensions, initializer=lambda: 0.0):
        result = []
        for i in range(dimensions[0]):
            result.append([])
            for j in range(dimensions[1]):
                result[i].append(initializer())
        return Matrix(result)

    def _get_dimensionality(self, native_nested_list, dimensionality):
        if type(native_nested_list) is not list:
            return
        
        dimensionality.append(len(native_nested_list))
        self.get_dimensionality(native_nested_list[0], dimensionality)

    def apply_function(self, func):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                self._arr[i][j] = func(self._arr[i][j])

    def multiply(self, other):
        if self.dimensions[-1] != other.dimensions[0]:
            raise ValueError((
                "Internal dimensions don't match: "
                "%s vs %s", (self.dimensions[-1], other.dimensions[0])
            ))

        result = Matrix.create([self.dimensions[0], other.dimensions[-1]])

        for i in range(self.dimensions[0]):
            for j in range(other.dimensions[-1]):
                res = 0
                for k in range(self.dimensions[1]):
                    res += self._arr[i][k] * other._arr[k][j]
                result._arr[i][j] = res

        return result

    def multiply_scalar(self, scalar):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                self[i][j] *= scalar
        return self

    def add(self, other):
        if self.dimensions != other.dimensions:
            raise ValueError((
                "Internal dimensions don't match: "
                "%s vs %s", (self.dimensions, other.dimensions)
            ))
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                self[i][j] += other[i][j]
        return self

    def transpose(self):
        result = Matrix.create(list(reversed(self.dimensions)), lambda: 0.0)
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                result[j][i] = self[i][j]
        return result

    def squared_l2_norm(self):
        total = 0
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                total += self[i][j] * self[i][j]
        return total

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __repr__(self):
        return json.dumps(self._arr)

    def __getitem__(self, key):
        return self._arr[key]

    def print_debug(self):
        for row in self._arr:
            output = "|"
            for num in row:
                output += ("%.2f" % num).center(7, ' ')
            output += "|"
            print(output)
        print('')
