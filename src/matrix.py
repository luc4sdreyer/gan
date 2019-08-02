class Matrix(object):
    def __init__(self, native_nested_list):
        self.dimentions = [len(native_nested_list), len(native_nested_list[0])]
        # self._get_dimensionality(native_nested_list, self.dimensionality)
        self._arr = native_nested_list

    def _get_dimensionality(self, native_nested_list, dimensionality):
        if type(native_nested_list) is not list:
            return
        
        dimensionality.append(len(native_nested_list))
        self.get_dimensionality(native_nested_list[0], dimensionality)

    def multiply(self, other):
        if self.dimentions[-1] != other.dimentions[0]:
            raise ValueError(f"Internal dimensions don't match: {
                self.dimentions[-1]} vs {other.dimentions[0]}")

        result = []
        for i in range(self.dimentions[0]):
            result.append([])
            for j in range(other.dimentions[-1]):
                result[i].append(0)

        for i in range(self.dimentions[0]):
            for j in range(other.dimentions[-1]):
                res = 0
                for k in range(self.dimentions[1]):
                    res += self._arr[i][k] * self._arr[k][j]
                result[i][j] = res

        return Matrix(res)
