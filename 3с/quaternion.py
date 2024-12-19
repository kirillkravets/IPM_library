import  numpy as np

class Quaternion:
    # инициализируем кватернион по данным np.array()
    def __init__(self, q = np.array([1.0, 0.0, 0.0, 0.0])):
        self.q = np.array(q, dtype=float)
        self.normalize()

    # нормируем np.array() для __init__
    def normalize(self):
        norm = np.linalg.norm(self.q)

        if norm > 0:
            self.q = self.q / norm
        else:
            self.q = np.array([1, 0, 0, 0], dtype=float)

    # переопределение оператора * для кватернионного умножения
    def __mul__(self, other):
        # other - объект класса Quaternion
        if isinstance(other, Quaternion):
            q1 = self.q
            q2 = other.q

            qMult = np.zeros(4, dtype=float)
            qMult[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
            qMult[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])

            return Quaternion(qMult)
        else:
            raise TypeError("Можно умножать только на кватернион")

    def vecMulQuat(self, vec):
        # other - объект класса Quaternion
        if isinstance(vec, np.ndarray):
            if len(vec) != 3:
                raise TypeError("Неверный размер массива")
            quatVec = Quaternion(np.array([0.0, *vec], dtype = float))
            return (quatVec * self).q
        else:
            raise TypeError("Можно умножать только на np.ndarray")

    def quatMulVec(self, vec):
        # other - объект класса Quaternion
        if isinstance(vec, np.ndarray):
            if len(vec) != 3:
                raise TypeError("Неверный размер массива")
            quatVec = Quaternion(np.array([0.0, *vec], dtype = float))
            return (self * quatVec).q
        else:
            raise TypeError("Можно умножать только на np.ndarray")


    # метод перевода вектора, заданного в ИСК, в ССК
    def IF2BF(self, vecIF):

        quatVecIF = Quaternion(np.array([0.0, *vecIF], dtype=float))
        quatVecBF = self.getConjugate() * quatVecIF * self
        vecBF = quatVecBF.q[1:]

        return vecBF
    # нахождение сопряжённого кватерниона
    def getConjugate(self):
        return Quaternion(np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]]))

