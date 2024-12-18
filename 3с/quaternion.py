import  numpy as np

class Quaternion:
    def __init__(self, q = np.array([1.0, 0.0, 0.0, 0.0])):
        self.q = np.array(q, dtype=float)

        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.q)

        if norm > 0:
            self.q = self.q / norm
        else:
            self.q = np.array([1, 0, 0, 0], dtype=float)

    def getValue(self):
        return self.q

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            q1 = self.q
            q2 = other.q

            qMult = np.zeros(4)
            qMult[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
            qMult[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])

            return Quaternion(qMult)
        else:
            raise TypeError("Можно умножать только на кватернион")

    def rotateVec(self, vec):

        quatVec = Quaternion(np.concatenate([0.0], vecOld))

        quatVec
        return (self.getConjugate() * quatVec) * self

    def getConjugate(self):
        return Quaternion(np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]]))

