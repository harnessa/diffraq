"""
test_rotations.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-12-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test of rotation matrices

"""

import diffraq
import numpy as np
from scipy.spatial.transform import Rotation as R

class Test_Rotations(object):

    def test_all(self):

        #Check known rotations
        ax = self.xRot(np.pi/2).dot(np.eye(3))
        ay = self.yRot(np.pi/2).dot(np.eye(3))
        az = self.zRot(np.pi/2).dot(np.eye(3))
        assert(np.allclose(ax, [[1, 0, 0],[0, 0, -1], [0, 1, 0]]))
        assert(np.allclose(ay, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]]))
        assert(np.allclose(az, [[0, -1, 0],[1, 0, 0], [0, 0, 1]]))

    ############################################

        #Starting points
        xyz0 = np.random.uniform(-20, 20, size=3)

        #Rotation angles (yaw, pitch, roll) (x,y,z)
        ypr = np.random.uniform(0, 2*np.pi, size=3)

        #Check full matrix is correct
        xr = self.xRot(ypr[0])
        yr = self.yRot(ypr[1])
        zr = self.zRot(ypr[2])
        M1 = xr @ yr @ zr
        M2 = self.build_full_rot_matrix(*ypr)

        assert(np.allclose(M1, M2))

    ############################################

        #Check dot product
        a1 = (M2 @ xyz0[:,None])[:,0]
        a2 = M2.dot(xyz0)
        a3 = R.from_matrix(M2).apply(xyz0)
        assert(np.allclose(a1, a2))
        assert(np.allclose(a2, a3))

        #Check large dot product
        big_xyz0 = np.random.uniform(-20, 20, size=(20, 3))
        a4 = big_xyz0.dot(M2.T)
        a5 = M2.dot(big_xyz0.T).T
        a6 = np.array([M2.dot(x) for x in big_xyz0])
        a7 = np.array([M2 @ x for x in big_xyz0])
        assert(np.allclose(a4, a5))
        assert(np.allclose(a5, a6))
        assert(np.allclose(a6, a7))

        #Check rotation while returning split (fast when x,y supplied separate)
        x0, y0, z0 = big_xyz0.T
        ax, ay, az = M2.dot(np.stack((x0, y0,z0),0))
        assert(np.allclose(ax, a4[:,0]))
        assert(np.allclose(ay, a4[:,1]))
        assert(np.allclose(az, a4[:,2]))

        #Edge rotation
        xy = np.stack((x0,y0),1)
        ae = np.hstack((xy, z0[:,None])).dot(M2.T)[:,:2]
        assert(np.allclose(ax, ae[:,0]))
        assert(np.allclose(ay, ae[:,1]))

    ############################################

        #Check that negative rotation is same as not included transpose (for old rotations)
        a8 = big_xyz0.dot(self.zRot(ypr[2]).T)      #New way
        a9 = big_xyz0.dot(self.zRot(-ypr[2]))       #Old way
        assert(np.allclose(a8, a9))

    ############################################

        #Get scipy rotation (extrinsic Euler)
        Me = R.from_euler('zyx', ypr[::-1])

        #Check matrices are the same
        assert(np.allclose(Me.as_matrix(), M2))

        #Cleanup
        del big_xyz0, xyz0, x0, y0, z0, ax, ay, az, xy
        del a1, a2, a3, a4, a5, a6, a7, a8, a9

############################################

    def build_full_rot_matrix(self, yaw, pit, rol):
        """Extrinsic (stationary coordinate frame) clockwise rotation about
            Z-Y-X (yaw-pitch-roll)"""
        xr = self.xRot(yaw)
        yr = self.yRot(pit)
        zr = self.zRot(rol)
        full_rot_mat = np.linalg.multi_dot((xr, yr, zr))

        return full_rot_mat

    def xRot(self, ang):
        """Clockwise rotation of X axis"""
        return np.array([[1,0,0], [0, np.cos(ang), -np.sin(ang)], \
            [0, np.sin(ang), np.cos(ang)]])

    def yRot(self, ang):
        """Clockwise rotation of Y axis"""
        return np.array([[np.cos(ang), 0.,  np.sin(ang)], [0,1,0], \
            [-np.sin(ang), 0, np.cos(ang)]])

    def zRot(self, ang):
        """Clockwise rotation of Z axis"""
        return np.array([[np.cos(ang), -np.sin(ang), 0.], \
            [np.sin(ang), np.cos(ang), 0], [0,0,1]])

############################################

if __name__ == '__main__':

    tt = Test_Rotations()
    tt.test_all()
