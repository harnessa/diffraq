import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

num_pts = 512
wave = 0.641e-6
width = 5e-3

#Source coordinates
dx = width / num_pts
xx = (np.arange(num_pts) - num_pts/2) * dx

lens = diffraq.diffraction.Lens_Element(\
    {'lens_name':'AC508-150-A-ML', 'diameter':width}, num_pts)

focal_length = lens.focal_length

# rr = np.hypot(xx, xx[:,None])
rr = xx

opd1 = rr**2/(2*focal_length)
opd2 = lens.lens_func(rr)

plt.plot(xx, opd1)
plt.plot(xx, opd2)

breakpoint()
