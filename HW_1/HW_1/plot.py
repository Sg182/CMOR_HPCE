import numpy as np
import matplotlib.pyplot as plt

# Load data (skip header line)
rec = np.loadtxt("runtime_recursive.txt", skiprows=1)
blk = np.loadtxt("runtime_blocked.txt", skiprows=1)
nai = np.loadtxt("runtime_naive.txt", skiprows=1)

# Extract columns
n_rec, t_rec = rec[:,0], rec[:,1]
n_blk, t_blk = blk[:,0], blk[:,1]
n_nai, t_nai = nai[:,0], nai[:,1]

# Plot
plt.figure()

plt.plot(n_rec, t_rec, marker='o', label="Recursive")
plt.plot(n_blk, t_blk, marker='s', label="Blocked")
plt.plot(n_nai, t_nai, marker='^', label="Naive")

plt.xlabel("Matrix Size (n)")
plt.ylabel("Runtime (seconds)")
plt.title("Matrix Transpose Runtime vs Matrix size")
plt.legend()
#plt.grid(True)
plt.savefig('runtime.pdf',dpi=300, bbox_inches='tight')


plt.show()
