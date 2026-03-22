import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# AXPY data
# ----------------------------
n = 100000000
I_axpy = 2.0 / 24.0   # flop/byte

T1 = 0.020286
T2 = 0.018886
T8 = 0.017705

P1 = (2.0 * n) / T1 / 1e9
P2 = (2.0 * n) / T2 / 1e9
P8 = (2.0 * n) / T8 / 1e9

# ----------------------------
# STREAM-like measured bandwidths
# ----------------------------
B1 = 115.579
B2 = 123.381
B8 = 125.687

# Large compute peak just to show the memory roofs clearly
P_peak = 500.0

I = np.logspace(-3, 2, 500)

def roofline(I, B, P_peak):
    return np.minimum(B * I, P_peak)

R1 = roofline(I, B1, P_peak)
R2 = roofline(I, B2, P_peak)
R8 = roofline(I, B8, P_peak)

fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(I, R1, linewidth=2, label=f'Roof 1 thread (B={B1:.1f} GB/s)')
ax.loglog(I, R2, linewidth=2, label=f'Roof 2 threads (B={B2:.1f} GB/s)')
ax.loglog(I, R8, linewidth=2, label=f'Roof 8 threads (B={B8:.1f} GB/s)')

ax.scatter([I_axpy], [P1], s=90, zorder=5, label=f'AXPY 1 thread ({P1:.2f} GFLOP/s)')
ax.scatter([I_axpy], [P2], s=90, zorder=5, label=f'AXPY 2 threads ({P2:.2f} GFLOP/s)')
ax.scatter([I_axpy], [P8], s=90, zorder=5, label=f'AXPY 8 threads ({P8:.2f} GFLOP/s)')

ax.set_xlabel('Arithmetic Intensity [flop/byte]')
ax.set_ylabel('Performance [GFLOP/s]')
ax.set_title('Roofline Plot for AXPY')
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend(fontsize=9)

plt.savefig("roofline_plot.png", dpi=300, bbox_inches="tight")
plt.show()

roof1 = B1 * I_axpy
roof2 = B2 * I_axpy
roof8 = B8 * I_axpy

print(f"I_axpy = {I_axpy:.6f} flop/byte")
print(f"1 thread: AXPY = {P1:.3f} GFLOP/s, roof = {roof1:.3f} GFLOP/s, %peak = {100*P1/roof1:.1f}%")
print(f"2 threads: AXPY = {P2:.3f} GFLOP/s, roof = {roof2:.3f} GFLOP/s, %peak = {100*P2/roof2:.1f}%")
print(f"8 threads: AXPY = {P8:.3f} GFLOP/s, roof = {roof8:.3f} GFLOP/s, %peak = {100*P8/roof8:.1f}%")