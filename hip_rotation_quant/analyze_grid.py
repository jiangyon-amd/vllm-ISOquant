#!/usr/bin/env python3
"""Analyze grid dimensions and kernel launch overhead."""

# Current kernel configuration:
# - BLOCK_M = 32 rows per block
# - RS = 128 cols per rotation segment
# - Grid: ((M + 31) / 32, K / 128)

# For K=2048, N_I=64:
# - Grid Y = 2048 / 128 = 16 blocks in Y dimension
# - For M=1: Grid = (1, 16) = 16 blocks total
# - For M=32: Grid = (1, 16) = 16 blocks total

# The issue: For small M, we have many small blocks
# Each block loads the same rotation matrix (32KB)

# Potential optimization: Process multiple rotation segments per block
# This would reduce the number of kernel launches and rotation matrix loads

print("Grid analysis:")
print("=" * 60)

K = 2048
RS = 128
BLOCK_M = 32

for M in [1, 2, 4, 8, 16, 32]:
    grid_x = (M + BLOCK_M - 1) // BLOCK_M
    grid_y = K // RS
    total_blocks = grid_x * grid_y
    print(f"M={M:2d}: Grid=({grid_x}, {grid_y}) = {total_blocks} blocks")

print()
print("Observation: For M<=32, grid_x=1, so we have 16 blocks")
print("Each block processes 32 rows x 128 cols")
print("Total work per block: 32 * 128 = 4096 elements")
print()
print("Potential optimization: Increase BLOCK_M to reduce grid_x")
print("Or: Process multiple RS segments per block")
