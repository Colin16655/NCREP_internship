import matplotlib.pyplot as plt

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.imshow([[1, 2], [7, 8]], cmap='viridis')

# Optionally, add titles, labels, etc.
ax.set_title("MAC Matrix")
ax.set_xlabel("Mode Frequency [Hz]")
ax.set_ylabel("Mode Frequency [Hz]")

# Ensure tight layout
fig.tight_layout()

# Explicitly render the canvas
fig.canvas.draw()

# Save the figure
fig.savefig("MAC_matrix.png")

# Optionally show the plot
plt.show()

# Close the figure to free up resources
plt.close(fig)
