# Test script to verify that matplotlib, seaborn, and IPython are installed correctly

# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
import IPython

# Print the versions to verify installation
print(f"matplotlib version: {matplotlib.__version__}")
print(f"seaborn version: {sns.__version__}")
print(f"IPython version: {IPython.__version__}")

# Create a simple plot to test matplotlib and seaborn
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a plot
sns.lineplot(x=x, y=y, marker='o')
plt.title('Test Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Save the plot to verify it works
plt.savefig('test_plot.png')
print("Plot saved as 'test_plot.png'")

print("All libraries are working correctly!")
