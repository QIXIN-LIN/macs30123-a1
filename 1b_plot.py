# Plot the data
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('mpi_times.csv')

# Plot
plt.plot(df['Cores'], df['Time'], marker='o')
plt.xlabel('Number of Cores')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. Number of Cores')
plt.grid(True)
plt.show()