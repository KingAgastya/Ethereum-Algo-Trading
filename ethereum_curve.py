import pandas as pd
import matplotlib.pyplot as plt

# === Load your CSV ===
# Replace with your file path
file_path = "ETH-USDT_1h.csv"

df = pd.read_csv(file_path)

# === Inspect columns (run once if unsure) ===
# print(df.columns)

# === Ensure correct column names ===
# Common names: 'timestamp', 'date', 'time', 'close'
# Adjust if needed
df['timestamp'] = pd.to_datetime(df['timestamp'])

# === Sort by time (important) ===
df = df.sort_values('timestamp')

# === Plot ===
plt.figure()
plt.plot(df['timestamp'], df['close'])

plt.title("Ethereum Closing Price (3 Years)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)

plt.tight_layout()
output_path = "eth_close_plot.png"  # change name/path if needed
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"Plot saved to {output_path}")

plt.show()