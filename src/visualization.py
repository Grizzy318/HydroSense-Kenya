import matplotlib.pyplot as plt
import seaborn as sns

def rainfall_trend(weather):
    plt.figure(figsize=(10, 5))
    plt.plot(weather["date"], weather["rainfall_mm"])
    plt.title("Rainfall Trend")
    plt.xlabel("Date")
    plt.ylabel("Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def temperature_trend(weather):
    plt.figure(figsize=(10, 5))
    plt.plot(weather["date"], weather["temperature_c"])
    plt.title("Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def soil_moisture_boxplot(soil):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=soil, x="zone_id", y="soil_moisture_pct")
    plt.title("Soil Moisture Distribution")
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.tight_layout()
    plt.show()
