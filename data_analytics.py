import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_bookings.csv")

df["arrival_date"] = pd.to_datetime(df["arrival_date"])

df["year_month"] = df["arrival_date"].dt.to_period("M")

monthly_revenue = df.groupby("year_month")["adr"].sum()

plt.figure(figsize=(10, 5))
monthly_revenue.plot(kind="line", marker="o", color="b")
plt.title("Revenue Trends Over Time")
plt.xlabel("Month")
plt.ylabel("Total Revenue (€)")
plt.grid(True)
plt.show()

cancellation_rate = df["is_canceled"].mean() * 100
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

country_counts = df["country"].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=country_counts.index, y=country_counts.values, palette="viridis")
plt.title("Top 10 Countries with Most Bookings")
plt.xlabel("Country")
plt.ylabel("Number of Bookings")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df["lead_time"], bins=50, kde=True)
plt.title("Distribution of Booking Lead Time")
plt.xlabel("Days Before Arrival")
plt.ylabel("Number of Bookings")
plt.show()