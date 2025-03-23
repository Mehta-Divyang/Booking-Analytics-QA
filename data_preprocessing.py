import pandas as pd

df = pd.read_csv("E:/Projects/Project4/booking_analytics_project/hotel_bookings.csv")

print(df.info())
print(df.head())

print(df.isnull().sum())

df.fillna({"children": 0, "country": "Unknown"}, inplace=True)
df.dropna(subset=["hotel", "arrival_date_year", "adr"], inplace=True)

df["arrival_date"] = pd.to_datetime(df["arrival_date_year"].astype(str) + "-" +
                                    df["arrival_date_month"] + "-" +
                                    df["arrival_date_day_of_month"].astype(str))

df["adr"] = df["adr"].astype(float) 

df.to_csv("E:/Projects/Project4/booking_analytics_project/cleaned_bookings.csv", index=False)
print("Data cleaned and saved successfully!")
