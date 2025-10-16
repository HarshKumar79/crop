import requests

def get_rainfall(state, year, season):
    coords = {
        "Assam": (26.2, 92.9),
        "Uttar Pradesh": (26.8, 80.9),
        "Maharashtra": (19.7, 75.7),
        "Tamil Nadu": (10.8, 78.7),
    }
    lat, lon = coords.get(state, (20.6, 78.9))
    url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={year}-01-01&end_date={year}-12-31&daily=precipitation_sum&timezone=Asia%2FKolkata"
    try:
        res = requests.get(url)
        data = res.json()
        vals = data["daily"]["precipitation_sum"]
        return sum(vals) / len(vals)
    except Exception:
        return 1000
