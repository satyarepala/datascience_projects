import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in km
    distance = R * c
    return distance

# Example usage:
lat1, lon1 = 13.0827, 80.2707  # Chennai, India
lat2, lon2 = 12.9716, 77.5946  # Bangalore, India

print("Distance:", haversine(lat1, lon1, lat2, lon2), "km")