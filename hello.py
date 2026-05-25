# Smart Glasses AI - by Keshav Goyal

objects_detected = ["chair", "laptop", "person", "bottle", "phone"]

for object in objects_detected:
    if object == "person":
        print("⚠️  ALERT: Person detected!")
    else:
        print("I can see a:", object)