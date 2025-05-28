import requests

url = "https://ab42-202-166-164-218.ngrok-free.app/run-auto-categorization"
data = {"message": "Hello from my friend"}

response = requests.post(url, json=data)
print(response.json())  # Prints the server's response
