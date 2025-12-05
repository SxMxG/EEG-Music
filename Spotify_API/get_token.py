from requests import post
import base64
import os
import json
# Example: Fetching data from a public API
client_id = "Client_id"
client_secret = "Client_secret"

def get_token():
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes),"utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization" : f"Basic {auth_base64}",
        "Content-Type" : "application/x-www-form-urlencoded"
    }
    data = {"grant_type" : "client_credentials"}
    result = post(url,headers=headers,data=data)
    json_result = json.loads(result.content)
    print(json_result)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": f"Bearer {token}"}


token = get_token()
print(token)
print(get_auth_header())



# response = requests.get("https://api.example.com/data")
# if response.status_code == 200:
#     data = response.json()  # Parse JSON response
#     print(data)
# else:
#     print(f"Error: {response.status_code}")