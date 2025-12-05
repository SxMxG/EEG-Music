import requests
import base64
import os
import json
# Example: Fetching data from a public API
client_id = "client_id" #change this
redirect_uri = "http://127.0.0.1:8888/callback"
scope = "user-read-playback-state app-remote-control"
url = "https://accounts.spotify.com/api/token"
params = {
    'response_type': 'authorization_code',
    'client_id': client_id,
    'redirect_uri': redirect_uri,
    'scope': scope,
}
# data = {"grant_type" : "client_credentials"}
result = requests.get(url,params=params)
print(result.json)
# def get_token():
#     auth_string = f"client_id:{client_id}"
#     scope = "user-read-playback-state app-remote-control"
#     url = "https://accounts.spotify.com/authorize?"
#     params = {
#         'response_type': 'code',
#         'client_id': client_id,
#         'redirect_uri': redirect_uri,
#         'scope': scope,
#     }
#     data = {"grant_type" : "client_credentials"}
#     result = requests.post(url,data=data,params=params)
#     json_result = result.json
#     print(json_result)
#     # token = json_result["access_token"]
#     return token
# def get_auth_header(token):
#     return {"Authorization": f"Bearer {token}"}
# def get_player_info(auth_header):
#     url = "https://api.spotify.com/v1/me/player"
#     result = requests.get(url,headers=auth_header)
#     data = result.json
#     return data
# token = get_token()
# print(token)
# auth = get_auth_header(token)
# print(auth)
# player = get_player_info(auth)
# print(player)

# # response = requests.get("https://api.example.com/data")
# # if response.status_code == 200:
# #     data = response.json()  # Parse JSON response
# #     print(data)
# # else:
# #     print(f"Error: {response.status_code}")