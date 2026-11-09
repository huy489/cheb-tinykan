"""
Colab Sync Tool for Cheb-TinyKAN.
Uses Google Service Account to automatically fetch model checkpoints and metrics
from Google Drive folder 'ChebTinyKAN_Results' into local 'checkpoints/' directory.
"""

import os
import json
import time
import jwt
import urllib.request
import urllib.parse

KEY_PATH = "/home/pd/hcmus_queue/lab_dientuysinh/agent-colab/topic1-cheb-d57210f40434.json"
RESULTS_FOLDER_ID = "1OnUpCcLCOhKLsaTgDGYXz6wGsMrya1qI"
LOCAL_CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def get_drive_access_token():
    if not os.path.exists(KEY_PATH):
        raise FileNotFoundError(f"Service account key not found at: {KEY_PATH}")

    with open(KEY_PATH) as f:
        key_data = json.load(f)

    now = int(time.time())
    payload = {
        "iss": key_data["client_email"],
        "sub": key_data["client_email"],
        "aud": "https://oauth2.googleapis.com/token",
        "iat": now,
        "exp": now + 3600,
        "scope": "https://www.googleapis.com/auth/drive"
    }

    signed_jwt = jwt.encode(payload, key_data["private_key"], algorithm="RS256")
    data = urllib.parse.urlencode({
        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
        "assertion": signed_jwt
    }).encode("utf-8")

    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def sync_results():
    os.makedirs(LOCAL_CHECKPOINTS_DIR, exist_ok=True)
    print("🔑 Authenticating with Google Cloud Service Account...")
    token = get_drive_access_token()

    print(f"📡 Querying Google Drive folder 'ChebTinyKAN_Results' ({RESULTS_FOLDER_ID})...")
    q = urllib.parse.quote(f"'{RESULTS_FOLDER_ID}' in parents and trashed = false")
    url = f"https://www.googleapis.com/drive/v3/files?q={q}&fields=files(id,name,size,modifiedTime)"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})

    with urllib.request.urlopen(req) as resp:
        files = json.loads(resp.read().decode("utf-8")).get("files", [])

    if not files:
        print("ℹ️ Thư mục 'ChebTinyKAN_Results' trên Google Drive hiện chưa có file nào.")
        print("   (Colab đang huấn luyện hoặc chưa lưu checkpoint vào Drive).")
        return

    print(f"Found {len(files)} file(s) in Drive:")
    for fl in files:
        file_id = fl["id"]
        file_name = fl["name"]
        size_kb = int(fl.get("size", 0)) / 1024
        local_path = os.path.join(LOCAL_CHECKPOINTS_DIR, file_name)

        print(f"  ⬇️ Downloading: {file_name} ({size_kb:.1f} KB)...", end=" ")
        dl_req = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
            headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(dl_req) as dl_resp:
            content = dl_resp.read()
            with open(local_path, "wb") as f_out:
                f_out.write(content)
        print("✅ Done!")

    print(f"\n🎉 Đồng bộ hoàn tất! Các file đã nằm tại: {LOCAL_CHECKPOINTS_DIR}/")


if __name__ == "__main__":
    sync_results()
