import os
import io
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import streamlit as st

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_gdrive():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            # This will open a browser window for you to log in
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

def download_csv_from_drive(file_id):
    """Downloads a CSV file from Drive and returns it as a Pandas DataFrame."""
    try:
        service = authenticate_gdrive()
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            # print(f"Download {int(status.progress() * 100)}%.")
            
        file_stream.seek(0)
        df = pd.read_csv(file_stream)
        return df
        
    except Exception as e:
        st.error(f"Google Drive API Error: {e}")
        return None

def extract_file_id(url):
    """Extracts the file ID from a standard Google Drive shareable link."""
    # Example link: https://drive.google.com/file/d/1aBcD_eFgHiJkLmNoPqRsTuVwXyZ/view?usp=sharing
    try:
        if "id=" in url:
            return url.split("id=")[1]
        elif "/d/" in url:
            return url.split("/d/")[1].split("/")[0]
        return url
    except:
        return None
