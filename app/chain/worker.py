import requests

class Worker:
    def __init__(self, worker_url, worker_port):
        self.worker_url = worker_url
        self.worker_port = worker_port

    def process_request(self, request_data):
        return requests.post(url=f"http://{self.worker_url}:{self.worker_port}/send_message/", json=request_data)