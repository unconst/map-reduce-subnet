import pexpect as pe
import json
import requests
import re
import time

# speedtest using speedtest cli
def speedtest():
    speedtest = pe.spawn('speedtest -f json', encoding='utf-8')
    index = speedtest.expect(['{"type":"result".*', pe.EOF, pe.TIMEOUT], timeout=30)  # waiting 30 seconds for the prompt
    if index > 0:
        print("Error", speedtest.before, speedtest.after)
        return None
    return json.loads(speedtest.after)
  
def verify_speedtest_result(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Use regular expression to find the INIT_DATA in the page content
        match = re.search(r"window\.OOKLA\.INIT_DATA\s*=\s*({.*?});", response.text, re.DOTALL)

        if match:
            # Extract the JSON data
            json_data = match.group(1)
            data = json.loads(json_data)
            return data
        else:
            print("Data not found in the page.")
            return None

    except requests.RequestException as e:
        if e.args[0][:3] == '429':
            print("Too many requests. Waiting few seconds...")
            time.sleep(6)
            return verify_speedtest_result(url)
        print(f"Error fetching the URL: {e}")
        return None

if __name__ == "__main__":
    # result = speedtest()
    
    # print("Timestamp: ", result['timestamp'])
    # print("Download: ", round(result['download']['bandwidth'] * 8 / 1000000, 2), "Mbps")
    # print("Upload: ", round(result['upload']['bandwidth'] * 8 / 1000000, 2), "Mbps")
    # print("Ping: ", result['ping']['latency'], "ms")
    # print("URL: ", result['result']['url'])
    
    # print(result)
    
    # Fetch and extract data
    url = "https://www.speedtest.net/result/c/81cd0ce5-caca-4bf5-b0e4-4ad835c2e498"
    while True:
        time.sleep(0.1)
        data = verify_speedtest_result(url)
        if data:
            print(json.dumps(data, indent=2))
