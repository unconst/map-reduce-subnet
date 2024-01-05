import pexpect as pe
import json
import requests
import re

# speedtest using speedtest cli
def speedtest():
    speedtest = pe.spawn('speedtest -f json', encoding='utf-8')
    index = speedtest.expect(['{"type":"result".*', pe.EOF, pe.TIMEOUT], timeout=30)  # waiting 30 seconds for the prompt
    if index > 0:
        print("Timeout")
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
        print(f"Error fetching the URL: {e}")
        return None

if __name__ == "__main__":
    result = speedtest()
    
    print("Timestamp: ", result['timestamp'])
    print("Download: ", round(result['download']['bandwidth'] * 8 / 1000000, 2), "Mbps")
    print("Upload: ", round(result['upload']['bandwidth'] * 8 / 1000000, 2), "Mbps")
    print("Ping: ", result['ping']['latency'], "ms")
    print("URL: ", result['result']['url'])
    
    print(result)
    
    # Fetch and extract data
    data = verify_speedtest_result(result['result']['url'])

    if data:
        print(json.dumps(data, indent=2))
