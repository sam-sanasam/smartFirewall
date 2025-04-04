import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')



def generate_firewall_rules(json_file_path):

    load_dotenv()

    openai.api_key = os.getenv('OPENAI_API_KEY')


    # Check if the API key is loaded correctly
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

    # Load JSON data
    with open(json_file_path, 'r') as json_file:
        anomaly_data = json.load(json_file)

    # Format data for GenAI model
    prompt = f"Generate firewall rules to mitigate the following anomalies:\n{json.dumps(anomaly_data, indent=4)}"

    prompt1 = (
        "Based on the following network traffic anomalies, generate firewall rules in "
        "the format of 'iptables' to block or allow traffic. Ensure the rules are specific "
        "and actionable, using details such as source bytes (sbytes), destination bytes (dbytes), "
        "rate, and TTL (Time to Live). Use common firewall rule syntax:\n"
        f"{json.dumps(anomaly_data, indent=4)}"
    )

    # Format data for GenAI model
    prompt3 = (
        "Based on the following network anomalies, generate sophisticated firewall rules to mitigate them. "
        "Consider specific IP address ranges, protocols, port numbers, thresholds, and conditions for each rule:\n\n"
        f"{json.dumps(anomaly_data, indent=4)}\n\n"
        "For each anomaly, provide firewall rules that specify:\n"
        "- IP address ranges (source and destination)\n"
        "- Protocol (TCP, UDP, ICMP, etc.)\n"
        "- Port numbers (source and destination)\n"
        "- Thresholds for bytes, rates, and packet counts\n"
        "- Conditions for blocking or allowing traffic\n"
        "- Any specific timing or frequency-based rules\n"
        "Format the rules in a way that can be directly applied to firewall configurations using iptables or similar tools."
    )

    prompt_final = (
        "Based on the following network anomalies, generate sophisticated firewall rules to mitigate them. "
        "Consider the following aspects for each rule:\n\n"
        "- IP address ranges (source and destination)\n"
        "- Protocols (TCP, UDP, ICMP, etc.)\n"
        "- Port numbers (source and destination)\n"
        "- Thresholds for bytes, rates, and packet counts\n"
        "- Conditions for blocking or allowing traffic\n"
        "- Any specific timing or frequency-based rules\n\n"
        "Here are the details of the detected anomalies:\n\n"
        f"{json.dumps(anomaly_data, indent=4)}\n\n"
        "Generate firewall rules in the iptables format or similar, suitable for deployment. "
        "Include rules for:\n"
        "- Blocking or allowing traffic based on IP address and port\n"
        "- Rate limiting to prevent DoS attacks\n"
        "- Geolocation-based IP blocking (if applicable)\n"
        "- User-based access control\n"
        "- Logging and monitoring of suspicious traffic\n"
        "- Content filtering (if applicable)\n"
        "- DDoS protection\n"
        "Format the rules so they can be directly applied to firewall configurations. And provide the details explaination for each rule"
    )




    # Interact with GenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a security expert."},
            {"role": "user", "content": prompt_final}
        ],
        max_tokens=150,
        temperature=0.7
    )

    # Extract and print generated rules
    rules = response.choices[0].message['content'].strip()
    print("Generated Firewall Rules:")
    print(rules)

    return rules


if __name__ == '__main__':
    json_file_path = "/path-to-the-data/data.json"
    generate_firewall_rules(json_file_path)
