Steps:

For the development purpose, we are using UNSW_NB15 dataset (by the IXIA PerfectStorm tool. Australian Centre for Cyber <br/>
Security (ACCS)) available in the official Website. [DOWNLOAD](https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS)

Why using this Dataset:
This dataset has nine types of attacks: <br/> 
* Fuzzers:
  * Description: Fuzzers involve sending random, invalid, or unexpected data to a program to discover vulnerabilities or cause it to crash.
  * Example: Using a fuzzer tool to input random data into a web application's form fields to identify input validation weaknesses.
* Analysis:
  * Description: Analysis attacks focus on intercepting, monitoring, and analyzing network traffic to gather information and uncover vulnerabilities.
  * Example: Using a packet sniffer like Wireshark to capture and analyze network packets to discover unencrypted sensitive information.
* Backdoors:
  * Description: Backdoors are methods or software that provide unauthorized access to a system, bypassing normal authentication mechanisms.
  * Example: A hacker installing a backdoor on a compromised server, which allows them to access the server at any time without going through the normal login process.
* DoS:
  * Description: DoS attacks aim to make a network service unavailable by overwhelming it with a flood of illegitimate requests.
  * Example: Sending a huge number of requests to a web server to exhaust its resources, making it unable to serve legitimate users (e.g., via a tool like LOIC - Low Orbit Ion Cannon).
* Exploits:
  * Description: Exploits are attacks that take advantage of specific vulnerabilities in software or systems to gain unauthorized access or cause harm.
  * Example: Using a publicized vulnerability in a specific version of a web server software (like Apache Struts vulnerability CVE-2017-5638) to execute arbitrary code.
* Generic:
  * Example: Using a publicized vulnerability in a specific version of a web server software (like Apache Struts vulnerability CVE-2017-5638) to execute arbitrary code.
  * Example: Brute forcing login credentials using a tool that tries a large number of possible combinations until it finds the correct one.
* Reconnaissance:
  * Description: Reconnaissance attacks involve gathering information about a target system or network to identify potential vulnerabilities.
  * Example: Performing a network scan using tools like Nmap to discover live hosts, open ports, and running services in preparation for a potential attack.
* Shellcode:
  * Description: Shellcode is a small piece of code used as a payload in the exploitation of software vulnerabilities, often leading to a remote code execution.
  * Example: Injecting a malicious shellcode payload into a vulnerable buffer overflow in an application to gain control over the system.
* Worms:
  * Description: Worms are standalone malware that replicate themselves to spread to other computers, often causing harm by consuming bandwidth or overloading systems.
  * Example: The "ILOVEYOU" worm, which spread via email attachments, infecting millions of computers by sending copies of itself to a person's contact list.


Undeerstanding the data:
This dataset contents 45 nos. of column entry.
1. *id*: Unique identifier for each network packet/session.
2. dur: Duration of the network session in seconds.
3. proto: Protocol used (e.g., TCP, UDP).
4. service: Intended service on the destination port (e.g., HTTP, FTP).
5. state: State of the network connection (e.g., FIN, CON).
6. spkts: Number of packets sent by the source.
7. dpkts: Number of packets received by the source.
8. sbytes: Number of bytes sent by the source.
9. dbytes: Number of bytes received by the source.
10. rate: Transmission rate in packets per second.
11. sttl: Source Time-To-Live value.
12. dttl: Destination Time-To-Live value.
13. sload: Load of the source in bits per second.
14. dload: Load of the destination in bits per second.
15. sloss: Number of packets lost by the source.
16. dloss: Number of packets lost by the destination.
17. sinpkt: Inter-arrival time of packets sent by the source.
18. dinpkt: Inter-arrival time of packets received by the source.
19. sjit: Jitter in packet transmission by the source.
20. djit: Jitter in packet reception at the destination.
21. swin: Source TCP window size.
22. stcpb: Source TCP sequence number.
23. dtcpb: Destination TCP sequence number.
24. dwin: Destination TCP window size.
25. tcprtt: Round Trip Time for TCP connections.
26. synack: SYN-ACK time interval in TCP.
27. ackdat: ACK data time interval in TCP.
28. smean: Average packet size sent by the source.
29. dmean: Average packet size received by the destination.
30. trans_depth: HTTP transaction depth in HTTP packets.
31. response_body_len: Length of the response body in HTTP.
32. ct_srv_src: Number of connections from the same source IP on the same service.
33. ct_state_ttl: Number of connections with same state and Time-To-Live.
34. ct_dst_ltm: Number of connections to the same destination IP.
35. ct_src_dport_ltm: Number of connections from same source IP to the same destination port.
36. ct_dst_sport_ltm: Number of connections from the same source port to the same destination port.
37. ct_dst_src_ltm: Number of connections between the same source and destination IP.
38. is_ftp_login: Indicates if the FTP login was successful.
39. ct_ftp_cmd: Number of FTP commands.
40. ct_flw_http_mthd: Number of HTTP methods.
41. ct_src_ltm: Number of connections from the same source IP.
42. ct_srv_dst: Number of connections to the same destination IP on the same service.
43. is_sm_ips_ports: Indicator if source and destination IP/port pairs are identical in all connections.
44. attack_cat: Category of the attack, if any (e.g., Normal, DoS, BruteForce).
45. label: Boolean indicator (0 or 1) where 1 implies an attack and 0 implies normal behavior.



# Data preprocessing
* load the data
* Data cleaning
  * check there is null data entry. 
  * If so, drop the colume or fill the data 
  ``` python
  data.dropna(inplace=True)
  data.drop_duplicates(inplace=True)
* get all the object data in column
  ```python
  df_object_column = data.dtypes[data.dtypes == "object"].index.tolist()
  print(df_object_column)




To determine which features contribute most to the detected anomalies, follow these steps:

Feature Importance Analysis: Use techniques to analyze the importance of each feature in the anomaly detection process.

Reconstruction Error Analysis: For models like autoencoders or GANs, analyze reconstruction errors to understand feature contributions.

Correlation Analysis: Examine correlations between features and anomaly scores.

Here's a step-by-step approach you can use to analyze feature importance for your dataset:

1. Feature Importance Analysis
   Using Reconstruction Error
   If you used an autoencoder or GAN, the reconstruction error for each feature can give insights into its importance. However, since you're using a GAN, focus on the discriminator's predictions.

Using Feature Perturbation
Perturb Features: Temporarily modify each feature in the dataset and observe changes in the anomaly scores.
Measure Impact: Calculate how much the anomaly score changes with each feature perturbation.
2. Correlation Analysis
   Calculate correlation coefficients between features and the anomaly scores. Features with higher correlation with the anomaly scores are likely more important.

3. Visualizing Feature Contributions
   Use visualization techniques to identify which features contribute most to anomalies
