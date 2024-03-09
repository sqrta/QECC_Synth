import re

x="""Top Assembly Part Number        : 800-25858-06
Top Assembly Revision Number    : A0
Version ID                      : V08
CLEI Code Number                : COMDE10BRA
Hardware Board Revision Number  : 0x01


Switch   Ports  Model              SW Version              SW Image
------   -----  -----              ----------              ----------
*    1   52     WS-C3750-48P       12.2(35)SE5             C3750-IPBASE-M
     2   52     WS-C3750-48P       12.2(35)SE5             C3750-IPBASE-M
     3   52     WS-C3750-48P       12.2(35)SE5             C3750-IPBASE-M
     4   52     WS-C3750-48P       12.2(35)SE5             C3750-IPBASE-M


Switch 02
---------
Switch Uptime                   : 11 weeks, 2 days, 16 hours, 27 minutes
Base ethernet MAC Address       : 00:26:52:96:2A:80
Motherboard assembly number     : 73-9675-15"""
print(re.findall("^\*?\s*(\d)\s*\d+\s*([A-Z\d-]+)",x,re.MULTILINE))

with open("local_generated_codes/test/gen/gen_round1_n0.txt", 'r') as f:
    code = f.read()
    # code = "def priority(edge: [int, int, int, int]) -> float:   \nsfdsfds    return score"
    valid = re.findall(r"def priority\(edge: \[int, int, int, int\]\) -> float:.*\n    return score", code,re.DOTALL)
    print(valid[0])