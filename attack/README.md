# IDS Attack Simulation Lab

## Purpose
Generate controlled network traffic between isolated attacker and receiver
VMs to evaluate the flow-based IDS.

## Workflow
1. Verify the attacker and receiver VM addresses.
2. Confirm both VMs are on an isolated lab network.
3. Start packet capture on the receiver or monitoring interface.
4. Generate benign traffic and controlled test traffic.
5. Stop capture and save the PCAP in `attack/pcaps/`.
6. Convert the PCAP to flow features using CICFlowMeter.
7. Run IDS predictions and review the results.

## Directories
- scripts/: traffic generation and test scripts
- configs/: lab configuration
- logs/: experiment logs
- pcaps/: packet captures

Only generate test traffic against VMs and services you are authorized
to test. Do not run experiments against public or third-party systems.

## Current scope
The IDS currently uses batch processing. Captures are processed after
the experiment; this is not yet continuous real-time detection or blocking.
