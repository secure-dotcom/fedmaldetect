#!/usr/bin/env python3
"""
nabiot_capture_reordered.py

Capture live IoT traffic, compute N-BaIoT’s features, and write out a continuous CSV
stream of feature vectors in the specified custom order, using rolling 15-minute intervals.
"""

import time
import csv
import threading
import math
import sys
import netifaces
import statistics
from collections import deque
from scapy.all import sniff, Packet

# --- Configuration ----------------------------------------------------------

INTERFACE = "eth0"                 # Network interface to sniff
OUTPUT_CSV = "features.csv"
ROTATION_INTERVAL = 15 * 60        # 15 minutes in seconds

# Define custom labeled windows (seconds)
# L5: 60s, L3: 10s, L1: 1.5s, L0.1: 0.5s, L0.01: 0.1s
WINDOWS = [60.0, 10.0, 1.5, 0.5, 0.1]
LABELS = ['L5', 'L3', 'L1', 'L0.1', 'L0.01']

# Your device identifiers (adjust as needed)
try:
    iface_info = netifaces.ifaddresses(INTERFACE)
    MY_DEVICE_IP = iface_info[netifaces.AF_INET][0]['addr']
    MY_DEVICE_MAC = iface_info[netifaces.AF_LINK][0]['addr']
except Exception as e:
    print(f"[Error] Could not get IP/MAC for interface '{INTERFACE}': {e}")
    sys.exit(0)

print(f"[INFO] Monitoring interface: {INTERFACE}")
print(f"[INFO] Detected IP:  {MY_DEVICE_IP}")
print(f"[INFO] Detected MAC: {MY_DEVICE_MAC}")

# --- Prepare CSV Header -----------------------------------------------------
fields = []
# MI features
for lbl in LABELS:
    fields += [
        f"MI_dir_{lbl}_weight", f"MI_dir_{lbl}_mean", f"MI_dir_{lbl}_variance"
    ]
# H features
for lbl in LABELS:
    fields += [
        f"H_{lbl}_weight", f"H_{lbl}_mean", f"H_{lbl}_variance"
    ]
# HH features
for lbl in LABELS:
    fields += [
        f"HH_{lbl}_weight", f"HH_{lbl}_mean", f"HH_{lbl}_std",
        f"HH_{lbl}_magnitude", f"HH_{lbl}_radius", f"HH_{lbl}_covariance", f"HH_{lbl}_pcc"
    ]
# HH jitter features
for lbl in LABELS:
    fields += [
        f"HH_jit_{lbl}_weight", f"HH_jit_{lbl}_mean", f"HH_jit_{lbl}_variance"
    ]
# HpHp features
for lbl in LABELS:
    fields += [
        f"HpHp_{lbl}_weight", f"HpHp_{lbl}_mean", f"HpHp_{lbl}_std",
        f"HpHp_{lbl}_magnitude", f"HpHp_{lbl}_radius", f"HpHp_{lbl}_covariance", f"HpHp_{lbl}_pcc"
    ]

header = [
    "timestamp", "flow_src_ip", "flow_dst_ip", "flow_src_port", "flow_dst_port", "flow_proto"
] + fields

# --- Data Structures --------------------------------------------------------
buffers = {
    w: {
        'H': deque(), 'MI': deque(),
        'HH': deque(), 'HH_jit': deque(),
        'HpHp': deque()
    }
    for w in WINDOWS
}

# --- Utility Functions ------------------------------------------------------

def extract_keys(pkt: Packet):
    ts = time.time()
    size = len(pkt)
    dirn = 'out' if pkt.haslayer('IP') and pkt['IP'].src == MY_DEVICE_IP else 'in'

    ip_src = pkt['IP'].src if pkt.haslayer('IP') else None
    ip_dst = pkt['IP'].dst if pkt.haslayer('IP') else None
    sport  = pkt['TCP'].sport if pkt.haslayer('TCP') else pkt['UDP'].sport if pkt.haslayer('UDP') else None
    dport  = pkt['TCP'].dport if pkt.haslayer('TCP') else pkt['UDP'].dport if pkt.haslayer('UDP') else None
    proto  = pkt['IP'].proto if pkt.haslayer('IP') else None
    mac_src = pkt.src if hasattr(pkt, 'src') else None

    H_key    = ip_src
    MI_key   = (mac_src, ip_src)
    HH_key   = (ip_src, ip_dst)
    HpHp_key = (ip_src, sport, ip_dst, dport)

    return ts, size, dirn, H_key, MI_key, HH_key, HpHp_key, ip_src, ip_dst, sport, dport, proto


def update_buffers(pkt_info):
    ts, size, dirn, Hk, MIk, HHk, Spk, *_ = pkt_info
    for w in WINDOWS:
        # update packet deques
        for stream, key in [('H', Hk), ('MI', MIk), ('HH', HHk), ('HpHp', Spk)]:
            dq = buffers[w][stream]
            dq.append((ts, size, dirn))
            while dq and dq[0][0] < ts - w:
                dq.popleft()
        # update jitter
        dqj = buffers[w]['HH_jit']
        dqj.append((ts, HHk))
        while dqj and dqj[0][0] < ts - w:
            dqj.popleft()


def stats_stream(dq):
    sizes = [sz for ts, sz, d in dq]
    cnt = len(sizes)
    mean = statistics.mean(sizes) if sizes else 0.0
    var  = statistics.pvariance(sizes) if len(sizes) > 1 else 0.0
    std  = math.sqrt(var)
    return cnt, mean, var, std


def dir_stats(dq):
    outs = [sz for ts, sz, d in dq if d=='out']
    ins  = [sz for ts, sz, d in dq if d=='in']
    cnt = len(outs)+len(ins)
    mean = statistics.mean(outs+ins) if (outs+ins) else 0.0
    var  = statistics.pvariance(outs+ins) if len(outs+ins)>1 else 0.0
    std  = math.sqrt(var)
    return cnt, mean, var, std


def jitter_stats(dq):
    times = [ts for ts, key in dq]
    diffs = [t2 - t1 for t1, t2 in zip(times, times[1:])]
    cnt = len(diffs)
    mean = statistics.mean(diffs) if diffs else 0.0
    var  = statistics.pvariance(diffs) if len(diffs)>1 else 0.0
    return cnt, mean, var


def combined_metrics(dq):
    in_s  = [sz for ts, sz, d in dq if d=='in']
    out_s = [sz for ts, sz, d in dq if d=='out']
    mag = math.hypot(statistics.mean(in_s) if in_s else 0.0,
                     statistics.mean(out_s) if out_s else 0.0)
    rad = math.hypot(statistics.pvariance(in_s) if len(in_s)>1 else 0.0,
                     statistics.pvariance(out_s) if len(out_s)>1 else 0.0)
    try:
        cov = statistics.pcovariance(in_s, out_s)
        corr = statistics.pearsonr(in_s, out_s)[0]
    except Exception:
        cov = corr = 0.0
    return mag, rad, cov, corr

# --- CSV Rotation -----------------------------------------------------------

def rotate_csv():
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    threading.Timer(ROTATION_INTERVAL, rotate_csv).start()

rotate_csv()

# --- Packet Processing ------------------------------------------------------

def process_packet(pkt):
    pkt_info = extract_keys(pkt)
    update_buffers(pkt_info)

    ts, _, _, _, _, _, _, ip_src, ip_dst, sport, dport, proto = pkt_info
    row = [ts, ip_src, ip_dst, sport, dport, proto]

    for lbl, w in zip(LABELS, WINDOWS):
        # MI directional stats
        mi_deque = buffers[w]['MI']
        cnt, mean, var, _ = dir_stats(mi_deque)
        row += [cnt, mean, var]
    for lbl, w in zip(LABELS, WINDOWS):
        # H directional stats
        h_deque = buffers[w]['H']
        cnt, mean, var, _ = dir_stats(h_deque)
        row += [cnt, mean, var]
    for lbl, w in zip(LABELS, WINDOWS):
        # HH combined stats
        hh_deque = buffers[w]['HH']
        cnt, mean, var, std = stats_stream(hh_deque)
        mag, rad, cov, corr = combined_metrics(hh_deque)
        row += [cnt, mean, std, mag, rad, cov, corr]
    for lbl, w in zip(LABELS, WINDOWS):
        # HH jitter
        j_deque = buffers[w]['HH_jit']
        cnt, mean, var = jitter_stats(j_deque)
        row += [cnt, mean, var]
    for lbl, w in zip(LABELS, WINDOWS):
        # HpHp combined stats
        hp_deque = buffers[w]['HpHp']
        cnt, mean, var, std = stats_stream(hp_deque)
        mag, rad, cov, corr = combined_metrics(hp_deque)
        row += [cnt, mean, std, mag, rad, cov, corr]

    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

if __name__ == "__main__":
    print(f"Starting reordered capture on {INTERFACE}, output: {OUTPUT_CSV}")
    sniff(iface=INTERFACE, prn=process_packet, store=False)
